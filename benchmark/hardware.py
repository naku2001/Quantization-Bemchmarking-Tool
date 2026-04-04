"""Hardware detection — GPU/CPU discovery reported at benchmark startup."""

from __future__ import annotations

import subprocess
from dataclasses import dataclass, field
from typing import Any


@dataclass
class HardwareInfo:
    """Detected hardware profile for the current machine.

    Attributes:
        device: Human-readable device label, e.g. ``"NVIDIA RTX 3080"`` or
            ``"CPU"``.
        vram_mb: Total VRAM in megabytes, or ``0`` for CPU-only systems.
        gpu_layers: Number of model layers offloaded to GPU as reported by
            Ollama's ``/api/show`` response, or ``None`` if unavailable.
        backend: One of ``"nvidia"``, ``"amd"``, or ``"cpu"``.
    """

    device: str
    vram_mb: int
    gpu_layers: int | None
    backend: str

    @property
    def label(self) -> str:
        """Short display label used in table output."""
        if self.backend == "cpu":
            return "CPU"
        vram_gb = self.vram_mb / 1024
        layers = f" · {self.gpu_layers}L" if self.gpu_layers is not None else ""
        return f"{self.device} ({vram_gb:.1f} GB VRAM{layers})"


def _run(cmd: list[str]) -> str | None:
    """Run *cmd* and return stdout, or ``None`` on any error."""
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0:
            return result.stdout.strip()
    except Exception:
        pass
    return None


def _detect_nvidia() -> HardwareInfo | None:
    """Return NVIDIA GPU info, or ``None`` if no NVIDIA GPU is found."""
    out = _run([
        "nvidia-smi",
        "--query-gpu=name,memory.total",
        "--format=csv,noheader",
    ])
    if not out:
        return None

    # Take the first GPU line.
    first_line = out.splitlines()[0].strip()
    parts = [p.strip() for p in first_line.split(",")]
    if len(parts) < 2:
        return None

    name = parts[0]
    # memory.total comes as "8192 MiB" — extract the integer.
    vram_str = parts[1].split()[0]
    try:
        vram_mb = int(vram_str)
    except ValueError:
        vram_mb = 0

    return HardwareInfo(device=name, vram_mb=vram_mb, gpu_layers=None, backend="nvidia")


def _detect_amd() -> HardwareInfo | None:
    """Return AMD GPU info via rocm-smi, or ``None`` if unavailable."""
    out = _run(["rocm-smi", "--showmeminfo", "vram"])
    if not out:
        return None

    vram_mb = 0
    device_name = "AMD GPU"
    for line in out.splitlines():
        line_lower = line.lower()
        # Look for a VRAM total line, e.g. "  VRAM Total Memory (B): 8589934592"
        if "vram total" in line_lower and "b)" in line_lower:
            try:
                vram_bytes = int(line.split(":")[-1].strip())
                vram_mb = vram_bytes // (1024 * 1024)
            except ValueError:
                pass

    return HardwareInfo(device=device_name, vram_mb=vram_mb, gpu_layers=None, backend="amd")


def detect_hardware() -> HardwareInfo:
    """Detect the best available compute device.

    Checks for NVIDIA GPU first, then AMD, then falls back to CPU.

    Returns:
        A :class:`HardwareInfo` instance describing the detected hardware.
    """
    nvidia = _detect_nvidia()
    if nvidia is not None:
        return nvidia

    amd = _detect_amd()
    if amd is not None:
        return amd

    return HardwareInfo(device="CPU", vram_mb=0, gpu_layers=None, backend="cpu")


def enrich_with_gpu_layers(
    hw: HardwareInfo,
    model_show_response: dict[str, Any],
) -> HardwareInfo:
    """Parse Ollama's ``/api/show`` response and attach ``gpu_layers`` to *hw*.

    Args:
        hw: Existing :class:`HardwareInfo` to update.
        model_show_response: Parsed JSON dict from ``POST /api/show``.

    Returns:
        The same *hw* object with ``gpu_layers`` set if the field is present.
    """
    details: dict[str, Any] = model_show_response.get("details", {})
    layers = details.get("num_gpu_layers")
    if layers is not None:
        try:
            hw.gpu_layers = int(layers)
        except (TypeError, ValueError):
            pass
    return hw
