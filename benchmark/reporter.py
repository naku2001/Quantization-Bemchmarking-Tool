"""All output — terminal, JSON, Markdown, and chart — is produced here.

Nothing outside this module should write to disk or render structured output.
"""

from __future__ import annotations

import json
import os
from typing import Any

import matplotlib.pyplot as plt
from rich.console import Console
from rich.table import Table

_console = Console()


def _model_averages(model: dict[str, Any]) -> dict[str, float]:
    """Compute per-model averages across all prompts.

    Args:
        model: A single model result dict.

    Returns:
        Dict with ``avg_ttft_ms``, ``avg_tokens_per_sec``,
        ``avg_similarity_score``, ``avg_rouge_l_score``, and ``avg_ram_gb``.
    """
    prompts: list[dict[str, Any]] = model.get("prompts", [])
    if not prompts:
        return {
            "avg_ttft_ms": 0.0,
            "avg_tokens_per_sec": 0.0,
            "avg_similarity_score": 0.0,
            "avg_rouge_l_score": 0.0,
            "avg_ram_gb": 0.0,
        }

    ttft_values = [p.get("avg_ttft_ms", 0.0) for p in prompts]
    tps_values = [p.get("avg_tokens_per_sec", 0.0) for p in prompts]
    sim_values = [p.get("similarity_score", 0.0) for p in prompts]
    rouge_values = [p.get("rouge_l_score", 0.0) for p in prompts]

    ram_values: list[float] = []
    for p in prompts:
        runs: list[dict[str, Any]] = p.get("runs", [])
        if runs:
            ram_values.append(runs[-1].get("ram_gb", 0.0))

    return {
        "avg_ttft_ms": sum(ttft_values) / len(ttft_values),
        "avg_tokens_per_sec": sum(tps_values) / len(tps_values),
        "avg_similarity_score": sum(sim_values) / len(sim_values) if sim_values else 0.0,
        "avg_rouge_l_score": sum(rouge_values) / len(rouge_values) if rouge_values else 0.0,
        "avg_ram_gb": sum(ram_values) / len(ram_values) if ram_values else 0.0,
    }


def _pareto_pick(results: list[dict[str, Any]]) -> dict[str, Any] | None:
    """Return the model with the best tokens/sec × similarity_score product.

    Args:
        results: List of model result dicts with scored prompts.

    Returns:
        The result dict of the recommended model, or ``None`` if *results*
        is empty.
    """
    if not results:
        return None
    return max(
        results,
        key=lambda r: (
            _model_averages(r)["avg_tokens_per_sec"]
            * _model_averages(r)["avg_similarity_score"]
        ),
    )


class Reporter:
    """Handles all output for a benchmark run.

    Args:
        output_dir: Directory where result files will be written.
            Created automatically if it does not exist.
    """

    def __init__(self, output_dir: str = "results") -> None:
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

    # ------------------------------------------------------------------
    # Terminal output
    # ------------------------------------------------------------------

    def print_table(
        self,
        results: list[dict[str, Any]],
        hw_label: str = "CPU",
    ) -> None:
        """Print a rich table summarising benchmark results.

        Columns: Model | Params | TTFT (ms) | Tokens/sec | Similarity |
                 ROUGE-L | RAM (GB) | Hardware

        Args:
            results: List of model result dicts.
            hw_label: Hardware label string from
                :attr:`~benchmark.hardware.HardwareInfo.label`.
        """
        table = Table(
            title="[bold]Model Variant Benchmark Results[/bold]",
            show_header=True,
            header_style="bold magenta",
        )
        table.add_column("Model", style="cyan", no_wrap=True)
        table.add_column("Params", style="yellow")
        table.add_column("TTFT (ms)", justify="right")
        table.add_column("Tokens/sec", justify="right")
        table.add_column("Similarity", justify="right")
        table.add_column("ROUGE-L", justify="right")
        table.add_column("RAM (GB)", justify="right")
        table.add_column("Hardware", style="dim")

        recommended = _pareto_pick(results)

        for model in results:
            avgs = _model_averages(model)
            name = model.get("name", "—")
            is_recommended = recommended is not None and model is recommended
            name_display = f"[bold green]{name} ★[/bold green]" if is_recommended else name

            table.add_row(
                name_display,
                model.get("params", "—"),
                f"{avgs['avg_ttft_ms']:.1f}",
                f"{avgs['avg_tokens_per_sec']:.2f}",
                f"{avgs['avg_similarity_score']:.3f}",
                f"{avgs['avg_rouge_l_score']:.3f}",
                f"{avgs['avg_ram_gb']:.2f}",
                hw_label,
            )

        _console.print(table)

    def print_context_sweep_table(
        self,
        sweep_results: list[dict[str, Any]],
        hw_label: str = "CPU",
    ) -> None:
        """Print a rich table summarising context-sweep results.

        Columns: Model | Params | Context (tokens) | TTFT (ms) | Tokens/sec |
                 Similarity | ROUGE-L | Hardware

        Args:
            sweep_results: List of result dicts with a ``"context_size"`` key.
            hw_label: Hardware label string.
        """
        table = Table(
            title="[bold]Context Sweep Results[/bold]",
            show_header=True,
            header_style="bold magenta",
        )
        table.add_column("Model", style="cyan", no_wrap=True)
        table.add_column("Params", style="yellow")
        table.add_column("Context (tokens)", justify="right")
        table.add_column("TTFT (ms)", justify="right")
        table.add_column("Tokens/sec", justify="right")
        table.add_column("Similarity", justify="right")
        table.add_column("ROUGE-L", justify="right")
        table.add_column("Hardware", style="dim")

        sorted_results = sorted(
            sweep_results,
            key=lambda r: (r.get("name", ""), r.get("context_size", 0)),
        )

        for result in sorted_results:
            avgs = _model_averages(result)
            table.add_row(
                result.get("name", "—"),
                result.get("params", "—"),
                str(result.get("context_size", "—")),
                f"{avgs['avg_ttft_ms']:.1f}",
                f"{avgs['avg_tokens_per_sec']:.2f}",
                f"{avgs['avg_similarity_score']:.3f}",
                f"{avgs['avg_rouge_l_score']:.3f}",
                hw_label,
            )

        _console.print(table)

    # ------------------------------------------------------------------
    # File output
    # ------------------------------------------------------------------

    def save_json(self, run_id: str, results: list[dict[str, Any]]) -> None:
        """Write the full benchmark results to ``results/results.json``.

        Args:
            run_id: ISO-8601 timestamp string identifying this run.
            results: List of model result dicts.
        """
        output = {"run_id": run_id, "models": results}
        path = os.path.join(self.output_dir, "results.json")
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(output, fh, indent=2, ensure_ascii=False)
        _console.print(f"[green]JSON saved →[/green] {path}")

    def save_markdown(self, results: list[dict[str, Any]]) -> None:
        """Write a Markdown summary report to ``results/report.md``.

        Includes a table of averaged metrics per model and a **Recommended**
        line identifying the best speed/quality (Pareto) model.

        Args:
            results: List of model result dicts.
        """
        lines: list[str] = []
        lines.append("# Model Variant Benchmark Report\n")
        lines.append(
            "| Model | Params | Avg TTFT (ms) | Avg Tokens/sec "
            "| Avg Similarity | Avg ROUGE-L |"
        )
        lines.append(
            "|-------|--------|---------------|----------------|"
            "----------------|-------------|"
        )

        for model in results:
            avgs = _model_averages(model)
            name = model.get("name", "—")
            params = model.get("params", "—")
            lines.append(
                f"| {name} | {params} | {avgs['avg_ttft_ms']:.1f} "
                f"| {avgs['avg_tokens_per_sec']:.2f} "
                f"| {avgs['avg_similarity_score']:.3f} "
                f"| {avgs['avg_rouge_l_score']:.3f} |"
            )

        lines.append("")
        recommended = _pareto_pick(results)
        if recommended:
            rec_avgs = _model_averages(recommended)
            lines.append(
                f"**Recommended:** `{recommended.get('name', '?')}` "
                f"({recommended.get('params', '?')}) — best quality-per-second ratio "
                f"({rec_avgs['avg_tokens_per_sec']:.2f} tok/s · "
                f"{rec_avgs['avg_similarity_score']:.3f} similarity)."
            )
        else:
            lines.append("**Recommended:** No results to evaluate.")

        path = os.path.join(self.output_dir, "report.md")
        with open(path, "w", encoding="utf-8") as fh:
            fh.write("\n".join(lines) + "\n")
        _console.print(f"[green]Markdown saved →[/green] {path}")

    def save_chart(self, results: list[dict[str, Any]]) -> None:
        """Save a scatter plot of tokens/sec vs similarity score.

        Each point represents one model variant, labelled with the model name
        and parameter count.  The recommended model is highlighted.
        Saved to ``results/chart.png``.

        Args:
            results: List of model result dicts.
        """
        fig, ax = plt.subplots(figsize=(10, 6))
        recommended = _pareto_pick(results)

        for model in results:
            avgs = _model_averages(model)
            x = avgs["avg_tokens_per_sec"]
            y = avgs["avg_similarity_score"]
            name = model.get("name", "?")
            params = model.get("params", "?")
            label = f"{name} ({params})"
            is_rec = recommended is not None and model is recommended

            color = "#e67e22" if is_rec else "#2980b9"
            marker = "*" if is_rec else "o"
            size = 180 if is_rec else 80

            ax.scatter(x, y, s=size, color=color, marker=marker, zorder=5)
            ax.annotate(
                label,
                (x, y),
                textcoords="offset points",
                xytext=(8, 4),
                fontsize=8,
                fontweight="bold" if is_rec else "normal",
            )

        ax.set_xlabel("Tokens / sec", fontsize=12)
        ax.set_ylabel("Similarity Score (cosine)", fontsize=12)
        ax.set_title("Model Variant: Speed vs Quality Tradeoff", fontsize=14)
        ax.set_ylim(0.0, 1.05)
        ax.grid(True, linestyle="--", alpha=0.5)

        # Legend entry for recommended marker
        from matplotlib.lines import Line2D
        legend_elements = [
            Line2D([0], [0], marker="*", color="w", markerfacecolor="#e67e22",
                   markersize=12, label="Recommended (best tok/s × similarity)"),
            Line2D([0], [0], marker="o", color="w", markerfacecolor="#2980b9",
                   markersize=8, label="Other variants"),
        ]
        ax.legend(handles=legend_elements, fontsize=9)

        plt.tight_layout()
        path = os.path.join(self.output_dir, "chart.png")
        fig.savefig(path, dpi=150)
        plt.close(fig)
        _console.print(f"[green]Chart saved →[/green] {path}")

    def save_context_sweep_chart(self, sweep_results: list[dict[str, Any]]) -> None:
        """Save a line chart of similarity score vs input context length.

        One line per model variant. Saved to ``results/context_sweep.png``.

        Args:
            sweep_results: List of result dicts with a ``"context_size"`` key.
        """
        series: dict[str, dict[int, float]] = {}

        for result in sweep_results:
            label = f"{result.get('name', '?')} ({result.get('params', '?')})"
            ctx = result.get("context_size", 0)
            prompt_entries = result.get("prompts", [])
            if not prompt_entries:
                continue
            sim_scores = [p.get("similarity_score", 0.0) for p in prompt_entries]
            avg_sim = sum(sim_scores) / len(sim_scores)
            if label not in series:
                series[label] = {}
            series[label][ctx] = avg_sim

        fig, ax = plt.subplots(figsize=(10, 6))

        for label, ctx_to_sim in series.items():
            xs = sorted(ctx_to_sim.keys())
            ys = [ctx_to_sim[x] for x in xs]
            ax.plot(xs, ys, marker="o", label=label)

        ax.set_xlabel("Input Context Size (tokens)", fontsize=12)
        ax.set_ylabel("Similarity Score (cosine)", fontsize=12)
        ax.set_title("Similarity Score vs Input Context Length by Model Variant", fontsize=14)
        ax.set_ylim(0.0, 1.05)
        ax.legend(fontsize=9)
        ax.grid(True, linestyle="--", alpha=0.5)

        plt.tight_layout()
        path = os.path.join(self.output_dir, "context_sweep.png")
        fig.savefig(path, dpi=150)
        plt.close(fig)
        _console.print(f"[green]Context sweep chart saved →[/green] {path}")

    def save_all(self, run_id: str, results: list[dict[str, Any]]) -> None:
        """Save JSON, Markdown, and chart outputs.

        Args:
            run_id: ISO-8601 timestamp string identifying this run.
            results: List of model result dicts.
        """
        self.save_json(run_id, results)
        self.save_markdown(results)
        self.save_chart(results)
