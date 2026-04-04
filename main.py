"""variant-bench — CLI entry point.

Run ``variant-bench --help`` for usage information.
"""

from __future__ import annotations

import sys
from datetime import datetime
from importlib.resources import files as _pkg_files
from pathlib import Path
from typing import Any

import click
from rich.console import Console

from benchmark.hardware import detect_hardware, enrich_with_gpu_layers
from benchmark.quality import QualityScorer
from benchmark.reporter import Reporter
from benchmark.runner import BenchmarkRunner, OllamaConnectionError

console = Console()


def _load_prompts(path: str) -> list[str]:
    """Read prompts from *path*, one per line.

    Lines that are blank or start with ``#`` are skipped.

    Resolution order:
    1. The path as given (absolute, or relative to the current working directory).
    2. Bundled package data under ``benchmark/prompts/`` — used when the tool
       is installed via pip and the prompts directory is not on disk.

    Args:
        path: Path to the prompt file.

    Returns:
        List of non-empty, non-comment prompt strings.

    Raises:
        SystemExit: If the file cannot be found by either method.
    """
    p = Path(path)
    content: str | None = None

    if p.exists():
        content = p.read_text(encoding="utf-8")
    else:
        try:
            pkg_file = _pkg_files("benchmark.prompts").joinpath(p.name)
            content = pkg_file.read_text(encoding="utf-8")
        except Exception:
            pass

    if content is None:
        console.print(f"[bold red]Error:[/bold red] Prompt file not found: {path}")
        sys.exit(1)

    prompts: list[str] = []
    for line in content.splitlines():
        stripped = line.strip()
        if stripped and not stripped.startswith("#"):
            prompts.append(stripped)

    if not prompts:
        console.print(
            f"[bold red]Error:[/bold red] No prompts found in {path}. "
            "Check that the file has at least one non-blank, non-comment line."
        )
        sys.exit(1)

    return prompts


@click.command()
@click.option(
    "--family",
    default=None,
    help=(
        "Model family prefix to benchmark all pulled variants of "
        "(e.g. 'qwen2.5' matches qwen2.5:3b, qwen2.5:7b, qwen2.5:14b). "
        "Mutually exclusive with --models."
    ),
)
@click.option(
    "--models",
    multiple=True,
    help=(
        "Explicit list of model tags to benchmark "
        "(e.g. --models qwen2.5:3b --models llama3:8b). "
        "Mutually exclusive with --family."
    ),
)
@click.option(
    "--runs",
    default=3,
    show_default=True,
    help="Runs per prompt to average. The first run is a warmup and is discarded.",
)
@click.option(
    "--prompts",
    default="prompts/factual.txt",
    show_default=True,
    help="Path to a prompt file (one prompt per line).",
)
@click.option(
    "--format",
    "output_format",
    default="all",
    show_default=True,
    help="Output format: table | json | chart | all",
)
@click.option(
    "--output",
    default="results",
    show_default=True,
    help="Directory for result files.",
)
@click.option(
    "--context-sweep",
    "context_sweep",
    is_flag=True,
    default=False,
    help=(
        "Run each prompt at multiple input context sizes (512, 2048, 4096 tokens) "
        "and produce a quality-vs-context-length chart."
    ),
)
def main(
    family: str | None,
    models: tuple[str, ...],
    runs: int,
    prompts: str,
    output_format: str,
    output: str,
    context_sweep: bool,
) -> None:
    """Benchmark Ollama model variants by size and family.

    Answers: which model variant gives the best quality-per-second tradeoff
    on your hardware?

    Use --family to benchmark all pulled variants of a model family, or
    --models to benchmark an explicit list of model tags.
    """
    if family and models:
        console.print(
            "[bold red]Error:[/bold red] --family and --models are mutually exclusive. "
            "Use one or the other."
        )
        sys.exit(1)

    if not family and not models:
        console.print(
            "[bold red]Error:[/bold red] Specify either --family <name> or "
            "--models <tag> [--models <tag> …]."
        )
        sys.exit(1)

    # ------------------------------------------------------------------ #
    # 1. Ollama connection check.                                         #
    # ------------------------------------------------------------------ #
    runner = BenchmarkRunner()

    console.print("[bold cyan]variant-bench[/bold cyan] — Model Variant Benchmarking Tool")
    console.print(f"Connecting to Ollama at {runner.base_url} …")

    try:
        runner.check_connection()
    except OllamaConnectionError as exc:
        console.print(f"[bold red]Connection error:[/bold red] {exc}")
        sys.exit(1)

    console.print("[green]Ollama is reachable.[/green]\n")

    # ------------------------------------------------------------------ #
    # 2. Hardware detection.                                              #
    # ------------------------------------------------------------------ #
    hw = detect_hardware()
    console.print(f"Hardware detected: [bold]{hw.label}[/bold]\n")

    # ------------------------------------------------------------------ #
    # 3. Resolve model list.                                              #
    # ------------------------------------------------------------------ #
    if family:
        model_list = runner.list_models(family=family)
        if not model_list:
            console.print(
                f"[bold red]Error:[/bold red] No pulled models found matching "
                f"family prefix '{family}'. Run 'ollama list' to see what's available."
            )
            sys.exit(1)
        console.print(
            f"Family [cyan]{family}[/cyan] — found "
            f"[bold]{len(model_list)}[/bold] variant(s): "
            + ", ".join(m["name"] for m in model_list)
        )
    else:
        model_list = [{"name": tag, "params": ""} for tag in models]
        console.print(
            f"Benchmarking [bold]{len(model_list)}[/bold] model(s): "
            + ", ".join(m["name"] for m in model_list)
        )

    # ------------------------------------------------------------------ #
    # 4. Load prompts.                                                    #
    # ------------------------------------------------------------------ #
    prompt_list = _load_prompts(prompts)
    console.print(
        f"Loaded [bold]{len(prompt_list)}[/bold] prompt(s) from [cyan]{prompts}[/cyan]."
    )
    console.print(
        f"Runs per prompt: [bold]{runs}[/bold] "
        f"(run 1 is warmup; averaging runs 2–{runs}).\n"
    )

    scorer = QualityScorer()
    run_id = datetime.now().isoformat(timespec="seconds")
    reporter = Reporter(output_dir=output)

    # ------------------------------------------------------------------ #
    # 5a. Context-sweep mode.                                             #
    # ------------------------------------------------------------------ #
    if context_sweep:
        console.print(
            "[bold cyan]Context-sweep mode:[/bold cyan] running each prompt at "
            "512, 2048, and 4096 input tokens.\n"
        )
        all_sweep_results: list[dict[str, Any]] = []

        for model_info in model_list:
            model_name = model_info["name"]
            console.print(f"Sweeping [cyan]{model_name}[/cyan] …")
            with console.status("Running sweep across 3 context sizes…"):
                sweep = runner.run_context_sweep(model_name, prompt_list, runs)
            all_sweep_results.extend(sweep)
            console.print(f"  [green]Done.[/green]")

        if not all_sweep_results:
            console.print("[bold red]No results collected. Exiting.[/bold red]")
            sys.exit(1)

        console.rule("[bold]Quality Scoring[/bold]")
        all_sweep_results = scorer.score_sweep_results(all_sweep_results)
        console.print("[green]Quality scoring complete.[/green]\n")

        want_table = output_format in ("table", "all")
        want_json = output_format in ("json", "all")
        want_chart = output_format in ("chart", "all")

        if want_table:
            console.rule("[bold]Context Sweep Results[/bold]")
            reporter.print_context_sweep_table(all_sweep_results, hw_label=hw.label)

        if want_json or want_chart:
            console.rule("[bold]Saving Files[/bold]")

        if want_json:
            reporter.save_json(run_id, all_sweep_results)

        if want_chart:
            reporter.save_context_sweep_chart(all_sweep_results)

        if output_format == "table":
            reporter.save_json(run_id, all_sweep_results)

        console.print(
            f"\n[bold green]Context sweep complete.[/bold green] "
            f"Results written to [cyan]{output}/[/cyan]"
        )
        return

    # ------------------------------------------------------------------ #
    # 5b. Standard benchmark mode.                                        #
    # ------------------------------------------------------------------ #
    all_results: list[dict[str, Any]] = []

    for model_info in model_list:
        model_name = model_info["name"]
        console.rule(f"[bold yellow]{model_name}[/bold yellow]")

        # Enrich hardware info with GPU layer count from /api/show.
        show_data = runner.get_model_details(model_name)
        enrich_with_gpu_layers(hw, show_data)

        console.print(f"Benchmarking [cyan]{model_name}[/cyan] …")
        with console.status(f"Running {runs} × {len(prompt_list)} prompt(s)…"):
            result = runner.run_benchmark(model_name, prompt_list, runs)
        all_results.append(result)

        avg_ttft = sum(
            p["avg_ttft_ms"] for p in result["prompts"]
        ) / len(result["prompts"])
        console.print(f"  [green]Done.[/green] Avg TTFT: {avg_ttft:.1f} ms")

    if not all_results:
        console.print("[bold red]No results collected. Exiting.[/bold red]")
        sys.exit(1)

    # ------------------------------------------------------------------ #
    # 6. Quality scoring.                                                 #
    # ------------------------------------------------------------------ #
    console.rule("[bold]Quality Scoring[/bold]")
    baseline = scorer.pick_baseline(all_results)
    console.print(
        f"Baseline model (largest): [yellow]{baseline.get('name', '?')}[/yellow]"
    )
    all_results = scorer.score_results(all_results, baseline=baseline)
    console.print("[green]Quality scoring complete.[/green]\n")

    # ------------------------------------------------------------------ #
    # 7. Output.                                                          #
    # ------------------------------------------------------------------ #
    want_table = output_format in ("table", "all")
    want_json = output_format in ("json", "all")
    want_chart = output_format in ("chart", "all")

    if want_table:
        console.rule("[bold]Results Table[/bold]")
        reporter.print_table(all_results, hw_label=hw.label)

    if want_json or want_chart:
        console.rule("[bold]Saving Files[/bold]")

    if want_json:
        reporter.save_json(run_id, all_results)
        reporter.save_markdown(all_results)

    if want_chart:
        reporter.save_chart(all_results)

    if output_format == "table":
        reporter.save_json(run_id, all_results)
        reporter.save_markdown(all_results)

    console.print(
        f"\n[bold green]Benchmark complete.[/bold green] "
        f"Results written to [cyan]{output}/[/cyan]"
    )


if __name__ == "__main__":
    main()
