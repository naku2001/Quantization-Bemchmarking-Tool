# quant-bench — LLM Quantization Benchmarking Tool

> Benchmark Ollama models across quantization levels — measuring speed, throughput, and semantic quality degradation — entirely on CPU-only hardware.

---

## Overview

**quant-bench** is a CLI tool that runs structured benchmarks against locally served [Ollama](https://ollama.com) models at different quantization levels (Q4, Q5, Q6, Q8). For each model variant it measures:

- **Time to First Token (TTFT)** — latency from request to first streamed token, in milliseconds
- **Throughput** — tokens per second, using Ollama's native `eval_count` / `eval_duration`
- **Semantic quality score** — cosine similarity of responses relative to the highest available quant (Q8 baseline), using `all-MiniLM-L6-v2`

Results are written as a rich terminal table, structured JSON, a Markdown report, and a scatter-plot chart visualising the speed/quality tradeoff curve.

---

## Features

- Runs entirely locally — no cloud API keys required
- Designed for CPU-only hardware
- Supports multiple models in a single run
- Warmup-aware: first run per prompt is discarded, remaining runs are averaged
- Auto-discovers quantization variants from Ollama tags
- Outputs results in `table`, `json`, `chart`, or `all` formats

---

## Requirements

- Python 3.10+
- [Ollama](https://ollama.com) running locally at `localhost:11434`
- At least one model pulled with multiple quant variants (e.g. `llama3:8b-q4_K_M`, `llama3:8b-q8_0`)

---

## Installation

```bash
git clone https://github.com/naku2001/Quantization-Bemchmarking-Tool.git
cd Quantization-Bemchmarking-Tool
pip install -r requirements.txt
```

> **Note:** `sentence-transformers` will download the `all-MiniLM-L6-v2` model (~90 MB) on first run. A progress spinner will indicate this is happening.

---

## Usage

```bash
# Basic benchmark — single model, 3 runs per prompt
python main.py --model llama3

# Multiple models in one run
python main.py --model llama3 --model mistral --runs 5

# Use a different prompt category
python main.py --model llama3 --prompts prompts/reasoning.txt

# Output only JSON (skips terminal table)
python main.py --model llama3 --format json

# Specify a custom output directory
python main.py --model llama3 --output my_results/
```

### CLI Options

| Option | Default | Description |
|---|---|---|
| `--model` | *(required)* | Ollama model name. Repeatable for multiple models. |
| `--runs` | `3` | Runs per prompt. First run is warmup and is discarded. |
| `--prompts` | `prompts/factual.txt` | Path to a prompt file (one prompt per line). |
| `--format` | `all` | Output format: `table` \| `json` \| `chart` \| `all` |
| `--output` | `results/` | Directory for result files. |

---

## Project Structure

```
quant-bench/
├── main.py                   # CLI entry point (click)
├── requirements.txt
├── CLAUDE.md                 # Architecture & implementation notes
├── benchmark/
│   ├── runner.py             # Core timing logic, Ollama API calls
│   ├── metrics.py            # TTFT, throughput, memory helpers
│   ├── quality.py            # Semantic similarity scorer
│   └── reporter.py           # JSON, Markdown, and chart output
├── prompts/
│   ├── factual.txt           # Short factual questions
│   ├── reasoning.txt         # Multi-step reasoning prompts
│   └── creative.txt          # Open-ended generation prompts
├── results/                  # Auto-created on first run (gitignored)
│   ├── results.json
│   ├── report.md
│   └── chart.png
└── tests/
    ├── test_runner.py
    ├── test_metrics.py
    └── test_quality.py
```

---

## Output Formats

### Terminal Table

Displayed live after each model completes using `rich`. Columns: Model, Quant, TTFT (ms), Tokens/sec, Quality Score, RAM (GB).

### `results/results.json`

Full structured data including all raw per-run values, averaged metrics, model metadata, and timestamp.

```json
{
  "run_id": "2026-03-31T14:22:00",
  "models": [
    {
      "name": "llama3:8b-q4_K_M",
      "quant": "Q4_K_M",
      "prompts": [
        {
          "prompt": "...",
          "runs": [...],
          "avg_ttft_ms": 521,
          "avg_tokens_per_sec": 7.3,
          "quality_score": 0.961
        }
      ]
    }
  ]
}
```

### `results/report.md`

Markdown table with averaged metrics per model/quant, plus a **Key Finding** line identifying the best speed/quality ratio.

### `results/chart.png`

Scatter plot with tokens/sec on the X-axis and quality score on the Y-axis. Each point represents one model/quant combination, labelled for easy comparison.

---

## How It Works

**TTFT** is measured by streaming the Ollama `/api/generate` response and recording `time.perf_counter()` before the request and again when the first non-empty chunk arrives.

**Throughput** uses `eval_count` and `eval_duration` from Ollama's final done-chunk JSON — no manual token counting.

**Quality scoring** uses Q8 (or the highest available quant) as a baseline. Both the baseline and test responses are encoded with `all-MiniLM-L6-v2` and compared via cosine similarity. A score of `1.0` means semantically identical output.

**Quant detection** queries `/api/tags`, filters by model name prefix, and extracts the quant suffix from the tag string (e.g. `llama3:8b-q4_K_M` → `Q4_K_M`).

---

## Running Tests

```bash
pytest tests/ -v
```

Tests mock the Ollama REST API using the `responses` library — no live Ollama instance required. The quality scorer encoder is also mocked in CI to avoid the model download.

```bash
# Lint
ruff check .
```

---

## Stack

| Component | Library |
|---|---|
| CLI | `click` |
| Terminal output | `rich` |
| Ollama API | `requests` (REST, no SDK) |
| Quality scoring | `sentence-transformers` (`all-MiniLM-L6-v2`) |
| Charts | `matplotlib` |
| System metrics | `psutil` |
| Testing | `pytest`, `responses` |
| Linting | `ruff` |

---

## Notes

- Ollama must be running before invoking the CLI. The tool checks connectivity at startup with a `GET /api/tags` call and exits with a clear error if it fails.
- On CPU-only hardware, sentence-transformers scoring is slow. Quality scoring runs once per prompt (not per run) to keep total benchmark time manageable.
- The `results/` directory is gitignored. Raw JSON results are always saved even when `--format table` is specified, so no data is lost.

---

## License

MIT
