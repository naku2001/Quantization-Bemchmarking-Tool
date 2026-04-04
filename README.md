# variant-bench

A CLI tool that benchmarks Ollama model variants (by size and family) measuring
Time to First Token, tokens/sec throughput, cosine similarity, and ROUGE-L
quality scores.  Designed to answer: **which model variant gives the best
quality-per-second tradeoff on my hardware?**

---

## Pulling Model Variants

**Ollama does not auto-download all sizes.** Each variant must be pulled
individually before benchmarking.

```bash
# Pull the variants you want to compare
ollama pull qwen2.5:3b
ollama pull qwen2.5:7b
ollama pull qwen2.5:14b

# Or mix models across families
ollama pull llama3:8b
ollama pull mistral:7b
```

To see what you have available:

```bash
ollama list
```

variant-bench queries this list at startup and filters by the family prefix you
pass with `--family`, or benchmarks the exact tags you pass with `--models`.

### Context windows and model size

Larger models generally produce higher quality output but are slower on CPU and
require more RAM.  variant-bench measures both dimensions simultaneously —
tokens/sec for speed and cosine similarity + ROUGE-L for quality — so you can
see the exact tradeoff curve for your hardware.

The **Recommended** model in the output is selected by the best
`tokens/sec × similarity` product (Pareto-optimal tradeoff).

---

## Requirements

- Python 3.10+
- [Ollama](https://ollama.com) running locally at `localhost:11434`
- At least two model variants pulled (see above)

---

## Installation

### Editable install (recommended for development)

```bash
git clone https://github.com/naku2001/Quantization-Benchmarking-Tool.git
cd Quantization-Benchmarking-Tool
pip install -e .
pip install -e ".[dev]"   # adds pytest, ruff, responses
```

### From a wheel

```bash
pip install build
python -m build
pip install dist/variant_bench-0.2.0-py3-none-any.whl
```

After either install the `variant-bench` command is available globally:

```bash
variant-bench --family qwen2.5
```

> **Note:** `sentence-transformers` downloads `all-MiniLM-L6-v2` (~90 MB)
> on first use.  A rich spinner will indicate this is in progress.

---

## Usage

```bash
# Benchmark all pulled variants of a model family
variant-bench --family qwen2.5

# Benchmark an explicit list of models
variant-bench --models qwen2.5:3b --models llama3:8b --models mistral:7b

# Use reasoning prompts, 5 runs per prompt
variant-bench --family qwen2.5 --prompts prompts/reasoning.txt --runs 5

# JSON output only
variant-bench --family qwen2.5 --format json

# Context-length sweep (quality vs input size per variant)
variant-bench --family qwen2.5 --prompts prompts/long_context.txt --context-sweep

# Write results to a custom directory
variant-bench --family qwen2.5 --output my_results/
```

### CLI Options

| Option | Default | Description |
|--------|---------|-------------|
| `--family` | — | Model family prefix. Benchmarks all pulled variants matching it. |
| `--models` | — | Explicit model tags. Repeatable. Mutually exclusive with `--family`. |
| `--runs` | `3` | Runs per prompt. Run 1 is warmup and discarded. |
| `--prompts` | `prompts/factual.txt` | Path to prompt file (one prompt per line). |
| `--format` | `all` | `table` \| `json` \| `chart` \| `all` |
| `--output` | `results/` | Directory for result files. |
| `--context-sweep` | off | Run each prompt at 512 / 2048 / 4096 input tokens. |

---

## Prompt Files

| File | Description |
|------|-------------|
| `prompts/factual.txt` | Short factual questions (fast, low variance) |
| `prompts/reasoning.txt` | Multi-step arithmetic and geometry problems |
| `prompts/creative.txt` | Open-ended generation prompts |
| `prompts/long_context.txt` | Dense 1000–2000 token passages for context sweep |

---

## Output Files

| File | Contents |
|------|----------|
| `results/results.json` | Full structured data: all raw runs, averages, metadata, timestamp |
| `results/report.md` | Markdown table + **Recommended** model line |
| `results/chart.png` | Scatter plot: tokens/sec (x) vs similarity (y), recommended model highlighted |
| `results/context_sweep.png` | Line chart: similarity (y) vs context size (x), one line per variant |

`context_sweep.png` is only produced when `--context-sweep` is passed.

---

## Terminal Table Columns

| Column | Description |
|--------|-------------|
| Model | Full Ollama model tag |
| Params | Parameter count parsed from tag (e.g. `7b`) |
| TTFT (ms) | Average time to first token |
| Tokens/sec | Average throughput |
| Similarity | Cosine similarity vs largest model baseline |
| ROUGE-L | ROUGE-L F1 vs largest model baseline |
| RAM (GB) | Process RSS at end of run |
| Hardware | Detected device (CPU / GPU name + VRAM) |

The **★** marker in the table identifies the Recommended model.

---

## How It Works

**Baseline selection:** The largest model in the benchmark set (by parameter
count parsed from the model name) is used as the quality reference.  All other
variants are scored against its responses.

**TTFT** is measured with `time.perf_counter()` before the request and again
on the first non-empty streamed chunk.

**Throughput** reads `eval_count` / `eval_duration` from Ollama's done-chunk —
no manual token counting.

**Similarity** encodes both responses with `all-MiniLM-L6-v2` and computes
cosine similarity (0–1).

**ROUGE-L** computes longest common subsequence F1 between the baseline and
candidate responses.

**Hardware detection** runs `nvidia-smi` (NVIDIA) or `rocm-smi` (AMD) at
startup.  Falls back to CPU if neither is found.  Ollama's own GPU routing is
not changed — this is reporting only.

**Recommended model** is the variant with the highest `tokens/sec × similarity`
product — the Pareto-optimal speed/quality tradeoff.

---

## Architecture

```
main.py                  CLI entry point (click)
benchmark/
  hardware.py            GPU/CPU detection (nvidia-smi, rocm-smi)
  metrics.py             Pure functions: TTFT, throughput, RAM
  runner.py              BenchmarkRunner — Ollama REST calls, timing, sweeps
  quality.py             QualityScorer — cosine similarity + ROUGE-L
  reporter.py            All output: terminal table, JSON, Markdown, charts
```

---

## Running Tests

```bash
pytest tests/ -v
```

No live Ollama instance required — HTTP calls are mocked with `responses` and
the embedding model is mocked to avoid network downloads in CI.

```bash
ruff check .   # lint
```

---

## License

MIT
