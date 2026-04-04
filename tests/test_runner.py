"""Unit tests for benchmark/runner.py.

Uses the `responses` library to mock HTTP calls so no live Ollama instance
is required.
"""

from __future__ import annotations

import json

import pytest
import responses as responses_mock

from benchmark.runner import (
    BenchmarkRunner,
    OllamaConnectionError,
    _resize_prompt,
    parse_param_count,
)

BASE_URL = "http://localhost:11434"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_ndjson(*chunks: dict) -> bytes:
    """Encode a sequence of dicts as newline-delimited JSON bytes."""
    return b"\n".join(json.dumps(c).encode() for c in chunks)


# ---------------------------------------------------------------------------
# parse_param_count
# ---------------------------------------------------------------------------

class TestParseParamCount:
    """Tests for the parse_param_count helper."""

    def test_extracts_integer_billions(self) -> None:
        assert parse_param_count("qwen2.5:7b") == pytest.approx(7.0)

    def test_extracts_large_model(self) -> None:
        assert parse_param_count("llama3:70b-instruct") == pytest.approx(70.0)

    def test_extracts_fractional_billions(self) -> None:
        assert parse_param_count("qwen2.5:0.5b") == pytest.approx(0.5)

    def test_returns_zero_when_no_match(self) -> None:
        assert parse_param_count("mistral:latest") == pytest.approx(0.0)

    def test_case_insensitive(self) -> None:
        assert parse_param_count("MODEL:14B") == pytest.approx(14.0)


# ---------------------------------------------------------------------------
# check_connection
# ---------------------------------------------------------------------------

class TestCheckConnection:
    """Tests for BenchmarkRunner.check_connection."""

    @responses_mock.activate
    def test_success(self) -> None:
        """A 200 response from /api/tags should not raise."""
        responses_mock.add(
            responses_mock.GET,
            f"{BASE_URL}/api/tags",
            json={"models": []},
            status=200,
        )
        runner = BenchmarkRunner(base_url=BASE_URL)
        runner.check_connection()

    @responses_mock.activate
    def test_connection_error_raises(self) -> None:
        """A connection error should raise OllamaConnectionError."""
        responses_mock.add(
            responses_mock.GET,
            f"{BASE_URL}/api/tags",
            body=ConnectionError("refused"),
        )
        runner = BenchmarkRunner(base_url=BASE_URL)
        with pytest.raises(OllamaConnectionError, match="Cannot connect"):
            runner.check_connection()

    @responses_mock.activate
    def test_timeout_raises(self) -> None:
        """A timeout should raise OllamaConnectionError."""
        import requests as req_lib

        responses_mock.add(
            responses_mock.GET,
            f"{BASE_URL}/api/tags",
            body=req_lib.exceptions.Timeout("timed out"),
        )
        runner = BenchmarkRunner(base_url=BASE_URL)
        with pytest.raises(OllamaConnectionError, match="timed out"):
            runner.check_connection()


# ---------------------------------------------------------------------------
# list_models
# ---------------------------------------------------------------------------

class TestListModels:
    """Tests for BenchmarkRunner.list_models."""

    @responses_mock.activate
    def test_returns_all_models_when_no_family(self) -> None:
        """Without a family filter, all pulled models are returned."""
        responses_mock.add(
            responses_mock.GET,
            f"{BASE_URL}/api/tags",
            json={"models": [
                {"name": "qwen2.5:3b"},
                {"name": "llama3:8b"},
                {"name": "mistral:7b"},
            ]},
            status=200,
        )
        runner = BenchmarkRunner(base_url=BASE_URL)
        models = runner.list_models()

        assert len(models) == 3
        names = [m["name"] for m in models]
        assert "qwen2.5:3b" in names
        assert "llama3:8b" in names

    @responses_mock.activate
    def test_filters_by_family_prefix(self) -> None:
        """With a family filter, only matching models are returned."""
        responses_mock.add(
            responses_mock.GET,
            f"{BASE_URL}/api/tags",
            json={"models": [
                {"name": "qwen2.5:3b"},
                {"name": "qwen2.5:7b"},
                {"name": "llama3:8b"},
            ]},
            status=200,
        )
        runner = BenchmarkRunner(base_url=BASE_URL)
        models = runner.list_models(family="qwen2.5")

        assert len(models) == 2
        names = [m["name"] for m in models]
        assert "llama3:8b" not in names

    @responses_mock.activate
    def test_param_count_parsed_from_name(self) -> None:
        """Parameter count should be extracted and formatted from the model name."""
        responses_mock.add(
            responses_mock.GET,
            f"{BASE_URL}/api/tags",
            json={"models": [{"name": "qwen2.5:14b"}]},
            status=200,
        )
        runner = BenchmarkRunner(base_url=BASE_URL)
        models = runner.list_models()

        assert len(models) == 1
        assert models[0]["params"] == "14b"

    @responses_mock.activate
    def test_unknown_params_when_no_size_in_name(self) -> None:
        """Models with no parseable size should have params == 'unknown'."""
        responses_mock.add(
            responses_mock.GET,
            f"{BASE_URL}/api/tags",
            json={"models": [{"name": "mistral:latest"}]},
            status=200,
        )
        runner = BenchmarkRunner(base_url=BASE_URL)
        models = runner.list_models()

        assert models[0]["params"] == "unknown"

    @responses_mock.activate
    def test_empty_list_when_no_models_match(self) -> None:
        """A family filter that matches nothing should return an empty list."""
        responses_mock.add(
            responses_mock.GET,
            f"{BASE_URL}/api/tags",
            json={"models": [{"name": "llama3:8b"}]},
            status=200,
        )
        runner = BenchmarkRunner(base_url=BASE_URL)
        models = runner.list_models(family="qwen2.5")

        assert models == []


# ---------------------------------------------------------------------------
# run_single
# ---------------------------------------------------------------------------

class TestRunSingle:
    """Tests for BenchmarkRunner.run_single."""

    @responses_mock.activate
    def test_basic_streaming_response(self) -> None:
        """run_single should return ttft_ms, tokens_per_sec, response, ram_gb."""
        ndjson_body = _make_ndjson(
            {"model": "qwen2.5:7b", "response": "Hello", "done": False},
            {"model": "qwen2.5:7b", "response": " world", "done": False},
            {
                "model": "qwen2.5:7b",
                "response": "",
                "done": True,
                "eval_count": 10,
                "eval_duration": 1_000_000_000,
            },
        )
        responses_mock.add(
            responses_mock.POST,
            f"{BASE_URL}/api/generate",
            body=ndjson_body,
            status=200,
            stream=True,
        )

        runner = BenchmarkRunner(base_url=BASE_URL)
        result = runner.run_single("qwen2.5:7b", "Say hello")

        assert "ttft_ms" in result
        assert result["ttft_ms"] >= 0.0
        assert result["tokens_per_sec"] == pytest.approx(10.0, rel=1e-3)
        assert result["response"] == "Hello world"
        assert result["ram_gb"] > 0.0

    @responses_mock.activate
    def test_connection_error_raises(self) -> None:
        """A connection error during streaming should raise OllamaConnectionError."""
        import requests as req_lib

        responses_mock.add(
            responses_mock.POST,
            f"{BASE_URL}/api/generate",
            body=req_lib.exceptions.ConnectionError("refused"),
        )
        runner = BenchmarkRunner(base_url=BASE_URL)
        with pytest.raises(OllamaConnectionError):
            runner.run_single("qwen2.5:7b", "Hello")


# ---------------------------------------------------------------------------
# run_benchmark
# ---------------------------------------------------------------------------

class TestRunBenchmark:
    """Tests for BenchmarkRunner.run_benchmark."""

    def _mock_single(self, runner: BenchmarkRunner, responses_list: list[dict]) -> None:
        call_results = iter(responses_list)

        def fake_run_single(_model: str, _prompt: str) -> dict:
            return next(call_results)

        runner.run_single = fake_run_single  # type: ignore[method-assign]

    def test_warmup_is_discarded(self) -> None:
        """With runs=2, only the non-warmup run should contribute to averages."""
        runner = BenchmarkRunner(base_url=BASE_URL)
        self._mock_single(runner, [
            {"ttft_ms": 9999.0, "tokens_per_sec": 1.0, "response": "warmup", "ram_gb": 1.0},
            {"ttft_ms": 100.0, "tokens_per_sec": 10.0, "response": "real", "ram_gb": 1.0},
        ])

        result = runner.run_benchmark("qwen2.5:7b", ["What is 2+2?"], runs=2)

        assert result["prompts"][0]["avg_ttft_ms"] == pytest.approx(100.0)
        assert result["prompts"][0]["avg_tokens_per_sec"] == pytest.approx(10.0)

    def test_averages_are_correct(self) -> None:
        """Averages should be computed correctly across non-warmup runs."""
        runner = BenchmarkRunner(base_url=BASE_URL)
        self._mock_single(runner, [
            {"ttft_ms": 999.0, "tokens_per_sec": 0.0, "response": "warmup", "ram_gb": 1.0},
            {"ttft_ms": 200.0, "tokens_per_sec": 8.0, "response": "r1", "ram_gb": 1.0},
            {"ttft_ms": 100.0, "tokens_per_sec": 12.0, "response": "r2", "ram_gb": 1.0},
        ])

        result = runner.run_benchmark("qwen2.5:7b", ["Prompt A"], runs=3)
        prompt_data = result["prompts"][0]

        assert prompt_data["avg_ttft_ms"] == pytest.approx(150.0)
        assert prompt_data["avg_tokens_per_sec"] == pytest.approx(10.0)
        assert prompt_data["last_response"] == "r2"

    def test_params_extracted_from_model_name(self) -> None:
        """params field should be extracted from the model tag."""
        runner = BenchmarkRunner(base_url=BASE_URL)
        self._mock_single(runner, [
            {"ttft_ms": 100.0, "tokens_per_sec": 5.0, "response": "hi", "ram_gb": 1.0},
            {"ttft_ms": 120.0, "tokens_per_sec": 5.0, "response": "hi", "ram_gb": 1.0},
        ])

        result = runner.run_benchmark("qwen2.5:14b", ["Hello"], runs=2)
        assert result["params"] == "14b"

    def test_multiple_prompts(self) -> None:
        """Each prompt should produce its own entry in result['prompts']."""
        runner = BenchmarkRunner(base_url=BASE_URL)
        self._mock_single(runner, [
            {"ttft_ms": 10.0, "tokens_per_sec": 5.0, "response": "w1", "ram_gb": 1.0},
            {"ttft_ms": 20.0, "tokens_per_sec": 6.0, "response": "r1", "ram_gb": 1.0},
            {"ttft_ms": 30.0, "tokens_per_sec": 7.0, "response": "w2", "ram_gb": 1.0},
            {"ttft_ms": 40.0, "tokens_per_sec": 8.0, "response": "r2", "ram_gb": 1.0},
        ])

        result = runner.run_benchmark("qwen2.5:7b", ["P1", "P2"], runs=2)
        assert len(result["prompts"]) == 2
        assert result["prompts"][0]["prompt"] == "P1"
        assert result["prompts"][1]["prompt"] == "P2"


# ---------------------------------------------------------------------------
# _resize_prompt
# ---------------------------------------------------------------------------

class TestResizePrompt:
    """Tests for the _resize_prompt helper."""

    def test_truncates_long_prompt(self) -> None:
        """A prompt longer than the target should be truncated to target_tokens * 4 chars."""
        prompt = "a" * 10_000
        result = _resize_prompt(prompt, 512)
        assert len(result) == 512 * 4

    def test_pads_short_prompt(self) -> None:
        """A prompt shorter than the target should be padded to target_tokens * 4 chars."""
        result = _resize_prompt("hello ", 100)
        assert len(result) == 100 * 4

    def test_exact_length_unchanged(self) -> None:
        """A prompt exactly at the target length should be returned as-is."""
        prompt = "x" * (256 * 4)
        result = _resize_prompt(prompt, 256)
        assert len(result) == 256 * 4

    def test_padded_content_is_repetition_of_original(self) -> None:
        """Padding should repeat the original prompt text."""
        prompt = "abc"
        result = _resize_prompt(prompt, 10)
        expected = ("abc" * ((40 // 3) + 1))[:40]
        assert result == expected


# ---------------------------------------------------------------------------
# run_context_sweep
# ---------------------------------------------------------------------------

class TestRunContextSweep:
    """Tests for BenchmarkRunner.run_context_sweep."""

    def _mock_benchmark(self, runner: BenchmarkRunner, return_value: dict) -> None:
        def fake_bm(_model: str, prompts: list[str], _runs: int = 3) -> dict:
            return dict(
                return_value,
                prompts=[{
                    "prompt": p, "runs": [], "avg_ttft_ms": 100.0,
                    "avg_tokens_per_sec": 5.0, "last_response": "r",
                } for p in prompts],
            )
        runner.run_benchmark = fake_bm  # type: ignore[method-assign]

    def test_produces_one_result_per_context_size(self) -> None:
        runner = BenchmarkRunner(base_url=BASE_URL)
        self._mock_benchmark(runner, {"name": "qwen2.5:7b", "params": "7b"})

        results = runner.run_context_sweep("qwen2.5:7b", ["Hello"], runs=2, context_sizes=[512, 2048])

        assert len(results) == 2
        assert results[0]["context_size"] == 512
        assert results[1]["context_size"] == 2048

    def test_prompts_resized_to_target(self) -> None:
        runner = BenchmarkRunner(base_url=BASE_URL)
        captured: list[list[str]] = []

        def fake_bm(_model: str, prompts: list[str], _runs: int = 3) -> dict:
            captured.append(list(prompts))
            return {"name": _model, "params": "7b", "prompts": [
                {"prompt": p, "runs": [], "avg_ttft_ms": 10.0,
                 "avg_tokens_per_sec": 5.0, "last_response": "r"} for p in prompts
            ]}

        runner.run_benchmark = fake_bm  # type: ignore[method-assign]
        runner.run_context_sweep("qwen2.5:7b", ["x" * 100], runs=2, context_sizes=[512])

        assert len(captured[0][0]) == 512 * 4

    def test_original_prompt_stored(self) -> None:
        runner = BenchmarkRunner(base_url=BASE_URL)
        self._mock_benchmark(runner, {"name": "qwen2.5:7b", "params": "7b"})

        original = "Hello world"
        results = runner.run_context_sweep("qwen2.5:7b", [original], runs=2, context_sizes=[512])

        assert results[0]["prompts"][0]["original_prompt"] == original

    def test_uses_default_context_sizes(self) -> None:
        runner = BenchmarkRunner(base_url=BASE_URL)
        call_count = 0

        def fake_bm(_model: str, prompts: list[str], _runs: int = 3) -> dict:
            nonlocal call_count
            call_count += 1
            return {"name": _model, "params": "7b", "prompts": [
                {"prompt": p, "runs": [], "avg_ttft_ms": 10.0,
                 "avg_tokens_per_sec": 5.0, "last_response": "r"} for p in prompts
            ]}

        runner.run_benchmark = fake_bm  # type: ignore[method-assign]
        results = runner.run_context_sweep("qwen2.5:7b", ["Hi"], runs=2)

        assert call_count == 3
        assert [r["context_size"] for r in results] == [512, 2048, 4096]
