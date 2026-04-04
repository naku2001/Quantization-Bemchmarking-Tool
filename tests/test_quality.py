"""Unit tests for benchmark/quality.py.

SentenceTransformer and rouge_scorer are mocked throughout so no network
download occurs during CI runs.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from benchmark.quality import QualityScorer


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _unit_vec(values: list[float]) -> np.ndarray:
    """Return a normalised numpy vector."""
    arr = np.array(values, dtype=float)
    return arr / np.linalg.norm(arr)


def _mock_encoder(embeddings: list[np.ndarray]) -> MagicMock:
    """Return a mock SentenceTransformer whose encode() returns *embeddings*."""
    mock = MagicMock()
    mock.encode.return_value = np.array(embeddings)
    return mock


def _mock_rouge(fmeasure: float = 0.75) -> MagicMock:
    """Return a mock RougeScorer whose score() returns a fixed fmeasure."""
    score_result = MagicMock()
    score_result.fmeasure = fmeasure
    mock = MagicMock()
    mock.score.return_value = {"rougeL": score_result}
    return mock


def _make_results() -> list[dict]:
    """Build a minimal two-model result list for scoring tests."""
    return [
        {
            "name": "qwen2.5:14b",
            "params": "14b",
            "prompts": [
                {
                    "prompt": "What is 2+2?",
                    "last_response": "The answer is 4.",
                    "avg_ttft_ms": 200.0,
                    "avg_tokens_per_sec": 5.0,
                },
            ],
        },
        {
            "name": "qwen2.5:3b",
            "params": "3b",
            "prompts": [
                {
                    "prompt": "What is 2+2?",
                    "last_response": "4",
                    "avg_ttft_ms": 80.0,
                    "avg_tokens_per_sec": 12.0,
                },
            ],
        },
    ]


# ---------------------------------------------------------------------------
# QualityScorer.similarity
# ---------------------------------------------------------------------------

class TestSimilarity:
    """Tests for QualityScorer.similarity."""

    def test_identical_embeddings_score_one(self) -> None:
        """Identical embeddings should produce cosine similarity of 1.0."""
        vec = _unit_vec([1.0, 0.0, 0.0])
        scorer = QualityScorer()
        scorer._embed_model = _mock_encoder([vec, vec])

        result = scorer.similarity("hello world", "hello world")

        assert result == pytest.approx(1.0, abs=1e-6)

    def test_orthogonal_vectors_score_zero(self) -> None:
        """Orthogonal embeddings should produce cosine similarity of 0.0."""
        vec_a = _unit_vec([1.0, 0.0])
        vec_b = _unit_vec([0.0, 1.0])
        scorer = QualityScorer()
        scorer._embed_model = _mock_encoder([vec_a, vec_b])

        result = scorer.similarity("hello world", "completely different")

        assert result == pytest.approx(0.0, abs=1e-6)

    def test_different_embeddings_score_less_than_one(self) -> None:
        """Non-identical embeddings should produce similarity < 1.0."""
        vec_a = _unit_vec([1.0, 0.0, 0.0])
        vec_b = _unit_vec([0.5, 0.5, 0.0])
        scorer = QualityScorer()
        scorer._embed_model = _mock_encoder([vec_a, vec_b])

        result = scorer.similarity("hello world", "completely different")

        assert result < 1.0

    def test_zero_vector_returns_zero(self) -> None:
        """A zero-norm vector should return 0.0 without raising."""
        vec_a = np.array([0.0, 0.0, 0.0])
        vec_b = _unit_vec([1.0, 0.0, 0.0])
        scorer = QualityScorer()
        scorer._embed_model = _mock_encoder([vec_a, vec_b])

        result = scorer.similarity("", "hello")

        assert result == 0.0

    def test_result_clamped_to_one(self) -> None:
        """Floating-point overshoot should be clamped to 1.0."""
        vec = _unit_vec([1.0, 0.0])
        mock_model = MagicMock()
        mock_model.encode.return_value = np.array([vec * 1.0000001, vec])
        scorer = QualityScorer()
        scorer._embed_model = mock_model

        result = scorer.similarity("x", "x")

        assert result <= 1.0


# ---------------------------------------------------------------------------
# QualityScorer.rouge_l
# ---------------------------------------------------------------------------

class TestRougeL:
    """Tests for QualityScorer.rouge_l."""

    def test_returns_fmeasure(self) -> None:
        """rouge_l should return the ROUGE-L F1 fmeasure value."""
        scorer = QualityScorer()
        scorer._rouge = _mock_rouge(fmeasure=0.6)

        result = scorer.rouge_l("The cat sat on the mat.", "A cat sat on a mat.")

        assert result == pytest.approx(0.6, abs=1e-6)

    def test_identical_text_high_score(self) -> None:
        """Identical text should receive a high ROUGE-L score from the mock."""
        scorer = QualityScorer()
        scorer._rouge = _mock_rouge(fmeasure=1.0)

        result = scorer.rouge_l("same text", "same text")

        assert result == pytest.approx(1.0, abs=1e-6)


# ---------------------------------------------------------------------------
# QualityScorer._load_embed_model
# ---------------------------------------------------------------------------

class TestLoadEmbedModel:
    """Tests for QualityScorer._load_embed_model."""

    def test_loads_once(self) -> None:
        """SentenceTransformer should only be instantiated once."""
        scorer = QualityScorer()
        mock_st_cls = MagicMock(return_value=MagicMock())

        with patch("benchmark.quality.SentenceTransformer", mock_st_cls):
            scorer._load_embed_model()
            scorer._load_embed_model()  # second call is a no-op

        mock_st_cls.assert_called_once_with("all-MiniLM-L6-v2")


# ---------------------------------------------------------------------------
# QualityScorer.pick_baseline
# ---------------------------------------------------------------------------

class TestPickBaseline:
    """Tests for QualityScorer.pick_baseline."""

    def test_picks_largest_by_param_count(self) -> None:
        """The model with the highest parsed parameter count should be chosen."""
        results = _make_results()  # qwen2.5:14b and qwen2.5:3b
        baseline = QualityScorer.pick_baseline(results)
        assert baseline["name"] == "qwen2.5:14b"

    def test_falls_back_to_first_when_no_size_found(self) -> None:
        """If no size can be parsed, the first model is used as baseline."""
        results = [
            {"name": "model-a", "params": "unknown", "prompts": []},
            {"name": "model-b", "params": "unknown", "prompts": []},
        ]
        baseline = QualityScorer.pick_baseline(results)
        assert baseline["name"] == "model-a"

    def test_raises_on_empty_list(self) -> None:
        """An empty results list should raise ValueError."""
        with pytest.raises(ValueError):
            QualityScorer.pick_baseline([])


# ---------------------------------------------------------------------------
# QualityScorer.score_results
# ---------------------------------------------------------------------------

class TestScoreResults:
    """Tests for QualityScorer.score_results."""

    def test_scores_added_to_all_prompt_dicts(self) -> None:
        """Every prompt dict should gain similarity_score and rouge_l_score."""
        scorer = QualityScorer()
        scorer.similarity = MagicMock(return_value=0.85)  # type: ignore[method-assign]
        scorer.rouge_l = MagicMock(return_value=0.70)  # type: ignore[method-assign]

        results = _make_results()
        updated = scorer.score_results(results)

        for model in updated:
            for p in model["prompts"]:
                assert "similarity_score" in p
                assert "rouge_l_score" in p

    def test_baseline_scores_one(self) -> None:
        """The baseline model (largest) should receive 1.0 for both scores."""
        scorer = QualityScorer()
        scorer.similarity = MagicMock(return_value=0.75)  # type: ignore[method-assign]
        scorer.rouge_l = MagicMock(return_value=0.60)  # type: ignore[method-assign]

        results = _make_results()  # 14b is baseline
        updated = scorer.score_results(results)

        baseline_prompt = updated[0]["prompts"][0]  # qwen2.5:14b
        assert baseline_prompt["similarity_score"] == pytest.approx(1.0)
        assert baseline_prompt["rouge_l_score"] == pytest.approx(1.0)

    def test_non_baseline_uses_scorer_methods(self) -> None:
        """Non-baseline models should get scores from similarity() and rouge_l()."""
        scorer = QualityScorer()
        scorer.similarity = MagicMock(return_value=0.88)  # type: ignore[method-assign]
        scorer.rouge_l = MagicMock(return_value=0.55)  # type: ignore[method-assign]

        results = _make_results()
        updated = scorer.score_results(results)

        small_prompt = updated[1]["prompts"][0]  # qwen2.5:3b
        assert small_prompt["similarity_score"] == pytest.approx(0.88)
        assert small_prompt["rouge_l_score"] == pytest.approx(0.55)

    def test_explicit_baseline_overrides_auto_pick(self) -> None:
        """When baseline is passed explicitly, it should be used regardless of size."""
        scorer = QualityScorer()
        scorer.similarity = MagicMock(return_value=0.5)  # type: ignore[method-assign]
        scorer.rouge_l = MagicMock(return_value=0.4)  # type: ignore[method-assign]

        results = _make_results()
        # Force the 3b model as baseline.
        updated = scorer.score_results(results, baseline=results[1])

        # The 3b model (index 1) should now be 1.0 and 14b (index 0) gets scored.
        assert updated[1]["prompts"][0]["similarity_score"] == pytest.approx(1.0)
        assert updated[0]["prompts"][0]["similarity_score"] == pytest.approx(0.5)

    def test_empty_results_returns_empty(self) -> None:
        """Empty input should return empty output without error."""
        scorer = QualityScorer()
        result = scorer.score_results([])
        assert result == []
