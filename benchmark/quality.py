"""Semantic quality scorer using sentence-transformers and ROUGE-L."""

from __future__ import annotations

from typing import Any

import numpy as np
from rich.console import Console

# Import at module level so the name is patchable in tests.
try:
    from sentence_transformers import SentenceTransformer
except ImportError:  # pragma: no cover
    SentenceTransformer = None  # type: ignore[assignment,misc]

try:
    from rouge_score import rouge_scorer as _rouge_scorer_mod
except ImportError:  # pragma: no cover
    _rouge_scorer_mod = None  # type: ignore[assignment]

from benchmark.runner import parse_param_count

_console = Console()


class QualityScorer:
    """Scores semantic similarity and ROUGE-L between model responses.

    Uses ``all-MiniLM-L6-v2`` for cosine similarity and ``rouge-score`` for
    ROUGE-L.  Both models are loaded lazily on first use.

    The baseline for scoring is the **largest model** in the result set by
    parameter count (parsed from the model name, e.g. ``"14b"`` > ``"7b"``).
    If two models have the same parameter count the first in the list is used.
    """

    def __init__(self) -> None:
        self._embed_model: Any = None
        self._rouge: Any = None

    # ------------------------------------------------------------------
    # Lazy loading
    # ------------------------------------------------------------------

    def _load_embed_model(self) -> None:
        """Load ``all-MiniLM-L6-v2`` with a rich spinner.  No-op if loaded."""
        if self._embed_model is not None:
            return
        with _console.status(
            "[bold cyan]Loading sentence-transformers model "
            "(all-MiniLM-L6-v2, ~90 MB — one-time download)…[/bold cyan]"
        ):
            self._embed_model = SentenceTransformer("all-MiniLM-L6-v2")

    def _load_rouge(self) -> None:
        """Initialise the ROUGE scorer.  No-op if already initialised."""
        if self._rouge is not None:
            return
        if _rouge_scorer_mod is None:  # pragma: no cover
            raise ImportError(
                "rouge-score is required for quality scoring. "
                "Install it with: pip install rouge-score"
            )
        self._rouge = _rouge_scorer_mod.RougeScorer(["rougeL"], use_stemmer=True)

    # ------------------------------------------------------------------
    # Individual scores
    # ------------------------------------------------------------------

    def similarity(self, baseline: str, candidate: str) -> float:
        """Compute cosine similarity between *baseline* and *candidate*.

        Args:
            baseline: Reference response text.
            candidate: Response to evaluate.

        Returns:
            Float in ``[0.0, 1.0]`` where ``1.0`` means identical semantics.
        """
        self._load_embed_model()
        embeddings = self._embed_model.encode(
            [baseline, candidate], convert_to_numpy=True
        )
        vec_a: np.ndarray = embeddings[0]
        vec_b: np.ndarray = embeddings[1]

        norm_a = float(np.linalg.norm(vec_a))
        norm_b = float(np.linalg.norm(vec_b))
        if norm_a == 0.0 or norm_b == 0.0:
            return 0.0
        sim = float(np.dot(vec_a, vec_b) / (norm_a * norm_b))
        return max(0.0, min(1.0, sim))

    def rouge_l(self, baseline: str, candidate: str) -> float:
        """Compute ROUGE-L F1 between *baseline* and *candidate*.

        Args:
            baseline: Reference response text.
            candidate: Response to evaluate.

        Returns:
            ROUGE-L F1 score as a float in ``[0.0, 1.0]``.
        """
        self._load_rouge()
        scores = self._rouge.score(baseline, candidate)
        return float(scores["rougeL"].fmeasure)

    # ------------------------------------------------------------------
    # Baseline selection
    # ------------------------------------------------------------------

    @staticmethod
    def pick_baseline(results: list[dict[str, Any]]) -> dict[str, Any]:
        """Return the result dict for the largest model by parameter count.

        Parameter count is parsed from the ``"name"`` field using
        :func:`~benchmark.runner.parse_param_count`.  If all models return
        ``0.0`` (no parseable size), the first result is used.

        Args:
            results: List of model result dicts.

        Returns:
            The result dict of the chosen baseline model.
        """
        if not results:
            raise ValueError("Cannot pick baseline from an empty results list.")
        return max(results, key=lambda r: parse_param_count(r.get("name", "")))

    # ------------------------------------------------------------------
    # Batch scoring
    # ------------------------------------------------------------------

    def score_results(
        self,
        results: list[dict[str, Any]],
        baseline: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        """Add ``similarity_score`` and ``rouge_l_score`` to every prompt dict.

        The model in *baseline* (or the largest model if *baseline* is
        ``None``) is used as the reference.  Its own prompts score ``1.0``
        and ``1.0`` respectively.

        Quality is scored once per prompt using the ``last_response`` stored
        in each prompt dict.

        Args:
            results: List of model result dicts.
            baseline: Optional pre-selected baseline result dict.  If
                ``None``, :meth:`pick_baseline` is called automatically.

        Returns:
            The same *results* list, mutated in place, with
            ``"similarity_score"`` and ``"rouge_l_score"`` added to each
            prompt dict.
        """
        if not results:
            return results

        baseline_model = baseline if baseline is not None else self.pick_baseline(results)

        baseline_responses: dict[str, str] = {
            p["prompt"]: p.get("last_response", "")
            for p in baseline_model.get("prompts", [])
        }

        for model in results:
            for prompt_dict in model.get("prompts", []):
                prompt_text: str = prompt_dict["prompt"]
                baseline_text: str = baseline_responses.get(prompt_text, "")
                candidate_text: str = prompt_dict.get("last_response", "")

                if model is baseline_model:
                    prompt_dict["similarity_score"] = 1.0
                    prompt_dict["rouge_l_score"] = 1.0
                else:
                    prompt_dict["similarity_score"] = self.similarity(
                        baseline_text, candidate_text
                    )
                    prompt_dict["rouge_l_score"] = self.rouge_l(
                        baseline_text, candidate_text
                    )

        return results

    def score_sweep_results(
        self,
        sweep_results: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """Score a context-sweep result list.

        Groups by ``context_size``, picks the largest model within each group
        as baseline, then applies :meth:`score_results`.

        Args:
            sweep_results: List of result dicts with a ``"context_size"`` key.

        Returns:
            Updated sweep_results list with scores added.
        """
        from collections import defaultdict

        by_context: dict[int, list[dict[str, Any]]] = defaultdict(list)
        for result in sweep_results:
            ctx: int = result.get("context_size", 0)
            by_context[ctx].append(result)

        for ctx_size in sorted(by_context.keys()):
            self.score_results(by_context[ctx_size])

        return sweep_results
