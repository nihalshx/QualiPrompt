"""
evaluator.py
────────────
Automated multi-metric quality scoring system.

Three sub-scores (each normalised to [0, 10]):
    1. semantic_relevance  – cosine similarity between task embedding and
                             response embedding (via sentence-transformers)
    2. length_score        – response word-count versus task-type norms
    3. readability_score   – Flesch Reading Ease (via textstat)

Final quality score = weighted sum of the three sub-scores.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Dict, Optional

import numpy as np
import textstat
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


# ─── Weights ────────────────────────────────────────────────────────────────

WEIGHTS: Dict[str, float] = {
    "semantic_relevance": 0.50,
    "length_score": 0.20,
    "readability_score": 0.30,
}

# ─── Word-count norms by task type (min, target, max) ───────────────────────
# These are deliberately broad; the scoring function gives full marks anywhere
# within [min, max] and decays outside that range.

_LENGTH_NORMS = {
    "default": (80, 200, 500),
    "explain": (100, 250, 600),
    "summarise": (50, 150, 300),
    "write": (120, 300, 800),
    "list": (50, 150, 400),
    "compare": (100, 250, 600),
    "analyse": (120, 300, 700),
    "translate": (30, 100, 300),
    "code": (50, 200, 800),
}

_TASK_KEYWORDS: Dict[str, list[str]] = {
    "summarise": ["summarise", "summarize", "summary", "brief", "tldr"],
    "explain": ["explain", "describe", "what is", "define", "how does"],
    "write": ["write", "draft", "compose", "create a", "generate a"],
    "list": ["list", "enumerate", "give me", "provide a list"],
    "compare": ["compare", "contrast", "difference", "versus", "vs"],
    "analyse": ["analyse", "analyze", "evaluate", "assess", "critique"],
    "translate": ["translate", "in spanish", "in french", "in german"],
    "code": ["code", "function", "script", "program", "implement"],
}


# ─── Model loading (cached) ─────────────────────────────────────────────────

@lru_cache(maxsize=1)
def _get_model() -> SentenceTransformer:
    return SentenceTransformer("all-MiniLM-L6-v2")


# ─── Helpers ─────────────────────────────────────────────────────────────────

def _detect_task_type(task: str) -> str:
    task_lower = task.lower()
    for task_type, keywords in _TASK_KEYWORDS.items():
        if any(kw in task_lower for kw in keywords):
            return task_type
    return "default"


def _word_count(text: str) -> int:
    return len(re.findall(r"\w+", text))


def _compute_semantic_relevance(task: str, response: str) -> float:
    """Return cosine similarity [0, 1] scaled to [0, 10]."""
    if not response.strip():
        return 0.0
    model = _get_model()
    embeddings = model.encode([task, response])
    sim = float(cosine_similarity([embeddings[0]], [embeddings[1]])[0][0])
    sim = max(0.0, min(1.0, sim))          # clamp
    return round(sim * 10, 2)


def _compute_length_score(response: str, task_type: str) -> float:
    """Return a 0–10 score based on word-count proximity to task norms."""
    wc = _word_count(response)
    lo, target, hi = _LENGTH_NORMS.get(task_type, _LENGTH_NORMS["default"])
    if lo <= wc <= hi:
        # Triangular peak at target
        if wc <= target:
            score = (wc - lo) / max(target - lo, 1) * 10
        else:
            score = (hi - wc) / max(hi - target, 1) * 10
        score = max(7.0, score)            # within range = at least 7
    elif wc < lo:
        score = max(0.0, (wc / lo) * 7)   # below minimum — linear decay
    else:
        # above maximum
        excess_ratio = (wc - hi) / hi
        score = max(0.0, 7.0 - excess_ratio * 7)
    return round(min(score, 10.0), 2)


def _compute_readability(response: str) -> float:
    """
    Flesch Reading Ease → 0–10.

    Flesch: 0 (very hard) – 100 (very easy).
    We target 40–70 as ideal (professional writing) and penalise extremes.
    """
    if _word_count(response) < 10:
        return 0.0
    fe = textstat.flesch_reading_ease(response)
    fe = max(0.0, min(100.0, fe))
    # Map to 0–10 with a broad peak around 40–70
    if 40 <= fe <= 70:
        score = 10.0
    elif fe < 40:
        score = 10.0 * (fe / 40)
    else:
        score = 10.0 * ((100 - fe) / 30)
    return round(max(0.0, score), 2)


# ─── Public API ──────────────────────────────────────────────────────────────

@dataclass
class EvalResult:
    strategy: str
    label: str
    semantic_relevance: float
    length_score: float
    readability_score: float
    final_score: float
    word_count: int
    flesch_reading_ease: float
    error: Optional[str] = None
    sub_scores: dict = field(default_factory=dict)

    def as_dict(self) -> dict:
        return {
            "strategy": self.strategy,
            "label": self.label,
            "semantic_relevance": self.semantic_relevance,
            "length_score": self.length_score,
            "readability_score": self.readability_score,
            "final_score": self.final_score,
            "word_count": self.word_count,
            "flesch_reading_ease": self.flesch_reading_ease,
            "error": self.error,
        }


def evaluate_response(
    task: str,
    strategy: str,
    label: str,
    response: str,
    error: Optional[str] = None,
) -> EvalResult:
    """Compute all metrics for a single prompt-response pair."""
    if error or not response.strip():
        return EvalResult(
            strategy=strategy,
            label=label,
            semantic_relevance=0.0,
            length_score=0.0,
            readability_score=0.0,
            final_score=0.0,
            word_count=0,
            flesch_reading_ease=0.0,
            error=error or "Empty response",
        )

    task_type = _detect_task_type(task)

    sem = _compute_semantic_relevance(task, response)
    lng = _compute_length_score(response, task_type)
    rdg = _compute_readability(response)

    final = (
        WEIGHTS["semantic_relevance"] * sem
        + WEIGHTS["length_score"] * lng
        + WEIGHTS["readability_score"] * rdg
    )
    final = round(min(10.0, final), 2)

    return EvalResult(
        strategy=strategy,
        label=label,
        semantic_relevance=sem,
        length_score=lng,
        readability_score=rdg,
        final_score=final,
        word_count=_word_count(response),
        flesch_reading_ease=round(
            textstat.flesch_reading_ease(response), 2
        ) if _word_count(response) >= 10 else 0.0,
        error=None,
        sub_scores={
            "semantic_relevance": sem,
            "length_score": lng,
            "readability_score": rdg,
        },
    )


def evaluate_all(
    task: str,
    variants: dict,       # strategy → PromptVariant
    api_results: dict,    # strategy → {response, latency_s, error}
) -> Dict[str, EvalResult]:
    """Evaluate all strategies and return a ranked dict."""
    results = {}
    for name, variant in variants.items():
        api = api_results.get(name, {})
        results[name] = evaluate_response(
            task=task,
            strategy=name,
            label=variant.label,
            response=api.get("response", ""),
            error=api.get("error"),
        )
    return results


def rank_results(eval_results: Dict[str, EvalResult]) -> list[EvalResult]:
    """Return list of EvalResult sorted by final_score descending."""
    return sorted(eval_results.values(), key=lambda r: r.final_score, reverse=True)
