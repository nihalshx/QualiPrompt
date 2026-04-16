"""
history_manager.py
──────────────────
Persists every evaluation session as a row in a local CSV / JSON-Lines
file, accumulating the annotated dataset that can later be published to
Hugging Face.

Schema (per row)
────────────────
session_id          uuid4 string
timestamp           ISO-8601 UTC
task                original task description
strategy            prompt strategy name (e.g. "zero_shot")
strategy_label      human-readable label
prompt              full prompt text sent to the model
response            model response text
semantic_relevance  float 0–10
length_score        float 0–10
readability_score   float 0–10
final_score         float 0–10
word_count          int
flesch_reading_ease float
latency_s           float (API call duration)
model               model name used
"""

from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

from evaluator import EvalResult
from prompt_engine import PromptVariant


_DEFAULT_PATH = Path("session_history.jsonl")
_DEFAULT_CSV_PATH = Path("session_history.csv")


# ─── Public API ──────────────────────────────────────────────────────────────

def save_session(
    task: str,
    variants: Dict[str, PromptVariant],
    api_results: Dict[str, dict],
    eval_results: Dict[str, EvalResult],
    model: str = "gemini-1.5-flash",
    history_path: Path = _DEFAULT_PATH,
) -> str:
    """
    Append all five variant rows to the JSONL history file.
    Returns the session_id.
    """
    session_id = str(uuid.uuid4())
    timestamp = datetime.now(timezone.utc).isoformat()

    rows = []
    for strategy, variant in variants.items():
        api = api_results.get(strategy, {})
        ev = eval_results.get(strategy)
        if ev is None:
            continue
        row = {
            "session_id": session_id,
            "timestamp": timestamp,
            "task": task,
            "strategy": strategy,
            "strategy_label": variant.label,
            "prompt": variant.prompt,
            "response": api.get("response", ""),
            "semantic_relevance": ev.semantic_relevance,
            "length_score": ev.length_score,
            "readability_score": ev.readability_score,
            "final_score": ev.final_score,
            "word_count": ev.word_count,
            "flesch_reading_ease": ev.flesch_reading_ease,
            "latency_s": api.get("latency_s", 0.0),
            "model": model,
            "error": ev.error,
        }
        rows.append(row)

    history_path = Path(history_path)
    with history_path.open("a", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    _sync_csv(history_path, _DEFAULT_CSV_PATH)
    return session_id


def load_history(history_path: Path = _DEFAULT_PATH) -> pd.DataFrame:
    """Load the full JSONL history into a DataFrame."""
    history_path = Path(history_path)
    if not history_path.exists() or history_path.stat().st_size == 0:
        return _empty_df()

    rows = []
    with history_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    continue

    if not rows:
        return _empty_df()

    return pd.DataFrame(rows)


def _empty_df() -> pd.DataFrame:
    cols = [
        "session_id", "timestamp", "task", "strategy", "strategy_label",
        "prompt", "response", "semantic_relevance", "length_score",
        "readability_score", "final_score", "word_count",
        "flesch_reading_ease", "latency_s", "model", "error",
    ]
    return pd.DataFrame(columns=cols)


def _sync_csv(jsonl_path: Path, csv_path: Path) -> None:
    """Keep a CSV mirror in sync for easy inspection."""
    try:
        df = load_history(jsonl_path)
        if not df.empty:
            df.to_csv(csv_path, index=False, encoding="utf-8")
    except Exception:  # noqa: BLE001
        pass


def get_session_summary(df: pd.DataFrame) -> dict:
    """Return high-level statistics over the entire history."""
    if df.empty:
        return {}
    return {
        "total_sessions": df["session_id"].nunique(),
        "total_evaluations": len(df),
        "avg_final_score": round(df["final_score"].mean(), 2),
        "best_strategy": (
            df.groupby("strategy")["final_score"].mean().idxmax()
        ),
        "strategy_avg_scores": (
            df.groupby("strategy_label")["final_score"]
            .mean()
            .round(2)
            .to_dict()
        ),
    }
