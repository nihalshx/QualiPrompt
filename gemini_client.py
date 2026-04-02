"""
gemini_client.py
────────────────
Thin wrapper around the Google Generative AI SDK.

All five prompt variants are dispatched concurrently using
concurrent.futures so the total wall-clock time is close to
the slowest single call rather than the sum of all five.
"""

from __future__ import annotations

import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict

import google.generativeai as genai

from prompt_engine import PromptVariant


# Default generation settings
_GENERATION_CONFIG = {
    "temperature": 0.7,
    "top_p": 0.95,
    "top_k": 40,
    "max_output_tokens": 1024,
}

_MODEL_NAME = "gemini-1.5-flash"  # fast, cost-effective for batch evaluation


def _configure(api_key: str) -> None:
    genai.configure(api_key=api_key)


def _call_gemini(prompt: str, model_name: str) -> tuple[str, float]:
    """
    Send a single prompt to Gemini and return (response_text, latency_s).
    """
    model = genai.GenerativeModel(
        model_name=model_name,
        generation_config=_GENERATION_CONFIG,
    )
    t0 = time.perf_counter()
    response = model.generate_content(prompt)
    latency = time.perf_counter() - t0
    text = response.text if hasattr(response, "text") else ""
    return text, latency


def run_all_variants(
    variants: Dict[str, PromptVariant],
    api_key: str,
    model_name: str = _MODEL_NAME,
    max_workers: int = 5,
) -> Dict[str, dict]:
    """
    Concurrently dispatch all variants to Gemini.

    Returns
    -------
    dict mapping strategy_name → {
        "response": str,
        "latency_s": float,
        "error": str | None
    }
    """
    _configure(api_key)

    results: Dict[str, dict] = {name: {} for name in variants}

    def task(name: str, variant: PromptVariant):
        try:
            text, latency = _call_gemini(variant.prompt, model_name)
            return name, {"response": text, "latency_s": latency, "error": None}
        except Exception as exc:  # noqa: BLE001
            return name, {"response": "", "latency_s": 0.0, "error": str(exc)}

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(task, name, variant): name
            for name, variant in variants.items()
        }
        for future in as_completed(futures):
            name, result = future.result()
            results[name] = result

    return results
