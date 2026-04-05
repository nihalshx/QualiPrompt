"""
prompt_engine.py
────────────────
Auto-generates five canonical prompt-engineering variants from a single
task description.

Strategies
──────────
1. zero_shot       – direct task restatement, no scaffolding
2. few_shot        – two illustrative examples prepended
3. chain_of_thought– explicit step-by-step reasoning instruction appended
4. role_play       – domain-expert persona prefix
5. structured_output – format specification appended
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List


STRATEGY_NAMES: List[str] = [
    "zero_shot",
    "few_shot",
    "chain_of_thought",
    "role_play",
    "structured_output",
]

STRATEGY_LABELS: Dict[str, str] = {
    "zero_shot": "Zero-Shot",
    "few_shot": "Few-Shot",
    "chain_of_thought": "Chain-of-Thought",
    "role_play": "Role-Play",
    "structured_output": "Structured Output",
}

STRATEGY_DESCRIPTIONS: Dict[str, str] = {
    "zero_shot": (
        "The task is stated directly with no examples or scaffolding. "
        "Tests the model's baseline capability."
    ),
    "few_shot": (
        "Two illustrative input→output examples precede the task, "
        "priming the model's output style and format."
    ),
    "chain_of_thought": (
        "An explicit instruction to reason step-by-step before answering "
        "is appended to the task."
    ),
    "role_play": (
        "A domain-expert persona is assigned before the task, "
        "anchoring the model's perspective and vocabulary."
    ),
    "structured_output": (
        "A precise output-format specification is appended, "
        "guiding structure and completeness."
    ),
}


# ─── Prompt templates ───────────────────────────────────────────────────────

def _zero_shot(task: str) -> str:
    return f"""{task}

Provide a clear, accurate, and comprehensive response."""


def _few_shot(task: str) -> str:
    return f"""Below are two examples of high-quality task completions, \
followed by your actual task.

### Example 1
Task: Explain what machine learning is.
Response:
Machine learning is a subset of artificial intelligence in which systems \
learn from data to improve their performance over time without being \
explicitly programmed. Rather than following hand-crafted rules, a \
machine-learning model identifies statistical patterns in training data \
and generalises them to new, unseen inputs.

### Example 2
Task: Write a short description of the water cycle.
Response:
The water cycle is the continuous movement of water through Earth's \
systems. Solar energy evaporates water from oceans and lakes; water \
vapour rises, cools, and condenses into clouds; precipitation returns \
water to the surface; and runoff or infiltration feeds rivers and \
groundwater, restarting the cycle.

### Your Task
Task: {task}
Response:"""


def _chain_of_thought(task: str) -> str:
    return f"""{task}

Before writing your final answer, work through the problem step-by-step:
1. Identify the core question or objective.
2. Break the problem into logical sub-components.
3. Reason through each sub-component carefully.
4. Synthesise your reasoning into a coherent final answer.

Show your step-by-step thinking first, then provide your complete response."""


def _role_play(task: str) -> str:
    return f"""You are a world-class domain expert with decades of experience, \
deep theoretical knowledge, and a talent for clear, precise communication. \
You are trusted by researchers, practitioners, and decision-makers alike.

Acting in that expert capacity, please address the following:

{task}

Ensure your response reflects expert-level depth, accuracy, and nuance."""


def _structured_output(task: str) -> str:
    return f"""{task}

Please structure your response using the following format:

**Summary** (2–3 sentences capturing the core answer)

**Key Points**
- Point 1
- Point 2
- Point 3 (add more as needed)

**Explanation** (detailed prose expanding on the key points)

**Conclusion** (1–2 sentences with the main takeaway or recommendation)

Adhere strictly to this structure."""


# ─── Public API ─────────────────────────────────────────────────────────────

@dataclass
class PromptVariant:
    strategy: str
    label: str
    description: str
    prompt: str


def generate_variants(task: str) -> Dict[str, PromptVariant]:
    """
    Accept a task description and return a dict mapping strategy name →
    PromptVariant for all five canonical strategies.
    """
    task = task.strip()
    builders = {
        "zero_shot": _zero_shot,
        "few_shot": _few_shot,
        "chain_of_thought": _chain_of_thought,
        "role_play": _role_play,
        "structured_output": _structured_output,
    }
    return {
        name: PromptVariant(
            strategy=name,
            label=STRATEGY_LABELS[name],
            description=STRATEGY_DESCRIPTIONS[name],
            prompt=builders[name](task),
        )
        for name in STRATEGY_NAMES
    }
