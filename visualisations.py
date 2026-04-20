"""
visualisations.py
─────────────────
Plotly chart builders for the Streamlit interface.
"""

from __future__ import annotations

from typing import Dict, List

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from evaluator import EvalResult


# ─── Colour palette ──────────────────────────────────────────────────────────

_STRATEGY_COLORS = {
    "Zero-Shot": "#6366f1",
    "Few-Shot": "#22d3ee",
    "Chain-of-Thought": "#f59e0b",
    "Role-Play": "#10b981",
    "Structured Output": "#f43f5e",
}

_BG = "rgba(0,0,0,0)"
_FONT = dict(family="IBM Plex Mono, monospace", color="#e2e8f0")


def _base_layout(**kwargs) -> dict:
    return dict(
        paper_bgcolor=_BG,
        plot_bgcolor=_BG,
        font=_FONT,
        margin=dict(l=40, r=20, t=50, b=40),
        **kwargs,
    )


# ─── Chart functions ─────────────────────────────────────────────────────────

def bar_final_scores(ranked: List[EvalResult]) -> go.Figure:
    """Horizontal bar chart of final quality scores."""
    labels = [r.label for r in reversed(ranked)]
    scores = [r.final_score for r in reversed(ranked)]
    colors = [_STRATEGY_COLORS.get(l, "#94a3b8") for l in labels]

    fig = go.Figure(
        go.Bar(
            x=scores,
            y=labels,
            orientation="h",
            marker_color=colors,
            text=[f"{s:.1f}" for s in scores],
            textposition="outside",
            textfont=dict(size=13, color="#e2e8f0"),
            hovertemplate="%{y}: <b>%{x:.2f}</b><extra></extra>",
        )
    )
    fig.update_layout(
        **_base_layout(title=dict(text="Quality Score Ranking", x=0.02)),
        xaxis=dict(
            range=[0, 10.5],
            gridcolor="rgba(148,163,184,0.15)",
            title="Score (0–10)",
        ),
        yaxis=dict(gridcolor="rgba(148,163,184,0.0)"),
        height=320,
    )
    return fig


def radar_subscores(eval_results: Dict[str, EvalResult]) -> go.Figure:
    """Radar chart comparing sub-scores across strategies."""
    categories = ["Semantic\nRelevance", "Length\nScore", "Readability\nScore"]

    fig = go.Figure()
    for name, ev in eval_results.items():
        if ev.error:
            continue
        values = [
            ev.semantic_relevance,
            ev.length_score,
            ev.readability_score,
        ]
        color = _STRATEGY_COLORS.get(ev.label, "#94a3b8")
        fig.add_trace(
            go.Scatterpolar(
                r=values + [values[0]],
                theta=categories + [categories[0]],
                name=ev.label,
                line=dict(color=color, width=2),
                fill="toself",
                fillcolor=color.replace("#", "rgba(").rstrip(")") + ",0.08)"
                if color.startswith("#")
                else color,
                opacity=0.85,
            )
        )
    fig.update_layout(
        **_base_layout(
            title=dict(text="Sub-Score Breakdown", x=0.02),
        ),
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 10],
                gridcolor="rgba(148,163,184,0.2)",
                tickfont=dict(color="#94a3b8", size=10),
            ),
            angularaxis=dict(gridcolor="rgba(148,163,184,0.2)"),
            bgcolor=_BG,
        ),
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=-0.3, x=0.1),
        height=380,
    )
    return fig


def history_line_chart(df: pd.DataFrame) -> go.Figure:
    """
    Line chart of average final_score per strategy over time
    (grouped by session_id order).
    """
    if df.empty:
        return go.Figure()

    df = df.copy()
    df["session_order"] = df.groupby("strategy_label").cumcount() + 1
    pivot = df.pivot_table(
        index="session_order", columns="strategy_label", values="final_score", aggfunc="mean"
    ).reset_index()

    fig = go.Figure()
    for col in pivot.columns:
        if col == "session_order":
            continue
        color = _STRATEGY_COLORS.get(col, "#94a3b8")
        fig.add_trace(
            go.Scatter(
                x=pivot["session_order"],
                y=pivot[col],
                mode="lines+markers",
                name=col,
                line=dict(color=color, width=2),
                marker=dict(size=6),
                hovertemplate="Session %{x}<br>Score: %{y:.2f}<extra></extra>",
            )
        )
    fig.update_layout(
        **_base_layout(title=dict(text="Score Trend Over Sessions", x=0.02)),
        xaxis=dict(
            title="Session #",
            gridcolor="rgba(148,163,184,0.15)",
        ),
        yaxis=dict(
            title="Avg Final Score",
            range=[0, 10.5],
            gridcolor="rgba(148,163,184,0.15)",
        ),
        legend=dict(orientation="h", yanchor="bottom", y=-0.4, x=0),
        height=340,
    )
    return fig


def strategy_distribution(df: pd.DataFrame) -> go.Figure:
    """Bar chart showing how often each strategy wins (highest score per session)."""
    if df.empty:
        return go.Figure()

    winners = (
        df.loc[df.groupby("session_id")["final_score"].idxmax()]
        ["strategy_label"]
        .value_counts()
        .reset_index()
    )
    winners.columns = ["Strategy", "Wins"]
    colors = [_STRATEGY_COLORS.get(s, "#94a3b8") for s in winners["Strategy"]]

    fig = go.Figure(
        go.Bar(
            x=winners["Strategy"],
            y=winners["Wins"],
            marker_color=colors,
            text=winners["Wins"],
            textposition="outside",
            hovertemplate="%{x}: <b>%{y}</b> win(s)<extra></extra>",
        )
    )
    fig.update_layout(
        **_base_layout(title=dict(text="Strategy Win Count", x=0.02)),
        xaxis=dict(gridcolor="rgba(148,163,184,0.0)"),
        yaxis=dict(
            title="Times Ranked #1",
            gridcolor="rgba(148,163,184,0.15)",
        ),
        height=300,
    )
    return fig
