"""
app.py
──────
Streamlit interface for the Prompt Quality Tester.

Run:
    streamlit run app.py
"""

from __future__ import annotations

import os
import time
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

load_dotenv()

# ─── Page config (must be first Streamlit call) ──────────────────────────────
st.set_page_config(
    page_title="QualiPrompt",
    page_icon="⚗️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Lazy imports (heavy ML libs load once) ──────────────────────────────────
from prompt_engine import generate_variants, STRATEGY_NAMES, STRATEGY_LABELS  # noqa: E402
from gemini_client import run_all_variants  # noqa: E402
from evaluator import evaluate_all, rank_results  # noqa: E402
from history_manager import save_session, load_history, get_session_summary  # noqa: E402
from visualisations import (  # noqa: E402
    bar_final_scores,
    radar_subscores,
    history_line_chart,
    strategy_distribution,
)


# ─── Custom CSS ──────────────────────────────────────────────────────────────
st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600&family=Space+Grotesk:wght@400;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Space Grotesk', sans-serif;
}
.stApp {
    background: linear-gradient(135deg, #0f172a 0%, #1e1b4b 50%, #0f172a 100%);
    color: #e2e8f0;
}
.main-title {
    font-size: 2.8rem;
    font-weight: 700;
    background: linear-gradient(90deg, #6366f1, #22d3ee, #f59e0b);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    letter-spacing: -0.02em;
}
.subtitle {
    color: #94a3b8;
    font-size: 1rem;
    margin-bottom: 2rem;
}
.metric-card {
    background: rgba(99,102,241,0.08);
    border: 1px solid rgba(99,102,241,0.25);
    border-radius: 12px;
    padding: 1rem 1.25rem;
    margin-bottom: 0.5rem;
}
.score-badge {
    display: inline-block;
    padding: 0.25rem 0.75rem;
    border-radius: 999px;
    font-size: 0.75rem;
    font-weight: 600;
    font-family: 'IBM Plex Mono', monospace;
}
.rank-1 { background: rgba(251,191,36,0.15); border: 1px solid #fbbf24; color: #fbbf24; }
.rank-2 { background: rgba(148,163,184,0.1); border: 1px solid #94a3b8; color: #94a3b8; }
.rank-3 { background: rgba(205,127,50,0.1); border: 1px solid #cd7f32; color: #cd7f32; }
.rank-other { background: rgba(99,102,241,0.1); border: 1px solid #6366f1; color: #6366f1; }
.strategy-pill {
    display: inline-block;
    padding: 0.2rem 0.6rem;
    border-radius: 6px;
    font-size: 0.7rem;
    font-weight: 600;
    font-family: 'IBM Plex Mono', monospace;
    margin-bottom: 0.5rem;
}
.prompt-box {
    background: rgba(15,23,42,0.8);
    border: 1px solid rgba(99,102,241,0.2);
    border-radius: 8px;
    padding: 0.75rem 1rem;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.78rem;
    color: #94a3b8;
    white-space: pre-wrap;
    max-height: 160px;
    overflow-y: auto;
}
.response-box {
    background: rgba(15,23,42,0.6);
    border: 1px solid rgba(34,211,238,0.15);
    border-radius: 8px;
    padding: 1rem;
    font-size: 0.88rem;
    color: #cbd5e1;
    line-height: 1.6;
    max-height: 260px;
    overflow-y: auto;
}
.section-header {
    font-size: 1.1rem;
    font-weight: 600;
    color: #6366f1;
    border-bottom: 1px solid rgba(99,102,241,0.25);
    padding-bottom: 0.4rem;
    margin-bottom: 1rem;
}
div[data-testid="stSidebar"] {
    background: rgba(15,23,42,0.95) !important;
    border-right: 1px solid rgba(99,102,241,0.2);
}
div[data-testid="stExpander"] {
    border: 1px solid rgba(99,102,241,0.2) !important;
    border-radius: 10px !important;
    background: rgba(15,23,42,0.5) !important;
}
</style>
""",
    unsafe_allow_html=True,
)

# ─── Sidebar ─────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚗️ QualiPrompt")

    api_key = st.text_input(
        "Google Gemini API Key",
        value=os.getenv("GEMINI_API_KEY", ""),
        type="password",
        help="Get your key at https://aistudio.google.com/app/apikey",
    )

    st.markdown("---")
    st.markdown("### Model")
    model_choice = st.selectbox(
        "Gemini model",
        ["gemini-1.5-flash", "gemini-1.5-pro", "gemini-2.0-flash"],
        index=0,
    )

    st.markdown("---")
    st.markdown("### Hugging Face Publishing")
    hf_token = st.text_input(
        "HF Token (optional)",
        value=os.getenv("HF_TOKEN", ""),
        type="password",
    )
    hf_repo = st.text_input(
        "Dataset repo",
        value=os.getenv("HF_DATASET_REPO", ""),
        placeholder="username/my-prompt-dataset",
    )

    st.markdown("---")
    st.markdown(
        "<small style='color:#475569'>QualiPrompt v1.0<br>"
        "Quality-driven prompt evaluation, scientifically.<br>"
        "Powered by Gemini + sentence-transformers</small>",
        unsafe_allow_html=True,
    )


# ─── Header ──────────────────────────────────────────────────────────────────
st.markdown('<div class="main-title">⚗️ QualiPrompt</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="subtitle">Quality-driven prompt evaluation, scientifically. '
    "Auto-generate five canonical prompt variants, evaluate them with "
    "multi-metric scoring, and discover which strategy works best for your task.</div>",
    unsafe_allow_html=True,
)

# ─── Main tabs ───────────────────────────────────────────────────────────────
tab_test, tab_history, tab_publish = st.tabs(
    ["🧪 Evaluate", "📊 History", "🚀 Publish to HF"]
)

# ══════════════════════════════════════════════════════════════════════════════
# TAB 1: Evaluate
# ══════════════════════════════════════════════════════════════════════════════
with tab_test:
    task_input = st.text_area(
        "Task description",
        placeholder=(
            "e.g.  Explain the difference between supervised and "
            "unsupervised machine learning to a non-technical audience."
        ),
        height=100,
        key="task_input",
    )

    run_btn = st.button(
        "⚗️ Run QualiPrompt Evaluation",
        type="primary",
        disabled=not (api_key and task_input.strip()),
        use_container_width=True,
    )

    if not api_key:
        st.warning("⚠️  Enter your Gemini API key in the sidebar to begin.")

    # ── Run pipeline ─────────────────────────────────────────────────────────
    if run_btn and api_key and task_input.strip():
        task = task_input.strip()

        with st.status("Running evaluation pipeline…", expanded=True) as status:
            st.write("📝 Generating prompt variants…")
            variants = generate_variants(task)

            st.write("🤖 Calling Gemini for all five variants…")
            t_start = time.perf_counter()
            api_results = run_all_variants(
                variants, api_key=api_key, model_name=model_choice
            )
            elapsed = time.perf_counter() - t_start

            st.write("📐 Computing quality metrics…")
            eval_results = evaluate_all(task, variants, api_results)
            ranked = rank_results(eval_results)

            st.write("💾 Saving to session history…")
            session_id = save_session(task, variants, api_results, eval_results, model_choice)

            status.update(
                label=f"✅ Done in {elapsed:.1f}s  |  session `{session_id[:8]}…`",
                state="complete",
            )

        # Store in session state so UI can display without re-running
        st.session_state["last_ranked"] = ranked
        st.session_state["last_eval"] = eval_results
        st.session_state["last_variants"] = variants
        st.session_state["last_api"] = api_results
        st.session_state["last_task"] = task

    # ── Display results ──────────────────────────────────────────────────────
    if "last_ranked" in st.session_state:
        ranked = st.session_state["last_ranked"]
        eval_results = st.session_state["last_eval"]
        variants = st.session_state["last_variants"]
        api_results = st.session_state["last_api"]
        task = st.session_state["last_task"]

        st.markdown("---")

        # ── Score overview charts ─────────────────────────────────────────
        col_chart1, col_chart2 = st.columns([1, 1])
        with col_chart1:
            st.plotly_chart(bar_final_scores(ranked), use_container_width=True)
        with col_chart2:
            st.plotly_chart(radar_subscores(eval_results), use_container_width=True)

        # ── Per-strategy details ──────────────────────────────────────────
        st.markdown('<div class="section-header">📋 Detailed Results</div>', unsafe_allow_html=True)

        rank_badges = ["rank-1", "rank-2", "rank-3", "rank-other", "rank-other"]
        rank_icons = ["🥇", "🥈", "🥉", "4️⃣", "5️⃣"]

        for i, ev in enumerate(ranked):
            variant = variants[ev.strategy]
            api = api_results.get(ev.strategy, {})
            badge_cls = rank_badges[min(i, 4)]
            icon = rank_icons[min(i, 4)]

            with st.expander(
                f"{icon} **{ev.label}** — Score: {ev.final_score:.2f}/10",
                expanded=(i == 0),
            ):
                # Strategy description
                st.markdown(
                    f"<div style='color:#94a3b8;font-size:0.85rem;margin-bottom:0.75rem'>"
                    f"{variant.description}</div>",
                    unsafe_allow_html=True,
                )

                # Metrics row
                m1, m2, m3, m4 = st.columns(4)
                with m1:
                    st.metric("Final Score", f"{ev.final_score:.2f}")
                with m2:
                    st.metric("Semantic Relevance", f"{ev.semantic_relevance:.2f}")
                with m3:
                    st.metric("Readability", f"{ev.readability_score:.2f}")
                with m4:
                    st.metric("Length Score", f"{ev.length_score:.2f}")

                # Additional stats
                col_a, col_b = st.columns(2)
                with col_a:
                    st.markdown(
                        f"Words: **{ev.word_count}**  |  "
                        f"Flesch: **{ev.flesch_reading_ease:.1f}**  |  "
                        f"Latency: **{api.get('latency_s', 0):.2f}s**"
                    )

                if ev.error:
                    st.error(f"API error: {ev.error}")
                    continue

                # Prompt
                st.markdown("**Prompt sent to Gemini:**")
                st.markdown(
                    f"<div class='prompt-box'>{variant.prompt}</div>",
                    unsafe_allow_html=True,
                )

                # Response
                st.markdown("**Model response:**")
                st.markdown(
                    f"<div class='response-box'>{api.get('response','').replace(chr(10), '<br>')}</div>",
                    unsafe_allow_html=True,
                )

        # ── Best strategy recommendation ──────────────────────────────────
        best = ranked[0]
        st.success(
            f"🏆 **Recommended strategy for this task: {best.label}** "
            f"(score {best.final_score:.2f}/10)\n\n"
            f"{STRATEGY_LABELS[best.strategy]} produced the most relevant, "
            f"readable, and appropriately-lengthed response for your task."
        )


# ══════════════════════════════════════════════════════════════════════════════
# TAB 2: History
# ══════════════════════════════════════════════════════════════════════════════
with tab_history:
    df_hist = load_history()
    summary = get_session_summary(df_hist)

    if df_hist.empty:
        st.info("No sessions recorded yet. Run an evaluation first.")
    else:
        # Summary metrics
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Total Sessions", summary.get("total_sessions", 0))
        col2.metric("Total Evaluations", summary.get("total_evaluations", 0))
        col3.metric("Avg Score", f"{summary.get('avg_final_score', 0):.2f}")
        col4.metric("Overall Best Strategy", summary.get("best_strategy", "N/A"))

        st.markdown("---")

        col_h1, col_h2 = st.columns([3, 2])
        with col_h1:
            st.plotly_chart(history_line_chart(df_hist), use_container_width=True)
        with col_h2:
            st.plotly_chart(strategy_distribution(df_hist), use_container_width=True)

        # Strategy averages
        st.markdown('<div class="section-header">📈 Strategy Average Scores</div>', unsafe_allow_html=True)
        avg_df = (
            df_hist.groupby("strategy_label")["final_score"]
            .agg(["mean", "std", "count"])
            .round(2)
            .rename(columns={"mean": "Avg Score", "std": "Std Dev", "count": "Evaluations"})
            .sort_values("Avg Score", ascending=False)
        )
        st.dataframe(avg_df, use_container_width=True)

        # Raw history (paginated)
        with st.expander("🗄️ Raw session log"):
            st.dataframe(
                df_hist[
                    [
                        "timestamp", "task", "strategy_label",
                        "final_score", "semantic_relevance",
                        "readability_score", "length_score",
                        "word_count", "latency_s",
                    ]
                ].sort_values("timestamp", ascending=False),
                use_container_width=True,
                height=400,
            )

        # CSV download
        csv_bytes = df_hist.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="⬇️  Download history as CSV",
            data=csv_bytes,
            file_name="qualiprompt_history.csv",
            mime="text/csv",
        )


# ══════════════════════════════════════════════════════════════════════════════
# TAB 3: Publish to HF
# ══════════════════════════════════════════════════════════════════════════════
with tab_publish:
    st.markdown("### 🚀 Publish Dataset to Hugging Face")
    st.markdown(
        "Upload the accumulated session history as an annotated dataset to "
        "the Hugging Face Hub, with a full DatasetCard describing the schema "
        "and methodology."
    )

    df_hist_pub = load_history()

    if df_hist_pub.empty:
        st.info("Run at least one evaluation session before publishing.")
    else:
        st.markdown(f"**{len(df_hist_pub)} rows** ready to publish.")

        pub_private = st.checkbox("Make dataset private", value=False)

        if st.button("📤 Publish to Hugging Face", type="primary"):
            if not hf_token:
                st.error("Enter your Hugging Face token in the sidebar.")
            elif not hf_repo:
                st.error("Enter the target dataset repository name in the sidebar.")
            else:
                try:
                    from dataset_publisher import publish_to_huggingface

                    with st.spinner("Uploading to Hugging Face…"):
                        url = publish_to_huggingface(
                            df=df_hist_pub,
                            repo_id=hf_repo,
                            hf_token=hf_token,
                            private=pub_private,
                        )
                    st.success(f"✅ Dataset published! View at: {url}")
                    st.markdown(f"[Open dataset on HF Hub]({url})")
                except Exception as exc:  # noqa: BLE001
                    st.error(f"Publish failed: {exc}")
