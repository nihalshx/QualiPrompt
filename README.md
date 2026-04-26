# ⚗️ QualiPrompt

> *Quality-driven prompt evaluation, scientifically.*

**QualiPrompt** is an automated evaluation framework that generates **five structurally distinct prompt variants** for any task and quantitatively ranks the resulting LLM outputs using multi-metric quality scoring.

---

## 📌 Overview

| Feature | Detail |
|---|---|
| **Prompt strategies** | Zero-Shot · Few-Shot · Chain-of-Thought · Role-Play · Structured Output |
| **LLM backend** | Google Gemini (1.5 Flash / Pro / 2.0 Flash) |
| **Quality metrics** | Semantic Relevance · Length Appropriateness · Readability |
| **Interface** | Streamlit (dark, interactive, chart-rich) |
| **Data persistence** | JSONL + CSV session history |
| **Dataset publishing** | One-click Hugging Face upload |

---

## 🚀 Quickstart

### 1. Clone & install

\`\`\`bash
git clone https://github.com/nihalshx/QualiPrompt.git
cd QualiPrompt
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
\`\`\`

### 2. Configure environment

\`\`\`bash
cp .env.example .env
# Edit .env and set GEMINI_API_KEY
\`\`\`

Get your Gemini API key for free at https://aistudio.google.com/app/apikey

### 3. Run

\`\`\`bash
streamlit run app.py
\`\`\`

Open http://localhost:8501 in your browser.

---

## 🏗️ Project Structure

\`\`\`
QualiPrompt/
│
├── app.py                  # Streamlit UI (main entry point)
├── prompt_engine.py        # Generates five prompt variants from a task
├── gemini_client.py        # Concurrent Gemini API calls
├── evaluator.py            # Multi-metric quality scoring
├── history_manager.py      # JSONL/CSV session persistence
├── dataset_publisher.py    # Hugging Face dataset upload
├── visualisations.py       # Plotly chart builders
│
├── requirements.txt
├── .env.example
└── README.md
\`\`\`

---

## 📐 Methodology

### Prompt Strategies

| Strategy | What it does |
|---|---|
| **Zero-Shot** | States the task directly — tests baseline model capability |
| **Few-Shot** | Prepends two illustrative examples to prime the output style |
| **Chain-of-Thought** | Appends an explicit step-by-step reasoning instruction |
| **Role-Play** | Prefixes a domain-expert persona before the task |
| **Structured Output** | Appends a strict format specification |

### Quality Metrics

| Metric | Weight | Method |
|---|---|---|
| **Semantic Relevance** | 50% | Cosine similarity via all-MiniLM-L6-v2 |
| **Readability** | 30% | Flesch Reading Ease normalised to 0–10 |
| **Length Appropriateness** | 20% | Word count vs task-type norms |

---

## 🚀 Hugging Face Publishing

1. Set HF_TOKEN and HF_DATASET_REPO=yourname/qualiprompt-dataset in .env
2. Open the Publish to HF tab in the app
3. Click 📤 Publish to Hugging Face

---

## 📄 License

MIT License
