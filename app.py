"""
Persian Sentiment Analysis Pipeline
Rewritten using LangChain + OpenRouter (Qwen/QwQ-32B)
Replaces the Ollama/gemma3:1b integration with OpenRouter via LangChain.
"""

import streamlit as st
import torch
import torch.nn as nn
import re
import json
import time
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# LangChain + OpenRouter
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# from langchain.output_parsers import ResponseSchema, StructuredOutputParser


from langchain_core.messages import SystemMessage, HumanMessage

# ==================== 1. Streamlit Configuration ====================
st.set_page_config(
    page_title="Persian Sentiment Analysis",
    page_icon="🔍",
    layout="wide"
)

# ==================== 2. Constants & Configuration ====================
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
MODEL_NAME = "qwen/qwen3.6-27b"           # OpenRouter model identifier
THRESHOLD = 0.1                        # ML bad-probability threshold
CONFIDENCE_THRESHOLD = 0.6            # LLM confidence threshold

# ==================== 3. Model Architecture (must match training) ====================
class PersianVocabulary:
    """Vocabulary handler for Persian text."""

    def __init__(self):
        self.word2idx = {"<PAD>": 0, "<UNK>": 1, "<SOS>": 2, "<EOS>": 3}
        self.idx2word = {0: "<PAD>", 1: "<UNK>", 2: "<SOS>", 3: "<EOS>"}

    def build_from_dict(self, word2idx: dict):
        self.word2idx = word2idx
        self.idx2word = {v: k for k, v in word2idx.items()}

    def tokenize_persian(self, text: str) -> list[str]:
        if not isinstance(text, str):
            return []
        text = re.sub(r"[^\u0600-\u06FF\s\.\,\!\?]", "", text)
        text = re.sub(r"\s+", " ", text)
        return text.strip().split()

    def numericalize(self, text: str, max_length: int = 100) -> list[int]:
        words = self.tokenize_persian(text)
        if len(words) > max_length - 2:
            words = words[: max_length - 2]
        words = ["<SOS>"] + words + ["<EOS>"]
        indices = [self.word2idx.get(w, self.word2idx["<UNK>"]) for w in words]
        if len(indices) < max_length:
            indices += [self.word2idx["<PAD>"]] * (max_length - len(indices))
        else:
            indices = indices[:max_length]
            indices[-1] = self.word2idx["<EOS>"]
        return indices

    def __len__(self):
        return len(self.word2idx)


class SentimentClassifier(nn.Module):
    """BiLSTM + attention sentiment classifier (3 classes)."""

    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int = 128,
        hidden_dim: int = 256,
        output_dim: int = 3,
        n_layers: int = 2,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.lstm = nn.LSTM(
            embedding_dim,
            hidden_dim,
            num_layers=n_layers,
            bidirectional=True,
            batch_first=True,
            dropout=dropout if n_layers > 1 else 0,
        )
        self.dropout = nn.Dropout(dropout)
        self.attention = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
        )
        self.fc1 = nn.Linear(hidden_dim * 2, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        self.relu = nn.ReLU()

    def forward(self, text):
        embedded = self.embedding(text)
        lstm_out, _ = self.lstm(embedded)
        attn_w = torch.softmax(self.attention(lstm_out), dim=1)
        context = torch.sum(attn_w * lstm_out, dim=1)
        out = self.dropout(context)
        out = self.relu(self.fc1(out))
        out = self.dropout(out)
        return self.fc2(out)


# ==================== 4. Model Loading ====================
@st.cache_resource
def load_model_and_vocab(pth_path: str = "best_sentiment_model.pth"):
    """Load BiLSTM model + vocabulary from a .pth checkpoint."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = None
    try:
        # checkpoint = torch.load(pth_path, map_location=device)
        checkpoint = torch.load(pth_path, map_location=device, weights_only=False)


        if "vocab_word2idx" in checkpoint:
            vocab = PersianVocabulary()
            vocab.build_from_dict(checkpoint["vocab_word2idx"])
            max_length = checkpoint.get("max_length", 100)
        elif "vocab" in checkpoint:
            vocab = checkpoint["vocab"]
            max_length = checkpoint.get("max_length", 100)
        else:
            st.warning("No vocabulary found in checkpoint — using minimal vocabulary.")
            vocab = PersianVocabulary()
            max_length = 100

        vocab_size = checkpoint.get("vocab_size", len(vocab))

        # ── Infer architecture from saved weights so any checkpoint works ──
        state = checkpoint["model_state_dict"]

        # embedding.weight shape: (vocab_size, embedding_dim)
        embedding_dim = state["embedding.weight"].shape[1]

        # lstm.weight_ih_l0 shape: (4 * hidden_dim, embedding_dim)
        # hidden_dim is half the first dim (because bidirectional doubles it later)
        hidden_dim = state["lstm.weight_ih_l0"].shape[0] // 4

        # fc2.weight shape: (output_dim, hidden_dim)
        output_dim = state["fc2.weight"].shape[0]

        # count LSTM layers by scanning keys like lstm.weight_ih_l0, l1, ...
        n_layers = sum(
            1 for k in state if k.startswith("lstm.weight_ih_l") and "_reverse" not in k
        )

        model = SentimentClassifier(
            vocab_size=vocab_size,
            embedding_dim=embedding_dim,
            hidden_dim=hidden_dim,
            output_dim=output_dim,
            n_layers=n_layers,
            dropout=0.3,          # dropout doesn't affect weight shapes
        )
        model.load_state_dict(checkpoint["model_state_dict"])
        model.to(device)
        model.eval()

        st.success(
            f"✅ Model loaded from `{pth_path}` | "
            f"Vocab: {len(vocab)} | "
            f"embed={embedding_dim}, hidden={hidden_dim}, layers={n_layers}"
        )
        return model, vocab, max_length, device, checkpoint

    except Exception as exc:
        st.error(f"❌ Error loading model: {exc}")
        keys = list(checkpoint.keys()) if checkpoint else "N/A"
        st.error(f"Checkpoint keys: {keys}")
        return None, None, 100, device, None


# ==================== 5. Save Helper ====================
def save_model_with_vocab(model, vocab, path: str = "best_sentiment_model_with_vocab.pth", max_length: int = 100):
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "vocab_word2idx": vocab.word2idx,
        "vocab_size": len(vocab),
        "max_length": max_length,
        "model_config": {
            "embedding_dim": 128,
            "hidden_dim": 256,
            "output_dim": 3,
            "n_layers": 2,
            "dropout": 0.3,
        },
    }
    torch.save(checkpoint, path)
    st.success(f"Model saved with vocabulary to {path}")
    return path


# ==================== 6. Predictor ====================
class SentimentPredictor:
    """Wraps the BiLSTM model for single-text inference."""

    LABEL_MAP = {0: 1, 1: 2, 2: 3}           # model idx → label (1=Good,2=Neutral,3=Bad)
    LABEL_NAMES = {1: "Good (Positive)", 2: "Neutral", 3: "Bad (Negative)"}

    def __init__(self, model, vocab, max_length: int, device):
        self.model = model
        self.vocab = vocab
        self.max_length = max_length
        self.device = device
        self.model.eval()

    def predict(self, text: str) -> dict | None:
        try:
            indices = self.vocab.numericalize(text, self.max_length)
            tensor = torch.tensor(indices, dtype=torch.long).unsqueeze(0).to(self.device)
            with torch.no_grad():
                logits = self.model(tensor)
                probs = torch.softmax(logits, dim=1)
                pred = torch.argmax(logits, dim=1).item()
            p = probs[0].cpu().numpy()
            return {
                "suggestion": self.LABEL_MAP[pred],
                "probabilities": {"good": float(p[0]), "neutral": float(p[1]), "bad": float(p[2])},
                "is_bad_ml": float(p[2]) > THRESHOLD,
                "confidence": float(p[2]),
            }
        except Exception as exc:
            st.error(f"Prediction error: {exc}")
            return None


# ==================== 7. LangChain + OpenRouter Integration ====================
def build_llm_chain(api_key: str):
    """
    Build a LangChain chain that calls Qwen3 27b via OpenRouter and returns
    a structured JSON object: {is_bad, confidence, reasoning}.
    """
    # Response schema for structured output parsing
    response_schemas = [
        ResponseSchema(name="is_bad", description="true if the text expresses bad/negative sentiment, false otherwise"),
        ResponseSchema(name="confidence", description="float between 0.0 and 1.0 representing confidence in the assessment"),
        ResponseSchema(name="reasoning", description="brief explanation in English"),
    ]
    parser = StructuredOutputParser.from_response_schemas(response_schemas)
    format_instructions = parser.get_format_instructions()

    system_template = (
        "You are a sentiment analysis assistant specialising in Persian text. "
        "Determine whether the text expresses BAD sentiment (dissatisfaction, complaint, "
        "negative experience, poor quality, disappointment, or recommendation against purchase).\n\n"
        "Key negative Persian indicators: بد, ضعیف, خراب, افتضاح, بدترین, هدر, ناراضی, گرون, کیفیت پایین\n\n"
        "{format_instructions}\n\n"
        "Respond ONLY with the JSON object — no preamble, no markdown fences."
    )

    prompt = ChatPromptTemplate.from_messages([
        SystemMessage(content=system_template.format(format_instructions=format_instructions)),
        ("human", "Analyse this Persian text for sentiment:\n\n{text}"),
    ])

    llm = ChatOpenAI(
        model=MODEL_NAME,
        openai_api_key=api_key,
        openai_api_base=OPENROUTER_BASE_URL,
        temperature=0.1,
        max_tokens=512,
        default_headers={
            "HTTP-Referer": "https://persian-sentiment.app",
            "X-Title": "Persian Sentiment Analysis",
        },
    )

    chain = prompt | llm | parser
    return chain, parser


def check_with_llm(text: str, api_key: str) -> dict:
    """Run Qwen3 27b via OpenRouter using LangChain and return structured result."""
    try:
        chain, _ = build_llm_chain(api_key)
        result = chain.invoke({"text": text})
        # Normalise types
        return {
            "is_bad": bool(result.get("is_bad", False)),
            "confidence": float(result.get("confidence", 0.0)),
            "reasoning": str(result.get("reasoning", "")),
        }
    except Exception as exc:
        return {"is_bad": False, "confidence": 0.0, "reasoning": f"LLM error: {exc}"}


# ==================== 8. Visualisations ====================
def create_pipeline_diagram(step_status: dict):
    nodes = {
        "start":     {"label": "Start",                    "x": 0, "y": 2, "color": "#4CAF50"},
        "ml":        {"label": "ML Model",                 "x": 2, "y": 2, "color": "#2196F3"},
        "threshold": {"label": f"Check >{THRESHOLD:.0%}", "x": 4, "y": 2, "color": "#FF9800"},
        "llm":       {"label": "Qwen3 Check",             "x": 6, "y": 2, "color": "#9C27B0"},
        "decision":  {"label": "Final Decision",           "x": 8, "y": 2, "color": "#607D8B"},
        "ok":        {"label": "✅ OK",                    "x": 8, "y": 1, "color": "#4CAF50"},
        "bad":       {"label": "⚠️ BAD",                   "x": 8, "y": 3, "color": "#F44336"},
    }
    edges = [
        ("start", "ml"), ("ml", "threshold"), ("threshold", "llm"),
        ("threshold", "decision"), ("llm", "decision"),
        ("decision", "ok"), ("decision", "bad"),
    ]
    fig = go.Figure()

    for nid, n in nodes.items():
        active = nid in step_status.get("active_nodes", [])
        fig.add_trace(go.Scatter(
            x=[n["x"]], y=[n["y"]], mode="markers+text",
            marker=dict(size=40, color=n["color"] if active else "#E0E0E0",
                        line=dict(width=3, color="white")),
            text=[n["label"]], textposition="middle center",
            textfont=dict(size=11, color="white"),
            name=n["label"], hoverinfo="text",
        ))

    for s, e in edges:
        sn, en = nodes[s], nodes[e]
        active = (s, e) in step_status.get("active_edges", [])
        fig.add_trace(go.Scatter(
            x=[sn["x"], en["x"]], y=[sn["y"], en["y"]], mode="lines",
            line=dict(color="green" if active else "gray", width=3 if active else 1),
            hoverinfo="none", showlegend=False,
        ))
        if active:
            fig.add_annotation(
                x=en["x"], y=en["y"], ax=sn["x"], ay=sn["y"],
                xref="x", yref="y", axref="x", ayref="y",
                text="", showarrow=True, arrowhead=3, arrowwidth=2, arrowcolor="green",
            )

    fig.update_layout(
        title=dict(text="Sentiment Analysis Pipeline Flow", x=0.5, font=dict(size=18)),
        showlegend=False, plot_bgcolor="white", height=380,
        margin=dict(l=20, r=20, t=60, b=20),
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False, range=[-1, 9]),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False, range=[0, 4]),
    )
    return fig


def create_sentiment_gauge(probabilities: dict, current_suggestion: int):
    fig = make_subplots(
        rows=1, cols=3,
        subplot_titles=("Good", "Neutral", "Bad"),
        specs=[[{"type": "indicator"}] * 3],
    )
    mapping = [("good", "#4CAF50", 1), ("neutral", "#FF9800", 2), ("bad", "#F44336", 3)]
    for i, (key, color, label_id) in enumerate(mapping):
        value = probabilities[key] * 100
        current = current_suggestion == label_id
        fig.add_trace(go.Indicator(
            mode="gauge+number",
            value=value,
            title=dict(text=key.upper(), font=dict(size=14, color=color if current else "black")),
            number=dict(suffix="%", font=dict(size=20)),
            domain={"row": 0, "column": i},
            gauge={
                "axis": {"range": [0, 100], "tickwidth": 1},
                "bar": {"color": color, "thickness": 0.8},
                "bgcolor": "white",
                "borderwidth": 2,
                "bordercolor": color if current else "gray",
                "steps": [{"range": [0, 100], "color": "#F5F5F5"}],
                "threshold": {
                    "line": {"color": "black", "width": 4},
                    "thickness": 0.8,
                    "value": THRESHOLD * 100,
                },
            },
        ), row=1, col=i + 1)

    fig.update_layout(
        height=300, margin=dict(l=20, r=20, t=50, b=20),
        title_text="Sentiment Probability Distribution", title_x=0.5,
    )
    return fig


def create_timeline(steps: list):
    if not steps:
        return go.Figure()
    df = pd.DataFrame(steps)
    fig = go.Figure(data=[go.Bar(
        x=df["step"], y=df["duration"], text=df["status"],
        marker_color=df["color"], textposition="auto",
        textfont=dict(color="white", size=12),
    )])
    fig.update_layout(
        title="Processing Timeline", xaxis_title="Step",
        yaxis_title="Duration (s)", showlegend=False,
        height=250, plot_bgcolor="white",
    )
    return fig


# ==================== 9. Analysis Pipeline ====================
def run_analysis_pipeline(text: str, predictor: SentimentPredictor, api_key: str) -> dict | None:
    steps_data = []
    status = {"active_nodes": ["start"], "active_edges": [], "current_step": "Starting analysis..."}

    # Step 1 — ML Model
    status["current_step"] = "Running ML Model..."
    status["active_nodes"].append("ml")
    status["active_edges"].append(("start", "ml"))

    with st.spinner("Analysing with ML model..."):
        t0 = time.time()
        ml_result = predictor.predict(text)
        ml_time = time.time() - t0

    steps_data.append({"step": "ML Model", "duration": round(ml_time, 3), "status": "Completed", "color": "#2196F3"})
    if not ml_result:
        return None

    # Step 2 — Threshold check
    status["current_step"] = f"Checking threshold ({THRESHOLD:.0%})..."
    status["active_nodes"].append("threshold")
    status["active_edges"].append(("ml", "threshold"))

    if ml_result["is_bad_ml"]:
        status["current_step"] = "Bad sentiment detected — sending to Qwen3..."
        status["active_nodes"].append("llm")
        status["active_edges"].append(("threshold", "llm"))

        # Step 3 — LLM verification
        with st.spinner("Verifying with Qwen3 27b via OpenRouter..."):
            t0 = time.time()
            llm_result = check_with_llm(text, api_key)
            llm_time = time.time() - t0

        steps_data.append({"step": "Qwen3 Check", "duration": round(llm_time, 3), "status": "Completed", "color": "#9C27B0"})

        # Step 4 — Final decision
        status["active_nodes"].append("decision")
        status["active_edges"].append(("llm", "decision"))

        llm_confirms = llm_result.get("is_bad", False) and llm_result.get("confidence", 0) > CONFIDENCE_THRESHOLD
        final_decision = "BAD" if llm_confirms else "OK"
        status["active_nodes"].append("bad" if llm_confirms else "ok")
        status["active_edges"].append(("decision", "bad" if llm_confirms else "ok"))

    else:
        status["current_step"] = "Not bad sentiment — skipping LLM"
        status["active_nodes"] += ["decision", "ok"]
        status["active_edges"] += [("threshold", "decision"), ("decision", "ok")]
        llm_result = None
        final_decision = "OK"

    status["current_step"] = f"Analysis complete: {final_decision}"
    return {
        "text": text,
        "ml_result": ml_result,
        "llm_result": llm_result,
        "final_decision": final_decision,
        "steps_data": steps_data,
        "pipeline_status": status,
    }


# ==================== 10. Streamlit UI ====================
def main():
    # Session state
    if "analysis_history" not in st.session_state:
        st.session_state["analysis_history"] = []
    if "current_status" not in st.session_state:
        st.session_state["current_status"] = "Ready"

    # Header
    st.title("🔍 Persian Sentiment Analysis Pipeline")
    st.markdown(
        "Analyses Persian text using a two-step pipeline:\n"
        "1. **BiLSTM ML Model** — predicts Good / Neutral / Bad\n"
        "2. **Qwen3 27b via OpenRouter** — verifies bad-sentiment cases"
    )

    # ── Sidebar ──────────────────────────────────────────────────────────────
    with st.sidebar:
        st.header("⚙️ Configuration")

        # OpenRouter API Key
        api_key = st.text_input(
            "OpenRouter API Key",
            type="password",
            placeholder="sk-or-...",
            help="Get your key at https://openrouter.ai/keys",
        )

        st.divider()

        global THRESHOLD, CONFIDENCE_THRESHOLD
        THRESHOLD = st.slider("Bad Sentiment Threshold", 0.0, 1.0, 0.1, 0.01,
                              help="ML model probability for 'bad' class that triggers LLM verification")
        CONFIDENCE_THRESHOLD = st.slider("LLM Confidence Threshold", 0.0, 1.0, 0.6, 0.05,
                                         help="Minimum Qwen3 confidence to confirm bad sentiment")
        st.caption(f"Model: `{MODEL_NAME}`")
        st.caption(f"Bad threshold: >{THRESHOLD:.0%}")
        st.caption(f"LLM confidence: >{CONFIDENCE_THRESHOLD:.0%}")

        st.divider()
        st.header("📊 Statistics")
        history = st.session_state["analysis_history"]
        if history:
            total = len(history)
            bad_count = sum(1 for r in history if r["final_decision"] == "BAD")
            c1, c2 = st.columns(2)
            c1.metric("Total", total)
            c2.metric("Bad", bad_count)
            st.metric("Bad Rate", f"{bad_count / total * 100:.1f}%")
        else:
            st.info("No analyses yet")

        st.divider()
        if st.button("🗑️ Clear History"):
            st.session_state["analysis_history"] = []
            st.rerun()

    # ── Load Model ────────────────────────────────────────────────────────────
    model, vocab, max_length, device, checkpoint = load_model_and_vocab()

    if model is None or vocab is None:
        st.error(
            "Could not load model. Ensure `best_sentiment_model.pth` exists in the current directory "
            "and contains a vocabulary (`vocab_word2idx` key)."
        )
        if st.button("Create placeholder model with minimal vocabulary"):
            vocab = PersianVocabulary()
            for i, w in enumerate(["خوب", "بد", "متوسط", "عالی", "ضعیف", "مثبت", "منفی"], start=4):
                vocab.word2idx[w] = i
                vocab.idx2word[i] = w
            model = SentimentClassifier(vocab_size=len(vocab))
            save_model_with_vocab(model, vocab)
            st.rerun()
        return

    predictor = SentimentPredictor(model, vocab, max_length, device)

    # Sidebar — model info
    with st.sidebar:
        st.divider()
        st.header("🧠 Model Info")
        if checkpoint:
            st.caption(f"Vocab size: {len(vocab)}")
            st.caption(f"Max length: {max_length}")
            if "val_accuracy" in checkpoint:
                st.caption(f"Val accuracy: {checkpoint['val_accuracy']:.2%}")

    # ── Main Input Area ───────────────────────────────────────────────────────
    col_input, col_status = st.columns([3, 1])

    with col_input:
        st.subheader("📝 Enter Persian Text")

        example_texts = {
            "✅ Good":    "این محصول واقعا عالی بود. کیفیت فوق العاده",
            "⚪ Neutral": "محصول متوسطی بود، نه خوب نه بد",
            "❌ Bad":     "بدترین خریدم بود، پولم را هدر دادم",
        }
        ex_cols = st.columns(3)
        selected_example = None
        for idx, (label, txt) in enumerate(example_texts.items()):
            with ex_cols[idx]:
                if st.button(label, use_container_width=True):
                    selected_example = txt

        text_input = st.text_area(
            "Text to analyse:",
            value=selected_example or "",
            height=110,
            placeholder="مثال: محصول بدی بود، کیفیت پایینی داشت...",
            key="text_input",
        )

        api_key_missing = not api_key
        analyze_clicked = st.button(
            "🔍 Analyse Sentiment",
            type="primary",
            disabled=not text_input.strip() or api_key_missing,
            use_container_width=True,
        )
        if api_key_missing:
            st.caption("⚠️ Enter your OpenRouter API key in the sidebar to enable analysis.")

    with col_status:
        st.subheader("Status")
        current_step = st.session_state.get("current_status", "Ready")
        progress_map = {
            "Ready": 0,
            "Starting analysis...": 10,
            "Running ML Model...": 30,
            f"Checking threshold ({THRESHOLD:.0%})...": 50,
            "Bad sentiment detected — sending to Qwen3...": 70,
            "Not bad sentiment — skipping LLM": 80,
            "Analysis complete: OK": 100,
            "Analysis complete: BAD": 100,
        }
        progress = progress_map.get(current_step, 0)
        status_color = {
            "Ready": "blue",
            "Starting analysis...": "blue",
            "Running ML Model...": "blue",
            "Bad sentiment detected — sending to Qwen3...": "red",
            "Not bad sentiment — skipping LLM": "green",
            "Analysis complete: OK": "green",
            "Analysis complete: BAD": "red",
        }.get(current_step, "gray")

        st.markdown(
            f"""<div style="background-color:{status_color}20;padding:15px;
            border-radius:10px;border-left:5px solid {status_color};margin-bottom:15px;">
            <h4 style="margin:0;color:{status_color};">{current_step}</h4></div>""",
            unsafe_allow_html=True,
        )
        st.progress(progress / 100)
        st.caption(f"Progress: {progress}%")

    # ── Run Analysis ──────────────────────────────────────────────────────────
    if analyze_clicked and text_input.strip():
        st.session_state["current_status"] = "Starting analysis..."

        pipeline_container = st.container()
        results_container = st.container()
        details_container = st.container()

        with pipeline_container:
            st.subheader("Pipeline Flow")
            fig = create_pipeline_diagram({"active_nodes": ["start"], "active_edges": []})
            pipeline_ph = st.empty()
            pipeline_ph.plotly_chart(fig, use_container_width=True)

        result = run_analysis_pipeline(text_input, predictor, api_key)

        if result:
            st.session_state["current_status"] = result["pipeline_status"]["current_step"]
            st.session_state["analysis_history"].append(result)

            with pipeline_container:
                pipeline_ph.plotly_chart(
                    create_pipeline_diagram(result["pipeline_status"]),
                    use_container_width=True,
                )

            # ── Result Cards ──────────────────────────────────────────────────
            with results_container:
                st.subheader("Analysis Results")
                c1, c2, c3, c4 = st.columns(4)

                suggestion = result["ml_result"]["suggestion"]
                sentiment_label = {1: "Good", 2: "Neutral", 3: "Bad"}.get(suggestion, "?")
                c1.metric("ML Prediction", f"{sentiment_label} ({suggestion})")
                c2.metric("Bad Probability", f"{result['ml_result']['probabilities']['bad']:.1%}")

                llm_res = result["llm_result"]
                if llm_res:
                    c3.metric("Qwen3 Confidence", f"{llm_res.get('confidence', 0):.1%}")
                else:
                    c3.metric("Qwen3 Check", "Skipped")

                decision = result["final_decision"]
                color = "red" if decision == "BAD" else "green"
                icon = "⚠️" if decision == "BAD" else "✅"
                with c4:
                    st.markdown(
                        f"""<div style="background-color:{color}20;padding:10px;
                        border-radius:5px;border-left:5px solid {color};">
                        <h3 style="margin:0;color:{color};">Final Decision</h3>
                        <h1 style="margin:0;">{icon} {decision}</h1></div>""",
                        unsafe_allow_html=True,
                    )

            # ── Detail Tabs ───────────────────────────────────────────────────
            with details_container:
                tab1, tab2, tab3 = st.tabs(["📊 Probabilities", "🤖 Qwen3 Details", "⏱️ Timeline"])

                with tab1:
                    st.plotly_chart(
                        create_sentiment_gauge(result["ml_result"]["probabilities"], suggestion),
                        use_container_width=True,
                    )
                    prob_data = {
                        "Sentiment": ["Good (1)", "Neutral (2)", "Bad (3)"],
                        "Probability": [
                            f"{result['ml_result']['probabilities']['good']:.2%}",
                            f"{result['ml_result']['probabilities']['neutral']:.2%}",
                            f"{result['ml_result']['probabilities']['bad']:.2%}",
                        ],
                        "Threshold Check": [
                            "N/A", "N/A",
                            "✓ Pass" if result["ml_result"]["probabilities"]["bad"] > THRESHOLD else "✗ Fail",
                        ],
                    }
                    st.dataframe(pd.DataFrame(prob_data), use_container_width=True)

                with tab2:
                    if llm_res:
                        left, right = st.columns([1, 2])
                        with left:
                            st.info("**Qwen3 Response**")
                            st.json(llm_res)
                        with right:
                            st.info("**Interpretation**")
                            is_bad = llm_res.get("is_bad", False)
                            confidence = llm_res.get("confidence", 0.0)
                            if is_bad and confidence > CONFIDENCE_THRESHOLD:
                                st.error("### ⚠️ Qwen3 Confirms: BAD SENTIMENT")
                                st.write(f"**Confidence**: {confidence:.1%} (>{CONFIDENCE_THRESHOLD:.0%})")
                            elif is_bad:
                                st.warning("### ⚠️ Qwen3 detects bad sentiment but with low confidence")
                                st.write(f"**Confidence**: {confidence:.1%} (<{CONFIDENCE_THRESHOLD:.0%})")
                            else:
                                st.success("### ✅ Qwen3 says: NOT BAD")
                            st.write(f"**Reasoning**: {llm_res.get('reasoning', 'No reasoning provided')}")
                    else:
                        st.info("Qwen3 verification was skipped — ML model did not flag bad sentiment above threshold.")

                with tab3:
                    if result["steps_data"]:
                        st.plotly_chart(create_timeline(result["steps_data"]), use_container_width=True)
                        st.dataframe(pd.DataFrame(result["steps_data"]), use_container_width=True)

            # ── History ───────────────────────────────────────────────────────
            with st.expander("🕑 Analysis History (last 10)"):
                h = st.session_state["analysis_history"]
                if h:
                    rows = [
                        {
                            "#": i,
                            "Text": (r["text"][:50] + "...") if len(r["text"]) > 50 else r["text"],
                            "ML": f"Suggestion {r['ml_result']['suggestion']}",
                            "Bad Prob": f"{r['ml_result']['probabilities']['bad']:.1%}",
                            "Decision": r["final_decision"],
                        }
                        for i, r in enumerate(h[-10:], 1)
                    ]
                    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
                else:
                    st.info("No history yet")


# ==================== 11. Entry Point ====================
if __name__ == "__main__":
    main()
