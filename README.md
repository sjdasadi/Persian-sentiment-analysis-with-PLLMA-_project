# 🔍 Persian Sentiment Analysis with PLLMA

A hybrid Persian text sentiment analysis pipeline that combines a **BiLSTM deep learning model** with **Qwen3 LLM** (via OpenRouter) to classify Persian text into three categories: ✅ Good, ⚪ Neutral, and ❌ Bad.

---

## 🧠 How It Works

The pipeline uses a **two-stage approach**:

1. **Stage 1 — BiLSTM ML Model**: A bidirectional LSTM with attention mechanism analyzes the input text and predicts sentiment probabilities.
2. **Stage 2 — Qwen3 LLM Verification** *(only if bad sentiment is detected)*: If the ML model's "bad" probability exceeds the threshold (`10%`), the text is sent to `Qwen3-27B` via OpenRouter for a second opinion and confidence score.

```
Input Text ──► BiLSTM Model ──► Bad prob > 10%? ──► YES ──► Qwen3 LLM ──► Final Decision
                                        │
                                       NO
                                        │
                                  Final Decision: OK
```

---

## 📁 Project Structure

```
project2/
│
├── app.py                      # Main Streamlit application
└── best_sentiment_model.pth    # Trained BiLSTM model checkpoint
```

---

## ⚙️ Requirements

Install the required packages:

```bash
pip install streamlit torch langchain langchain-openai plotly pandas
```

---

## 🚀 How to Run

### 1. Get an OpenRouter API Key

- Go to [https://openrouter.ai](https://openrouter.ai) and create a free account
- Generate an API key from your dashboard

### 2. Run the App

```bash
streamlit run app.py
```

### 3. Use the App

1. Open your browser at `http://localhost:8501`
2. Enter your **OpenRouter API Key** in the sidebar
3. Type or paste any **Persian text** into the input box
4. Click **"🔍 Analyse Sentiment"**
5. View the results:
   - ML model prediction and probabilities
   - Qwen3 LLM verification (if triggered)
   - Final decision: ✅ OK or ⚠️ BAD

---

## 🖥️ App Features

| Feature | Description |
|---|---|
| 📝 Text Input | Enter any Persian text manually or use built-in examples |
| 📊 Probability Chart | Visual gauge showing Good / Neutral / Bad probabilities |
| 🤖 Qwen3 Details | LLM reasoning and confidence score |
| ⏱️ Timeline | Step-by-step execution time for each pipeline stage |
| 🕑 History | Last 10 analysis results saved during the session |

---

## 🔧 Configuration

You can adjust the following constants in `app.py`:

```python
THRESHOLD = 0.1            # ML bad-probability threshold to trigger LLM (default: 10%)
CONFIDENCE_THRESHOLD = 0.6 # Minimum LLM confidence to confirm BAD sentiment (default: 60%)
MODEL_NAME = "qwen/qwen3.6-27b"  # OpenRouter model identifier
```

---

## 📌 Example Texts

| Label | Example |
|---|---|
| ✅ Good | `این محصول واقعا عالی بود. کیفیت فوق العاده` |
| ⚪ Neutral | `محصول متوسطی بود، نه خوب نه بد` |
| ❌ Bad | `بدترین خریدم بود، پولم را هدر دادم` |

---

## 🛠️ Model Architecture

- **Type**: BiLSTM + Attention
- **Classes**: 3 (Good, Neutral, Bad)
- **Input**: Persian text (tokenized, max 100 tokens)
- **Checkpoint**: `best_sentiment_model.pth`

---

## 📄 License

This project is for educational purposes as part of an LLM course.
