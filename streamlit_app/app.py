import streamlit as st
import requests
import pandas as pd
from datetime import datetime

# ──────────────────────────────────────────────
# CONFIG
# ──────────────────────────────────────────────

API_URL = "https://sentiment-review-api.onrender.com"  # ← change to your deployed URL later
APP_TITLE = "Product Review Sentiment Analyzer"

# ──────────────────────────────────────────────
# SESSION STATE INIT
# ──────────────────────────────────────────────

if "results" not in st.session_state:
    st.session_state.results = []
if "current_input" not in st.session_state:
    st.session_state.current_input = ""

# ──────────────────────────────────────────────
# UI LAYOUT
# ──────────────────────────────────────────────

st.set_page_config(page_title=APP_TITLE, layout="wide")
st.title(APP_TITLE)
st.markdown("Paste a review or use examples below → get sentiment + confidence instantly")

# ── Example Buttons ────────────────────────────────────────────────────────
st.subheader("Quick Test Examples")

cols = st.columns(5)

examples = [
    ("Positive", "Love this! Super fast charging and the battery lasts all day. Best earbuds I've owned!"),
    ("Negative", "Arrived damaged, poor quality plastic, stopped working after 2 uses. Total waste of money."),
    ("Neutral", "It's okay. Does the job but feels a bit cheap compared to more expensive brands."),
    ("Mixed", "Great picture quality but battery drains way too fast and the app is buggy."),
    ("Sarcastic", "Wow... amazing product. Really love how it broke on day one. 5 stars 🙄")
]

for col, (label, text) in zip(cols, examples):
    if col.button(label, use_container_width=True):
        st.session_state.current_input = text
        st.rerun()  # refresh to show in textarea

# ── Main Input ─────────────────────────────────────────────────────────────
review_input = st.text_area(
    "Enter product review(s) here...",
    value=st.session_state.current_input,
    height=140,
    placeholder="Paste one review or multiple (one per line)",
    key="review_area"
)

analyze_btn = st.button("Analyze", type="primary", use_container_width=True)

# ── Processing ─────────────────────────────────────────────────────────────
if analyze_btn and review_input.strip():
    with st.spinner("Analyzing..."):
        lines = [line.strip() for line in review_input.split("\n") if line.strip()]
        
        if not lines:
            st.warning("No valid reviews found.")
        else:
            results = []
            for text in lines:
                try:
                    payload = {"text": text}
                    resp = requests.post(API_URL, json=payload, timeout=12)
                    if resp.status_code == 200:
                        data = resp.json()
                        data["original_text"] = text
                        data["timestamp"] = datetime.now().strftime("%H:%M:%S")
                        results.append(data)
                    else:
                        results.append({"original_text": text, "error": resp.text})
                except Exception as e:
                    results.append({"original_text": text, "error": str(e)})
            
            st.session_state.results = results + st.session_state.results  # append new to top
            st.success(f"Analyzed {len(results)} review(s)")

# ── Results Display ────────────────────────────────────────────────────────
if st.session_state.results:
    st.subheader("Analysis Results")
    
    result_rows = []
    for r in st.session_state.results:
        row = {"Review": r.get("original_text", "")[:150] + ("..." if len(r.get("original_text", "")) > 150 else "")}
        
        if "error" in r:
            row["Sentiment"] = "ERROR"
            row["Confidence"] = "-"
            row["Details"] = r["error"]
        else:
            sent = r["sentiment"].upper()
            conf_pct = r["confidence"] * 100
            row["Sentiment"] = sent
            row["Confidence"] = f"{conf_pct:.1f}%"
            row["Positive"] = f"{r['probabilities'].get('positive', 0) * 100:.1f}%"
            row["Neutral"]  = f"{r['probabilities'].get('neutral', 0) * 100:.1f}%"
            row["Negative"] = f"{r['probabilities'].get('negative', 0) * 100:.1f}%"
            row["Time"] = r.get("timestamp", "-")
        
        result_rows.append(row)
    
    df = pd.DataFrame(result_rows)
    
    # Color styling for sentiment
    def highlight_sentiment(val):
        if val == "POSITIVE": return "background-color: #d4edda; color: #155724;"
        if val == "NEGATIVE": return "background-color: #f8d7da; color: #721c24;"
        if val == "NEUTRAL":  return "background-color: #fff3cd; color: #856404;"
        if val == "ERROR":    return "background-color: #e2e3e5;"
        return ""
    
    styled_df = df.style.map(highlight_sentiment, subset=["Sentiment"])
    
    st.dataframe(styled_df, use_container_width=True, hide_index=True)
    
    # Probabilities bar chart – safe version using dict directly
    if st.session_state.results and "probabilities" in st.session_state.results[0]:
        st.subheader("Probability Breakdown (Latest Review)")
        latest_probs = st.session_state.results[0]["probabilities"]
        st.bar_chart(latest_probs)   # Streamlit handles dict → bars automatically

    # Download
    csv = df.to_csv(index=False).encode('utf-8')
    st.download_button("Download Results CSV", csv, "reviews_sentiment.csv", "text/csv")

    if st.button("Clear All Results"):
        st.session_state.results = []
        st.rerun()
import os

# Use environment variable for flexibility (local vs cloud)
BACKEND_URL = os.getenv("BACKEND_URL", "http://127.0.0.1:8000")
API_URL = f"{BACKEND_URL}/predict"