import streamlit as st
import joblib
import re, string

# ─── Load Models & Vectorizer ───────────────────────────────────────────────────
models = {
    "Logistic Regression": joblib.load("LR_TFIDF_model.jb"),
    "Random Forest"      : joblib.load("RF_TFIDF_model.jb"),
}
vectorizer = joblib.load("tfidf_vectorizer.jb")

# ─── Text Cleaning (same as notebook) ──────────────────────────────────────────
def clean_text(text):
    text = text.lower()
    text = re.sub(r'\[.*?\]', '', text)
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    text = re.sub(r'<.*?>+', '', text)
    text = re.sub(f"[{re.escape(string.punctuation)}]", '', text)
    text = re.sub(r'\n', ' ', text)
    text = re.sub(r'\w*\d\w*', '', text)
    return text

# ─── Streamlit UI ───────────────────────────────────────────────────────────────
st.set_page_config(page_title="Fake News Detector", layout="centered")
st.title("📰 Fake News Detector")
st.write("Select a model and paste a news article below to classify it.")

# Model selector
choice = st.selectbox("Choose Model", list(models.keys()))
model = models[choice]

# Text input area
input_text = st.text_area("News Article:", height=200)

if st.button("Check News"):
    if not input_text.strip():
        st.warning("⚠️ Please enter some text.")
    else:
        cleaned = clean_text(input_text)
        if len(cleaned.split()) < 10:
            st.warning("⚠️ Please enter a longer article for reliable results.")
        else:
            vec = vectorizer.transform([cleaned])
            proba = model.predict_proba(vec)[0]
            pred = model.predict(vec)[0]
            conf = round(max(proba) * 100, 2)

            if conf < 60:
                st.info(f"🤔 Uncertain (Confidence: {conf}%)")
            elif pred == 1:
                st.success(f"✅ Real News (Confidence: {conf}%)")
            else:
                st.error(f"🚫 Fake News (Confidence: {conf}%)")
