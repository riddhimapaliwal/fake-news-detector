import streamlit as st
import joblib, re, spacy, pathlib

st.set_page_config(page_title="Fake News Detector", page_icon="📰")
st.title("📰 Fake News Detector")

@st.cache_resource
def load_assets():
    nlp = spacy.load('en_core_web_sm', disable=['ner','parser'])
    model = joblib.load(pathlib.Path(__file__).parent/'models'/'model.pkl')
    return nlp, model

nlp, assets = None, None
try:
    nlp, assets = load_assets()
except Exception as e:
    st.error("Model not found. Run `python model_train.py` first.")
    st.stop()

vectorizer, clf = assets['vectorizer'], assets['clf']

def clean(text):
    doc = nlp(text.lower())
    return ' '.join([tok.lemma_ for tok in doc if not tok.is_stop and tok.is_alpha])

text_input = st.text_area("Paste a news article:", height=200)
if st.button("Predict") and text_input.strip():
    text_clean = clean(text_input)
    tfidf = vectorizer.transform([text_clean])
    pred = clf.decision_function(tfidf)
    label = "FAKE" if pred < 0 else "REAL"
    confidence = abs(pred[0])
    st.subheader(f"Prediction: {label}")
    st.write(f"Confidence score: {confidence:.2f}")
