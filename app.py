import re
import joblib
import streamlit as st

# Page Configuration
st.set_page_config(
    page_title="Disaster Tweet Detector", page_icon="🚨", layout="centered"
)


# Load trained model and vectorizer
@st.cache_resource
def load_assets():
    model = joblib.load("disaster_model.pkl")
    vectorizer = joblib.load("tfidf_vectorizer.pkl")
    return model, vectorizer


model, vectorizer = load_assets()


# Cleaning function matching training step
def clean_tweet(text):
    text = text.lower()
    text = re.sub(r"https?://\S+|www\.\S+", "", text)
    text = re.sub(r"<.*?>|&[a-z]+;", "", text)
    text = re.sub(r"[^a-z\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


# Header
st.title("🚨 Disaster Tweet Classifier")
st.write(
    "This app uses NLP to determine whether a tweet describes a **real disaster** or uses **metaphorical language**."
)

# Text Input Box
user_input = st.text_area(
    "Enter a Tweet or phrase:",
    placeholder="e.g., Structural fire reported downtown, emergency crews dispatched.",
)

# Interactive Sample Buttons
st.write("Or try a sample:")
col1, col2 = st.columns(2)

if col1.button("🔥 Test Metaphor Slang"):
    user_input = "This new track is absolute fire, listening on repeat!"
if col2.button("🚒 Test Real Emergency"):
    user_input = (
        "Emergency alert: Wildfire spreading rapidly near Highway 101."
    )

# Prediction Logic
if st.button("Classify Tweet", type="primary") or user_input:
    if user_input.strip() != "":
        cleaned = clean_tweet(user_input)
        vec = vectorizer.transform([cleaned])
        pred = model.predict(vec)[0]
        prob = model.predict_proba(vec)[0]

        st.divider()
        if pred == 1:
            confidence = prob[1] * 100
            st.error(
                f"🚨 **REAL DISASTER DETECTED** ({confidence:.1f}% confidence)"
            )
        else:
            confidence = prob[0] * 100
            st.success(
                f"🟢 **NOT A DISASTER / METAPHOR** ({confidence:.1f}% confidence)"
            )

        st.caption(f"**Cleaned Input:** `{cleaned}`")
    else:
        st.warning("Please enter a sentence to test.")