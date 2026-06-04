import streamlit as st
import pickle
import re
import nltk
from nltk.corpus import stopwords
import os

# 1. Page Configuration
st.set_page_config(page_title="Twitter US Airline Sentiment Analysis", layout="centered")
st.title("✈️ US Airline Sentiment Predictor")
st.markdown("Enter a tweet or airline review below to evaluate customer sentiment.")

# 2. Quietly download NLTK dependencies and cache them
@st.cache_resource
def download_nltk_dependencies():
    nltk.download('stopwords', quiet=True)
    return set(stopwords.words('english'))

stop_words = download_nltk_dependencies()

# 3. Cached Model Loading
@st.cache_resource
def load_pipeline():
    # Since model_pipeline.pkl is in the exact same folder as app.py,
    # we use a relative path.
    model_path = os.path.join(os.path.dirname(__file__), 'model_pipeline.pkl')
    with open(model_path, 'rb') as f:
        return pickle.load(f)

try:
    model_pipeline = load_pipeline()
except FileNotFoundError:
    st.error("⚠️ Critical Error: 'model_pipeline.pkl' was not found in the streamlit folder!")
    st.stop()

# 4. Text Preprocessing Function
def clean_text(text):
    if not isinstance(text, str):
        return ""
    text = text.lower()
    # Remove URLs, Mentions, Hashtags, and special characters
    text = re.sub(r'http\S+|@\w+|\#\w+|[^a-zA-Z\s]', '', text)
    tokens = [word for word in text.split() if word not in stop_words]
    return " ".join(tokens)

# 5. User Interface
user_review = st.text_area("Review Text:", placeholder="Type your airline review here...")

if st.button("Analyze Sentiment", type="primary"):
    if user_review.strip() != "":
        # Clean the input text
        cleaned = clean_text(user_review)
        
        # Predict using your model pipeline
        prediction = model_pipeline.predict([cleaned])[0]
        
        # Format the output label string cleanly (handling both upper/lowercase models)
        prediction_str = str(prediction).strip().lower()
        
        # Display Results
        st.subheader("Analysis Result:")
        if "pos" in prediction_str:
            st.success("🟢 Positive Sentiment")
        elif "neg" in prediction_str:
            st.error("🔴 Negative Sentiment")
        else:
            st.warning("🟡 Neutral Sentiment")
    else:
        st.warning("Please enter some text before analyzing!")