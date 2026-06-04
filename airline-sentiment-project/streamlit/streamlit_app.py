import streamlit as st
import pickle
import re
import nltk
from nltk.corpus import stopwords
import urllib.request

# 1. Page Configuration
st.set_page_config(page_title="Twitter US Airline Sentiment Analysis", layout="centered")
st.title("✈️ US Airline Sentiment Predictor")
st.markdown("Enter a tweet or airline review below to evaluate customer sentiment.")

# Your live Dropbox URL to the 123MB model pipeline file
DROPBOX_URL = "https://www.dropbox.com/scl/fi/htun6y9crwzw0fdfqx6tm/model_pipeline.pkl?rlkey=g5dwk2ed686axjpwmnifodw9r&st=y8lag9nn&dl=1"

# 2. Quietly download NLTK dependencies and cache them
@st.cache_resource
def download_nltk_dependencies():
    nltk.download('stopwords', quiet=True)
    return set(stopwords.words('english'))

stop_words = download_nltk_dependencies()

# 3. Download the large model from Dropbox directly into cache memory
@st.cache_resource
def load_pipeline_from_url(url):
    try:
        # Request configuration to bypass basic firewalls
        req = urllib.request.Request(
            url, 
            headers={'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'}
        )
        with urllib.request.urlopen(req) as response:
            return pickle.loads(response.read())
    except Exception as e:
        st.error(f"⚠️ Failed to download model from cloud storage: {e}")
        return None

# Spinner ensures the user knows it's downloading the large file on first startup
with st.spinner("📦 Fetching machine learning model from cloud storage... (Takes a few seconds on first load)"):
    model_pipeline = load_pipeline_from_url(DROPBOX_URL)

if model_pipeline is None:
    st.markdown("### 🔍 Troubleshooting Tips:")
    st.write("1. Check if the Dropbox link has expired or changed.")
    st.write("2. Make sure your `requirements.txt` includes the exact versions of the machine learning libraries you trained the model with (like `scikit-learn`).")
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

# 5. User Interface Setup
user_review = st.text_area("Review Text:", placeholder="Type your airline review here...")

if st.button("Analyze Sentiment", type="primary"):
    if user_review.strip() != "":
        # Clean the text input matching the model training expectations
        cleaned = clean_text(user_review)
        
        # Predict using the cached remote model pipeline
        prediction = model_pipeline.predict([cleaned])[0]
        prediction_str = str(prediction).strip().lower()
        
        # Render clean metric visualization blocks
        st.subheader("Analysis Result:")
        if "pos" in prediction_str:
            st.success("🟢 Positive Sentiment")
        elif "neg" in prediction_str:
            st.error("🔴 Negative Sentiment")
        else:
            st.warning("🟡 Neutral Sentiment")
    else:
        st.warning("Please enter some text before analyzing!")