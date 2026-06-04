import streamlit as st
import pickle
import re
import nltk
from nltk.corpus import stopwords
import urllib.request
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np
import os

# 1. Page Configuration
st.set_page_config(page_title="Twitter US Airline Sentiment Analysis", layout="centered")
st.title("✈️ US Airline Sentiment Predictor")
st.markdown("Enter a tweet or airline review below to evaluate customer sentiment.")

# 🔗 PASTE YOUR DROPBOX LINKS HERE (Ensure they all end in dl=1)
NN_MODEL_URL = "https://www.dropbox.com/scl/fi/4ydvvmtekzqfo6lwvl0uz/nn_model.keras?rlkey=kze33ukbfkxtgog6i9gj66uw9&st=fio3apnj&dl=1"
TOKENIZER_URL = "https://www.dropbox.com/scl/fi/7jpbrzonhixd3kl65gjju/tokenizer.pkl?rlkey=nkifa5ov0bn2u428zi3amzqyt&st=5eyhpl29&dl=1"
LABEL_ENCODER_URL = "https://www.dropbox.com/scl/fi/qibv0eqzq4xhvwxh00izz/label_encoder.pkl?rlkey=j3w18yynzq7qajafhlhox3jen&st=m4t83o8d&dl=1"

# 2. Downloading all NLTK dependencies
@st.cache_resource
def download_nltk_dependencies():
    nltk.download('stopwords', quiet=True)
    from nltk.corpus import stopwords  
    return set(stopwords.words('english'))

stop_words = download_nltk_dependencies()

# 3. Load all Neural Network Assets from Cloud Storage into Cache Memory
@st.cache_resource
def load_cloud_assets(model_url, tokenizer_url, encoder_url):
    try:
        temp_model_path = "temp_model.keras"
        headers = {'User-Agent': 'Mozilla/5.0'}
        
        # Download the .keras network model structure
        req_model = urllib.request.Request(model_url, headers=headers)
        with urllib.request.urlopen(req_model) as response, open(temp_model_path, 'wb') as out_file:
            out_file.write(response.read())
        nn_model = tf.keras.models.load_model(temp_model_path)
        
        if os.path.exists(temp_model_path):
            os.remove(temp_model_path)
            
        # Download Tokenizer
        req_tok = urllib.request.Request(tokenizer_url, headers=headers)
        with urllib.request.urlopen(req_tok) as response:
            tokenizer = pickle.loads(response.read())
            
        # Download Label Encoder
        req_enc = urllib.request.Request(encoder_url, headers=headers)
        with urllib.request.urlopen(req_enc) as response:
            label_encoder = pickle.loads(response.read())
            
        return nn_model, tokenizer, label_encoder
    except Exception as e:
        st.error(f"⚠️ Deep Learning Asset Download Failed: {e}")
        return None, None, None

with st.spinner("🧠 Reassembling Neural Network layers... (Takes a minute on first load)"):
    model, tokenizer, le = load_cloud_assets(NN_MODEL_URL, TOKENIZER_URL, LABEL_ENCODER_URL)

if model is None or tokenizer is None or le is None:
    st.stop()

# 4. Text Preprocessing
def clean_text(text):
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r'http\S+|@\w+|\#\w+|[^a-zA-Z\s]', '', text)
    tokens = [word for word in text.split() if word not in stop_words]
    return " ".join(tokens)

# 5. User Interface Setup
user_review = st.text_area("Review Text:", placeholder="Type your airline review here...")

if st.button("Analyze Sentiment", type="primary"):
    if user_review.strip() != "":
        cleaned = clean_text(user_review)
        
        # Tokenize and Pad the string sequence just like your notebook does
        seq = tokenizer.texts_to_sequences([cleaned])
        padded = pad_sequences(seq, maxlen=50) # Change 50 to match your notebook's max_len if different
        
        # Run prediction through Neural Network layers
        pred_probs = model.predict(padded)
        pred_class_idx = np.argmax(pred_probs, axis=1)
        
        # Map back to string class dynamically ('positive', 'neutral', 'negative')
        prediction_str = str(le.inverse_transform(pred_class_idx)[0]).strip().lower()
        
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