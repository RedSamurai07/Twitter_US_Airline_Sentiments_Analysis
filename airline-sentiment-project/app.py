from flask import Flask, request, jsonify
import pickle
import re
import nltk
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from nltk.corpus import stopwords

app = Flask(__name__)

# Load artifacts safely
def initialize_resources():
    nltk.download('stopwords', quiet=True)
    model = load_model('nn_model.keras')
    with open('tokenizer.pkl', 'rb') as f: tokenizer = pickle.load(f)
    with open('label_encoder.pkl', 'rb') as f: le = pickle.load(f)
    return model, tokenizer, le, set(stopwords.words('english'))

# Global scope
model, tokenizer, le, stop_words = initialize_resources()

def clean_text(text):
    if not isinstance(text, str): return ""
    text = re.sub(r'http\S+|@\w+|#\w+|[^a-zA-Z\s]', '', text.lower())
    return " ".join([w for w in text.split() if w not in stop_words])

@app.route('/health', methods=['GET'])
def health():
    return jsonify({"status": "healthy"}), 200

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    if not data or 'tweet' not in data:
        return jsonify({'error': 'No tweet provided'}), 400
    
    cleaned = clean_text(data['tweet'])
    seq = pad_sequences(tokenizer.texts_to_sequences([cleaned]), maxlen=50)
    
    prediction = model.predict(seq).argmax(axis=1)
    sentiment = le.inverse_transform(prediction)[0]
    
    return jsonify({'tweet': data['tweet'], 'sentiment': str(sentiment)})