from flask import Flask, request, jsonify
import pickle
import re
import nltk
from nltk.corpus import stopwords

app = Flask(__name__)

# 1. Initialization Function
# This allows tests to import 'app' without loading the model every time
def initialize_resources():
    nltk.download('stopwords', quiet=True)
    with open('model_pipeline.pkl', 'rb') as f:
        pipeline = pickle.load(f)
    return pipeline, set(stopwords.words('english'))

# Initialize once globally
model_pipeline, stop_words = initialize_resources()

def clean_text(text):
    """Standardizes input text."""
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r'http\S+|@\w+|#\w+|[^a-zA-Z\s]', '', text)
    tokens = [word for word in text.split() if word not in stop_words]
    return " ".join(tokens)

@app.route('/health', methods=['GET'])
def health():
    return jsonify({"status": "healthy"}), 200

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    if not data or 'tweet' not in data:
        return jsonify({'error': 'No tweet provided'}), 400
    
    raw_tweet = data['tweet']
    cleaned_tweet = clean_text(raw_tweet)
    
    # Predict using the Pipeline
    prediction = model_pipeline.predict([cleaned_tweet])[0]
    
    return jsonify({
        'tweet': raw_tweet,
        'sentiment': str(prediction)
    })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)