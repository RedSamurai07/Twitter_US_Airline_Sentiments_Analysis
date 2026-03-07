from flask import Flask, request, jsonify
import pickle
import re
import nltk
from nltk.corpus import stopwords

app = Flask(__name__)

# 1. Load the Pipeline (Contains both Vectorizer + Model)
# We use the 'model_pipeline.pkl' created in your updated train.py
with open('model_pipeline.pkl', 'rb') as f:
    model_pipeline = pickle.load(f)

# Download NLTK stopwords if not already present
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

def clean_text(text):
    """
    Standardizes the input text just like the training script.
    """
    if not isinstance(text, str):
        return ""
    text = text.lower()
    # Remove URLs, Mentions, Hashtags, and special characters
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
    
    # 2. Preprocess the text
    cleaned_tweet = clean_text(raw_tweet)
    
    # 3. Predict using the Pipeline
    # The pipeline automatically runs .transform() then .predict()
    # We pass it as a list [cleaned_tweet] because the model expects an iterable
    prediction = model_pipeline.predict([cleaned_tweet])[0]
    
    # Optional: Get probability scores
    # probs = model_pipeline.predict_proba([cleaned_tweet])[0]
    
    return jsonify({
        'tweet': raw_tweet,
        'sentiment': prediction
    })

if __name__ == '__main__':
    # Use 0.0.0.0 to make it accessible within Docker/AWS
    app.run(host='0.0.0.0', port=5000)