from flask import Flask, request, jsonify
import pickle
import re
import nltk
from nltk.corpus import stopwords

app = Flask(__name__)

# Load Model and Vectorizer
model = pickle.load(open('model.pkl', 'rb'))
vectorizer = pickle.load(open('vectorizer.pkl', 'rb'))
stop_words = set(stopwords.words('english'))

def clean_text(text):
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
    
    cleaned = clean_text(data['tweet'])
    vectorized = vectorizer.transform([cleaned]).toarray()
    prediction = model.predict(vectorized)[0]
    
    return jsonify({
        'tweet': data['tweet'],
        'sentiment': prediction
    })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)