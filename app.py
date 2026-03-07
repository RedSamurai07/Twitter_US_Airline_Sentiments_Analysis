from flask import Flask, request, jsonify
import pickle
import re
import nltk
from nltk.corpus import stopwords

app = Flask(__name__)

# Load the brain (Model) and the translator (Vectorizer)
model = pickle.load(open('model.pkl', 'rb'))
vectorizer = pickle.load(open('vectorizer.pkl', 'rb'))
stop_words = set(stopwords.words('english'))

def clean_text(text):
    text = text.lower()
    text = re.sub(r'http\S+|@\w+|#\w+|[^a-zA-Z\s]', '', text)
    tokens = [word for word in text.split() if word not in stop_words]
    return " ".join(tokens)

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    if not data or 'tweet' not in data:
        return jsonify({'error': 'No tweet provided'}), 400
    
    # 1. Preprocess the incoming tweet
    cleaned = clean_text(data['tweet'])
    
    # 2. Vectorize (Must use .transform, NOT .fit_transform)
    vectorized = vectorizer.transform([cleaned]).toarray()
    
    # 3. Predict
    prediction = model.predict(vectorized)[0]
    
    return jsonify({
        'tweet': data['tweet'],
        'sentiment': prediction
    })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)