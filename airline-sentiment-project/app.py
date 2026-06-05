from flask import Flask, request, jsonify
import pickle, re, nltk
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from nltk.corpus import stopwords

app = Flask(__name__)

model = tokenizer = le = stop_words = None  # lazy globals

def initialize_resources():
    nltk.download('stopwords', quiet=True)
    try:
        m = load_model('nn_model.keras')
        with open('tokenizer.pkl', 'rb') as f: tok = pickle.load(f)
        with open('label_encoder.pkl', 'rb') as f: enc = pickle.load(f)
        return m, tok, enc, set(stopwords.words('english'))
    except Exception:
        # Gracefully degrade in test/CI environments
        return None, None, None, set(stopwords.words('english'))

def get_resources():
    global model, tokenizer, le, stop_words
    if model is None:
        model, tokenizer, le, stop_words = initialize_resources()
    return model, tokenizer, le, stop_words

def clean_text(text):
    if not isinstance(text, str): return ""
    _, _, _, sw = get_resources()
    text = re.sub(r'http\S+|@\w+|#\w+|[^a-zA-Z\s]', '', text.lower())
    return " ".join([w for w in text.split() if w not in sw])

@app.route('/health', methods=['GET'])
def health():
    return jsonify({'status': 'healthy'})

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    if not data or 'tweet' not in data:
        return jsonify({'error': 'No tweet provided'}), 400
    m, tok, enc, _ = get_resources()
    cleaned = clean_text(data['tweet'])
    seq = pad_sequences(tok.texts_to_sequences([cleaned]), maxlen=50)
    prediction = m.predict(seq).argmax(axis=1)
    sentiment = enc.inverse_transform(prediction)[0]
    return jsonify({'tweet': data['tweet'], 'sentiment': str(sentiment)})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)