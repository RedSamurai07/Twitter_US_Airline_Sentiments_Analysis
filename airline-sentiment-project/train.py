import os
import re
import pickle
import pandas as pd
import nltk
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout

# Setup NLTK
try:
    nltk.data.find("corpora/stopwords")
except LookupError:
    nltk.download("stopwords")
stop_words = set(stopwords.words("english"))

def clean_text(text):
    if not isinstance(text, str): return ""
    text = text.lower()
    text = re.sub(r'http\S+|@\w+|#\w+|[^a-zA-Z\s]', '', text)
    return " ".join([w for w in text.split() if w not in stop_words])

if __name__ == "__main__":
    df = pd.read_csv("Tweets.csv")
    df["cleaned"] = df["text"].apply(clean_text)

    # 1. Prepare Labels
    le = LabelEncoder()
    y = le.fit_transform(df["airline_sentiment"])

    # 2. Tokenize & Pad
    tokenizer = Tokenizer(num_words=5000)
    tokenizer.fit_on_texts(df["cleaned"])
    X = pad_sequences(tokenizer.texts_to_sequences(df["cleaned"]), maxlen=50)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    # 3. Build Model
    model = Sequential([
        Embedding(5000, 64, input_length=50),
        LSTM(64),
        Dense(3, activation='softmax') # Assuming 3 sentiment classes
    ])
    model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.fit(X_train, y_train, epochs=5, batch_size=32)

    # 4. Save Artifacts (The 'Inference' bundle)
    model.save("nn_model.keras")
    with open("tokenizer.pkl", "wb") as f: pickle.dump(tokenizer, f)
    with open("label_encoder.pkl", "wb") as f: pickle.dump(le, f)

    print("Training complete. Artifacts saved: nn_model.keras, tokenizer.pkl, label_encoder.pkl")