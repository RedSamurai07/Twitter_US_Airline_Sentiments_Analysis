import pandas as pd
import re
import pickle
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import nltk
from nltk.corpus import stopwords

# --- MLflow Setup ---
mlflow.set_experiment("Airline_Sentiment_Analysis")
mlflow.sklearn.autolog() 

nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

def clean_text(text):
    if not isinstance(text, str): return ""
    text = text.lower()
    text = re.sub(r'http\S+|@\w+|#\w+|[^a-zA-Z\s]', '', text)
    tokens = [word for word in text.split() if word not in stop_words]
    return " ".join(tokens)

with mlflow.start_run(run_name="RandomForest_Base_Run"):
    # 1. Setup & Data Loading
    df = pd.read_csv('Tweets.csv')
    df['cleaned_text'] = df['text'].apply(clean_text)

    # 2. Feature Extraction
    vectorizer = TfidfVectorizer(max_features=2500)
    X = vectorizer.fit_transform(df['cleaned_text']).toarray()
    y = df['airline_sentiment'].values  # .values converts it to a clean array

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 3. Model Training
    model = RandomForestClassifier(n_estimators=200)
    model.fit(X_train, y_train)
    
    # 4. Evaluation (This fuels your metrics/badges)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    mlflow.log_metric("accuracy", accuracy) 
    print(f"Model Accuracy: {accuracy * 100:.2f}%")
    
    # 5. Save locally
    with open('model.pkl', 'wb') as f:
        pickle.dump(model, f)
    with open('vectorizer.pkl', 'wb') as f:
        pickle.dump(vectorizer, f)

    print("Model tracked in MLflow and saved locally!")