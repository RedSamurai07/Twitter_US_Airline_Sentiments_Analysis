import pandas as pd
import re
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.pipeline import Pipeline
import nltk
from nltk.corpus import stopwords
import os

# --- MLflow Setup ---
mlflow.set_experiment("Airline_Sentiment_Analysis")
# autolog() tracks parameters like n_estimators automatically
mlflow.sklearn.autolog() 

# NLTK setup
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

def clean_text(text):
    if not isinstance(text, str): return ""
    text = text.lower()
    # Remove URLs, Mentions, Hashtags, and non-alpha characters
    text = re.sub(r'http\S+|@\w+|#\w+|[^a-zA-Z\s]', '', text)
    tokens = [word for word in text.split() if word not in stop_words]
    return " ".join(tokens)

# 1. Setup & Data Loading
# Ensure Tweets.csv is in the same directory
if not os.path.exists('Tweets.csv'):
    raise FileNotFoundError("Tweets.csv not found! Please ensure it is in the project root.")

df = pd.read_csv('Tweets.csv')
df['cleaned_text'] = df['text'].apply(clean_text)

X = df['cleaned_text']
y = df['airline_sentiment']

# 2. Split Data (80% Train, 20% Test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

with mlflow.start_run(run_name="RandomForest_Pipeline_Run") as run:
    # 3. Create a Pipeline 
    # This bundles the vectorizer and model into one object
    pipeline = Pipeline([
        ('tfidf', TfidfVectorizer(max_features=2500)),
        ('rf', RandomForestClassifier(n_estimators=200, n_jobs=-1))
    ])

    # 4. Model Training
    pipeline.fit(X_train, y_train)
    
    # 5. Evaluation
    y_pred = pipeline.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    # Manually log the metric (though autolog does some of this)
    mlflow.log_metric("accuracy", accuracy) 
    print(f"Model Accuracy: {accuracy * 100:.2f}%")
    
    # 6. Log the entire Pipeline to MLflow
    # This allows you to deploy directly from MLflow later
    mlflow.sklearn.log_model(pipeline, artifact_path="model")
    
    # 7. Save locally for your Docker/Flask setup
    import pickle
    with open('model_pipeline.pkl', 'wb') as f:
        pickle.dump(pipeline, f)

    print(f"Run ID: {run.info.run_id}")
    print("Model and Pipeline tracked in MLflow and saved as model_pipeline.pkl!")