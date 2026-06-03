import os
import re
import pickle
import pandas as pd
import nltk
import mlflow
import mlflow.sklearn

from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report

# Download stopwords if not already available
try:
    nltk.data.find("corpora/stopwords")
except LookupError:
    nltk.download("stopwords")

stop_words = set(stopwords.words("english"))


def clean_text(text):
    """
    Clean tweet text by:
    - converting to lowercase
    - removing URLs
    - removing mentions
    - removing hashtags
    - removing special characters
    - removing stopwords
    """

    if not isinstance(text, str):
        return ""

    text = text.lower()

    text = re.sub(
        r'http\S+|www\S+|@\w+|#\w+|[^a-zA-Z\s]',
        '',
        text
    )

    tokens = [
        word for word in text.split()
        if word not in stop_words
    ]

    return " ".join(tokens)


if __name__ == "__main__":

    print("Starting Airline Sentiment Training Pipeline...")

    # MLflow Setup
    mlflow.set_experiment("Airline_Sentiment_Analysis")
    mlflow.sklearn.autolog()

    # Check dataset
    if not os.path.exists("Tweets.csv"):
        raise FileNotFoundError(
            "Tweets.csv not found in project directory."
        )

    # Load dataset
    df = pd.read_csv("Tweets.csv")

    # Verify required columns
    required_columns = ["text", "airline_sentiment"]

    for column in required_columns:
        if column not in df.columns:
            raise ValueError(
                f"Required column '{column}' not found in dataset."
            )

    print(f"Dataset Shape: {df.shape}")

    # Clean text
    df["cleaned_text"] = df["text"].apply(clean_text)

    X = df["cleaned_text"]
    y = df["airline_sentiment"]

    # Train/Test Split
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y
    )

    with mlflow.start_run(run_name="RandomForest_Pipeline_Run") as run:

        # Build Pipeline
        pipeline = Pipeline([
            (
                "tfidf",
                TfidfVectorizer(max_features=2500)
            ),
            (
                "rf",
                RandomForestClassifier(
                    n_estimators=200,
                    random_state=42,
                    n_jobs=-1
                )
            )
        ])

        # Train
        pipeline.fit(X_train, y_train)

        # Predict
        y_pred = pipeline.predict(X_test)

        # Accuracy
        accuracy = accuracy_score(y_test, y_pred)

        print(f"\nAccuracy: {accuracy:.4f}")

        # Classification Report
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred))

        # Log metric
        mlflow.log_metric("accuracy", accuracy)

        # Log model
        mlflow.sklearn.log_model(
            pipeline,
            artifact_path="model"
        )

        # Save pipeline locally
        with open("model_pipeline.pkl", "wb") as f:
            pickle.dump(pipeline, f)

        print("\nModel saved as model_pipeline.pkl")
        print(f"MLflow Run ID: {run.info.run_id}")
        print("Training completed successfully.")