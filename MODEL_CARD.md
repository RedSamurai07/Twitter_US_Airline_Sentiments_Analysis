# Model Card: Twitter Airline Sentiment Classifier

## 1. Model Objective

This model is designed to perform binary/multiclass classification on social media feedback to categorize customer sentiment (Positive, Negative, Neutral). The primary objective is to enable automated triage of customer service requests, allowing support teams to prioritize high-priority negative sentiment.

## 2. Intended Use

  - Primary Use: Automated routing of social media feedback to customer support queues.

  - Secondary Use: Generating high-level sentiment trends for marketing and service improvement reports.

Out-of-Scope: This model is not intended for automated social media moderation or legal decision-making, as it does not account for complex emotional nuance or sarcasm.

## 3. Training & Validation Data
   
   - Source: Publicly available US Airline Twitter Sentiment dataset.

   - Preprocessing: Implemented text cleaning (removal of handles, URLs, and special characters) using modularized Python scripts to ensure a standardized input pipeline.

   Validation Split: 70/30 to ensure performance metrics are representative of unseen data.

## 4. Performance Metrics

   - Primary Metric: F1-Score (Used due to class imbalance in the training data).
   
   - Secondary Metrics: Precision and Recall per class to ensure the model does not disproportionately misclassify negative sentiment.
   
   - Current Baseline: 81%

## 5. Limitations & Ethical Considerations

   - Sarcasm & Nuance: The model relies on linguistic patterns and may struggle with highly sarcastic inputs which are common on platforms like Twitter.

   - Bias: The model was trained on historical data. If the underlying data contains biases related to specific airline demographics or region-based language, the model may propagate these biases.

   - Data Drift: Sentiment language evolves rapidly. Without a scheduled retraining pipeline (outlined in the Deployment Roadmap), model performance is expected to decay over time.

## 6. Operational Guardrails

  - Input Validation: The system rejects inputs with excessive length or non-text characters to prevent downstream processing errors.

  - Confidence Thresholds: In a production setting, low-confidence predictions should be flagged for human-in-the-loop review rather than automated action.