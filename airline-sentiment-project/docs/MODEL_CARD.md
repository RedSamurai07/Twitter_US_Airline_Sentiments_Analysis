# Model Card: Twitter US Airline Sentiments Analysis

---

## 1. Model Details

| Field | Details |
|---|---|
| **Framework Name** | Twitter US Airline Sentiment Analysis — Multi-Model NLP Framework |
| **Python Version** | 3.10 |
| **Analysis Date** | March 2026 |
| **Production Model** | LSTM Neural Network (TensorFlow/Keras) |
| **Supporting Models** | Gaussian Naive Bayes, Multinomial Naive Bayes, Logistic Regression, Linear SVM, Random Forest |
| **Primary Metric** | Accuracy + Weighted F1-Score |
| **Secondary Metrics** | Precision, Recall per sentiment class (Negative / Neutral / Positive) |
| **Text Representation** | TF-IDF (ML models) · Tokenizer + Padding (LSTM) |
| **API Framework** | Flask |
| **Live App** | [Streamlit App](https://twitterusairlinesentimentsanalysis-cm6xlenrkvdosevattlrgf.streamlit.app/) |

---

## 2. Intended Use

- **Primary Use Case:** Classify Twitter feedback towards US airlines as Negative, Neutral, or Positive to provide actionable intelligence for Customer Service, Operations, Marketing, and Senior Management teams.
- **Target Users:** Airline Customer Service Teams, Operations Analysts, Marketing & Communications Managers, CRM Strategists.
- **Out of Scope:** Real-time tweet streaming classification, non-English language tweets, airlines outside the 6 covered in the dataset.

---

## 3. Dataset

| Property | Value |
|---|---|
| **Source File** | `Tweets.csv` |
| **Total Tweets** | 14,640 |
| **Airlines Covered** | 6 (United, US Airways, American, Southwest, Delta, Virgin America) |
| **Timeframe** | February 2015 (US domestic flight tweets) |
| **Label Type** | Pre-classified sentiment + negative reason category |

**Sentiment Distribution:**

| Sentiment | Count | Percentage |
|---|---|---|
| Negative | 9,178 | **62.7%** |
| Neutral | 3,099 | 21.2% |
| Positive | 2,363 | 16.1% |

> The dataset is heavily imbalanced — negative sentiment dominates. `class_weight='balanced'` was applied in ML models to address this.

**Dataset Schema:**

| Feature | Description | Data Type |
|---|---|---|
| `tweet_id` | Unique tweet identifier | int64 |
| `airline_sentiment` | Classified sentiment: negative / neutral / positive | object |
| `airline_sentiment_confidence` | Model confidence for sentiment label (0–1) | float64 |
| `negativereason` | Reason for negative feedback (if applicable) | object |
| `negativereason_confidence` | Confidence score for negative reason label | float64 |
| `airline` | Airline mentioned in tweet | object |
| `name` | Twitter username | object |
| `retweet_count` | Number of retweets | int64 |
| `text` | Full tweet content | object |
| `tweet_coord` | Geographic coordinates (mostly missing) | object |
| `tweet_created` | Timestamp of tweet | object |
| `tweet_location` | User-provided location | object |
| `user_timezone` | User timezone setting | object |

**Airline Tweet Volumes:**

| Airline | Tweets | % Negative | % Positive |
|---|---|---|---|
| United | 3,822 | 68.9% | — |
| US Airways | 2,913 | **77.7%** | — |
| American | 2,759 | 71.0% | — |
| Southwest | 2,420 | — | — |
| Delta | 2,222 | — | 24.5% |
| Virgin America | 504 | — | **30.2%** ← highest positive |

---

## 4. Data Preprocessing

**Missing Value Handling:**

| Column | Missing Rate | Action |
|---|---|---|
| `airline_sentiment_gold` | High | Dropped entirely |
| `negativereason_gold` | High | Dropped entirely |
| `tweet_coord` | High | Dropped entirely |
| `negativereason` | Partial | Filled with `'Others'` |
| `negativereason_confidence` | Partial | Filled with column mean |
| `tweet_location` | Partial | Filled with `'No location'` |
| `user_timezone` | Partial | Filled with `'No Timezone'` |

**Text Cleaning Pipeline:**
```python
def comprehensive_clean(text):
    text = text.lower()
    text = re.sub(r'https?://\S+|www\.\S+', '', text)  # Remove URLs
    text = re.sub(r'\S+@\S+', '', text)                # Remove emails
    text = text.replace('@', '')                        # Remove @ symbols
    text = re.sub(r'^\w+\s+', '', text)                # Remove leading airline handle
    text = re.sub(r'[^\w\s]', '', text)                # Remove punctuation
    words = word_tokenize(text)
    return " ".join([w for w in words if w not in stop_words])
```

**Label Encoding:** Negative = 0, Neutral = 1, Positive = 2

---

## 5. Feature Engineering

**TF-IDF (for ML models):**
- Vocabulary: Top-frequency words from cleaned tweet corpus
- Output: Sparse matrix → converted to dense array for Naive Bayes / Random Forest

**Tokenizer + Padding (for LSTM):**

| Parameter | Value |
|---|---|
| `num_words` (vocab size) | 5,000 |
| `maxlen` (sequence length) | 50 tokens |
| Padding | Post-padding with zeros |
| Embedding dimension | 128 |

---

## 6. Methodology & Pipeline Architecture

┌─────────────────────────────────────────────────────────────────────┐
│                        1. DATA LAYER                                │
│  Tweets.csv (14,640 tweets | 6 US airlines | Feb 2015)              │
│         │                                                           │
│         ▼  load_and_clean_data()                                    │
│  Unified DataFrame (label-encoded, missing values handled)          │
└──────────────────────────┬──────────────────────────────────────────┘
                           │
┌──────────────────────────▼──────────────────────────────────────────┐
│                     2. TEXT PREPROCESSING                           │
│  comprehensive_clean()                                              │
│  ├── Lowercase + URL / email removal                                │
│  ├── @ symbol + airline handle stripping                            │
│  ├── Punctuation removal + NLTK word tokenization                   │
│  └── Stopword removal                                               │
│                                                                     │
│  Label Encoding: Negative=0 | Neutral=1 | Positive=2                │
└──────────────────────────┬──────────────────────────────────────────┘
                           │
┌──────────────────────────▼──────────────────────────────────────────┐
│                     3. FEATURE ENGINEERING                          │
│  ├── TF-IDF Vectorizer       → sparse matrix (ML models)            │
│  └── Keras Tokenizer + Padding → sequences of length 50 (LSTM)      │
│       vocab_size=5000 | maxlen=50 | embedding_dim=128               │
└──────────────────────────┬──────────────────────────────────────────┘
                           │
┌──────────────────────────▼──────────────────────────────────────────┐
│                  4. EDA & STATISTICAL VALIDATION                    │
│  ├── Sentiment distribution analysis (imbalance detection)          │
│  ├── Airline-level breakdown (negative reasons, volumes)            │
│  ├── Geographic & temporal pattern analysis                         │
│  └── 6 Hypothesis tests (Chi-Square, ANOVA, Pearson, Tukey HSD)     │
└──────────────────────────┬──────────────────────────────────────────┘
                           │
┌──────────────────────────▼──────────────────────────────────────────┐
│                     5. MODEL TRAINING                               │
│  Train/Test Split: 70% train | 30% test                             │
│  class_weight='balanced' applied to all models                      │
│                                                                     │
│  ├── Baseline: Gaussian NB / Multinomial NB (TF-IDF)                │
│  ├── Classical: Logistic Regression, Linear SVM, Random Forest      │
│  └── Production: LSTM (TensorFlow/Keras)                            │
│       Embedding(5000,128) → SpatialDropout1D(0.4)                   │
│       → LSTM(128) → Dense(3, softmax)                               │
│                                                                     │
│  Loss: Sparse Categorical Crossentropy | Optimiser: Adam            │
│  EarlyStopping: val_loss, patience=3, restore_best_weights=True     │
└──────────────────────────┬──────────────────────────────────────────┘
                           │
┌──────────────────────────▼──────────────────────────────────────────┐
│                  6. EXPERIMENT TRACKING (MLflow)                    │
│  Logs: params, metrics (Accuracy / F1 / Precision / Recall)         │
│  Artifacts: nn_model.keras · tokenizer.pkl · label_encoder.pkl      │
└──────────────────────────┬──────────────────────────────────────────┘
                           │
┌──────────────────────────▼──────────────────────────────────────────┐
│                  7. PACKAGING & CI/CD                               │
│  ├── nn_model.keras → serialized artifact                           │
│  ├── Docker → containerized environment (Dockerfile)                │
│  ├── GitHub Actions → CI pipeline (Pytest + pytest-cov on push)     │
│  └── Codecov → coverage reporting                                   │
└──────────────────────────┬──────────────────────────────────────────┘
                           │
┌──────────────────────────▼──────────────────────────────────────────┐
│                     8. PRODUCTION SERVING                           │
│  ├── Flask API (POST /predict, GET /health) on AWS EC2              │
│  ├── Streamlit frontend (Streamlit Cloud)                           │
│  └── MLflow artifact registry for experiment management             │
└─────────────────────────────────────────────────────────────────────┘


## 7. Model Comparison

All models trained on 70% of data (30% held out for evaluation), with `class_weight='balanced'` for imbalanced classes:

| Model | Accuracy | Notes |
|---|---|---|
| Gaussian Naive Bayes | 52% | Baseline — struggles with dense TF-IDF |
| Multinomial Naive Bayes | ~60–65% | Better than Gaussian for text count features |
| Logistic Regression | ~70% | Strong linear baseline with balanced weights |
| Linear SVM | ~72% | Best classical ML performer on TF-IDF |
| Random Forest | ~70% | Competitive; useful for feature importance |
| **LSTM (production)** | **~80%** | Best overall — captures word sequence context |

**Why LSTM was selected:**
- Captures sequential word dependencies (e.g., "not good" ≠ "good")
- Uses gating mechanisms to preserve long-range tweet context
- Overcomes vanishing gradient problem of standard RNNs
- Outperformed all classical ML models by ~8–10pp

---

## 8. Production Model — LSTM Architecture

```
Embedding(5000, 128, input_length=50)
    ↓
SpatialDropout1D(0.4)
    ↓
LSTM(128, dropout=0.2, recurrent_dropout=0.2)
    ↓
Dense(3, activation='softmax')
```

**Training Configuration:**

| Parameter | Value |
|---|---|
| Loss | Sparse Categorical Crossentropy |
| Optimiser | Adam |
| Batch Size | 32 |
| Max Epochs | 15 |
| Early Stopping | `val_loss`, patience=3, restore_best_weights=True |
| Train / Test Split | 80% / 20% |
| Final Test Accuracy | ~80% |

**Saved Artifacts:**
- `nn_model.keras` — trained LSTM model
- `tokenizer.pkl` — fitted Keras tokeniser
- `label_encoder.pkl` — fitted LabelEncoder

---

## 9. Top Negative Reasons

| Reason | Count |
|---|---|
| Customer Service Issue | **2,910** |
| Late Flight | 1,665 |
| Can't Tell | 1,190 |
| Cancelled Flight | 847 |
| Lost Luggage | 724 |

**Airline-level negative reason breakdown:**
- **Delta, Southwest, United:** Miscellaneous issues (online cancellation, food service, wait times, seat reservation problems)
- **American, US Airways, United, Southwest:** Customer Service Issues
- **United:** Booking problems
- **American:** Late/delayed flights

---

## 10. Average Sentiment Confidence

| Sentiment | Avg Confidence |
|---|---|
| Negative | **0.933** — highest confidence, easiest to classify |
| Positive | 0.872 |
| Neutral | 0.823 — hardest to classify |

> High confidence on negative labels validates the dataset quality for training the negative class, which comprises 62.7% of all tweets.

---

## 11. Geographic & Temporal Analysis

**Top cities by negative tweet volume:**
Washington D.C., New York City, Los Angeles, Chicago, Boston

**Location-specific negative reasons:**
| City | Top Negative Reasons |
|---|---|
| Chicago | Bad flights, miscellaneous (hygiene, food, safety) |
| Austin, TX | Cancelled flights, customer support, booking issues |
| Boston, MA | Flight attendant complaints, lost luggage, late flights |
| Brooklyn, NY | Late flights, long lines |

**Timezone patterns:**
| Timezone | Notable Finding |
|---|---|
| Eastern (US & Canada) | Highest tweet volume — most negative |
| Amsterdam | Lower negative rate (48.6%), higher neutral — outlier |
| Quito | High negative rate (70.8%) despite smaller volume |

**Temporal patterns:**
- Peak negative tweeting: **Sunday** (highest day), **morning and night hours**
- Lowest tweet volumes: **Wednesday and Thursday**

---

## 12. Hypothesis Testing Results

### Test 1 — Chi-Square: Negative Reason Distribution
- **H₀:** Negative reasons are uniformly distributed
- **Result:** p < 0.001 → ✅ **REJECT H₀**
- "Customer Service Issue" and "Late Flight" are disproportionately over-represented

### Test 2 — Chi-Square: Airline vs Sentiment Independence
- **H₀:** Sentiment distribution is independent of airline
- **Result:** p < 0.001 → ✅ **REJECT H₀**
- Airline choice significantly predicts sentiment; Virgin America vs US Airways are statistical opposites

### Test 3 — Pearson Correlation: Confidence vs Negative Likelihood
- **Result:** Statistically significant correlation (p < 0.05)
- Higher confidence scores are slightly associated with negative classification
- Effect size: weak positive correlation

### Test 4 — ANOVA: Retweet Count by Sentiment
- **H₀:** No difference in mean retweet count across sentiments
- **Result:** Significant (p < 0.05) → ✅ **REJECT H₀**
- Tukey HSD post-hoc confirms significant pairwise differences

### Test 5 — Chi-Square: Day/Hour vs Sentiment
- **Result:** p < 0.05 → ✅ **REJECT H₀**
- Sentiment distribution differs significantly by day of week and hour of day

### Test 6 — Chi-Square: Timezone vs Sentiment
- **Result:** p < 0.05 → ✅ **REJECT H₀**
- User timezone significantly affects sentiment distribution

---

## 13. Flask API Endpoints

| Endpoint | Method | Description |
|---|---|---|
| `/health` | GET | Returns `{"status": "healthy"}` |
| `/predict` | POST | Classifies a tweet → Negative / Neutral / Positive |

**Request:**
```json
{ "tweet": "The flight was late and the service was terrible." }
```
**Response:**
```json
{ "tweet": "...", "sentiment": "negative" }
```

---

## 14. Ethical Considerations & Limitations

- **Class Imbalance:** 62.7% of tweets are negative. Models may still exhibit bias toward negative classification for ambiguous tweets. `class_weight='balanced'` partially mitigates this.
- **Temporal Scope:** All tweets are from February 2015. Airline service quality, customer expectations, and social media behaviour have changed significantly — retraining on recent data is essential.
- **Geographic Gaps:** Over 3,000 tweets lack location data, limiting reliable geographic analysis. Results for specific cities/timezones should be treated as directional, not causal.
- **"Can't Tell" Category:** 1,190 tweets are labelled as "Can't Tell" for negative reason — these represent genuine annotation uncertainty and reduce model precision on edge cases.
- **Single Platform:** Twitter feedback may not represent the full customer population. Frequent Twitter users may skew toward more vocal complainers.
- **Language:** The model is trained exclusively on English-language tweets. Non-English tweets from international passengers would not be reliably classified.

---

## 15. Infrastructure & Tools

| Category | Tool |
|---|---|
| Language | Python 3.10 |
| Deep Learning | TensorFlow / Keras (LSTM, Embedding, SpatialDropout1D) |
| Classical ML | Scikit-learn (Logistic Regression, LinearSVC, Random Forest, GaussianNB, MultinomialNB) |
| Text Processing | NLTK (stopwords, word_tokenize), Keras Tokenizer, TF-IDF Vectorizer |
| Visualisation | Matplotlib, Seaborn, Plotly, WordCloud |
| Statistical Tests | SciPy (chi2_contingency, chisquare, pearsonr), Statsmodels (ANOVA, Tukey HSD) |
| API Framework | Flask |
| Frontend | Streamlit |
| Experiment Tracking | MLflow |
| Testing | Pytest + pytest-cov |
| Coverage Reporting | Codecov |
| CI/CD | GitHub Actions |
| Containerisation | Docker |
| Cloud Infrastructure | AWS EC2 |
| Version Control | Git |
| Data Processing | Pandas, NumPy |

---

## 16. Final Decision Summary

```
══════════════════════════════════════════════════════════════
        TWITTER AIRLINE SENTIMENT — EXECUTIVE SUMMARY
══════════════════════════════════════════════════════════════
Dataset:         14,640 tweets | 6 US airlines | Feb 2015
Classes:         Negative (62.7%) | Neutral (21.2%) | Positive (16.1%)
Production Model: LSTM — Accuracy ~80%
══════════════════════════════════════════════════════════════
AIRLINE RANKINGS:
Most Negative:   US Airways (77.7%) | American (71.0%)
Most Positive:   Virgin America (30.2%) | Delta (24.5%)
Top Complaint:   Customer Service Issue (2,910 tweets)
══════════════════════════════════════════════════════════════
KEY DESIGN DECISIONS:
1. LSTM chosen over classical ML — captures sequential word context
2. Balanced class weights — corrects for 62.7% negative imbalance
3. EarlyStopping — prevents overfitting on imbalanced data
4. TF-IDF retained for classical model comparison baseline
5. 6 hypothesis tests — statistically validates all key findings
══════════════════════════════════════════════════════════════
PRODUCTION RECOMMENDATIONS:
• Retrain quarterly as new tweet data is collected
• Increase Sunday + morning/night customer service staffing
• Prioritise Customer Service & Late Flight resolution for US Airways, American
• Conduct Core Web Vitals review for United Airways booking portal
• Learn from Virgin America's playbook — apply to underperforming carriers
• Monitor Amsterdam timezone for positive signal amplification
══════════════════════════════════════════════════════════════
```
