# 1. Base Image
FROM python:3.13-slim

# 2. Set environment variables
ENV PYTHONUNBUFFERED=1

# 3. Working Directory
WORKDIR /app

# 4. Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 5. Download NLTK data
RUN python -m nltk.downloader stopwords

# 6. Copy local model files (Ensure these exist before building!)
COPY Tweets.csv .
COPY train.py .
COPY app.py .
COPY model.pkl .     
COPY vectorizer.pkl .

# 7. Expose Port
EXPOSE 5000

# 8. Start Gunicorn (Production server)
CMD ["gunicorn", "--bind", "0.0.0.0:5000", "app:app"]