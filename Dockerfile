# 1. Use an official Python base image (matching your 3.10 version)
FROM python:3.13-slim

# 2. Set the directory inside the container
WORKDIR /app

# 3. Copy only the requirements first (helps with caching)
COPY requirements.txt .

# 4. Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# 5. Download NLTK data (Critical for your sentiment analysis)
RUN python -m nltk.downloader stopwords

# 6. Copy the rest of your code and models
COPY . .

# 7. Expose the port Flask runs on
EXPOSE 5000

# 8. Command to run your app using Gunicorn (production grade)
CMD ["gunicorn", "--bind", "0.0.0.0:5000", "app:app"]