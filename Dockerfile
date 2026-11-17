# Base image: official Python 3.11 slim for Render's free tier (512 MB RAM)
FROM python:3.11-slim

# Environment settings
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app \
    PORT=10000

WORKDIR /app

# Install system dependencies (libgomp1 for XGBoost OpenMP acceleration, curl for healthcheck)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgomp1 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install pinned Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code, trained models, documentation, and committed 5k sample
COPY app/ ./app/
COPY app.py ./app.py
COPY wsgi.py ./wsgi.py
COPY gunicorn.conf.py ./gunicorn.conf.py
COPY src/ ./src/
COPY models/ ./models/
COPY docs/ ./docs/
COPY data/alerts_cache.json ./data/
COPY data/sample_transactions.csv ./data/
COPY data/sample_test_transactions.json ./data/

# Ensure SQLite runtime directory exists with read/write permissions
RUN mkdir -p /app/data && chmod -R 777 /app/data

EXPOSE 10000 7860 5001

# Container health monitoring
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD curl -f http://localhost:${PORT:-10000}/health || exit 1

# Serve with Gunicorn (1 worker, 2 threads) to stay comfortably within 512 MB RAM
CMD ["sh", "-c", "gunicorn --bind 0.0.0.0:${PORT:-10000} --workers 1 --threads 2 --timeout 60 wsgi:app"]
