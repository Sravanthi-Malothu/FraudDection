# Sentra Fraud Detection & Alert Investigation Platform
# Universal Container for Local Docker, Render, and Hugging Face Spaces
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app \
    PORT=7860

WORKDIR /app

# Install system dependencies (OpenMP for XGBoost, curl for health check)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgomp1 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install pinned Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Create standard non-root user (UID 1000) required by Hugging Face Spaces
RUN useradd -m -u 1000 user

# Copy application code, trained models, documentation, and lightweight data caches
COPY app/ ./app/
COPY src/ ./src/
COPY models/ ./models/
COPY docs/ ./docs/
COPY data/alerts_cache.json ./data/
COPY data/sample_test_transactions.json ./data/

# Ensure data directory exists with write permissions for SQLite database persistence
RUN mkdir -p /app/data && \
    chown -R user:user /app && \
    chmod -R 777 /app/data

# Switch to non-root user for security and Hugging Face Spaces compatibility
USER user

# Expose Hugging Face Spaces port (7860) and standard port (5001)
EXPOSE 7860 5001

# Health check adapted to dynamic port
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD curl -f http://localhost:${PORT:-7860}/health || exit 1

# Production WSGI server using Gunicorn dynamically binding to $PORT (7860 on HF, dynamic on Render)
CMD ["sh", "-c", "gunicorn --bind 0.0.0.0:${PORT:-7860} --workers 2 --threads 2 --timeout 60 app.app:app"]
