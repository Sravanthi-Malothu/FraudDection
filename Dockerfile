# Sentra Fraud & AML Intelligence Platform - Containerized API & Dashboard
FROM python:3.11-slim

# Prevent Python from writing .pyc files and enable unbuffered output
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PORT=5001 \
    PYTHONPATH=/app

WORKDIR /app

# Install system dependencies (libgomp1 for XGBoost OpenMP acceleration, curl for container health check)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgomp1 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install pinned Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application, ML components, documentation benchmarks, and alert cache
COPY app/ ./app/
COPY src/ ./src/
COPY models/ ./models/
COPY docs/ ./docs/
COPY data/alerts_cache.json ./data/

# Ensure runtime directory for SQLite decisions database exists
RUN mkdir -p data

EXPOSE 5001

# Container health monitoring
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD curl -f http://localhost:5001/health || exit 1

# Production WSGI server using Gunicorn
CMD ["gunicorn", "--bind", "0.0.0.0:5001", "--workers", "2", "--threads", "2", "--timeout", "60", "app.app:app"]
