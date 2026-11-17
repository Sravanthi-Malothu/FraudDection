"""
gunicorn.conf.py — Gunicorn configuration file.
Automatically picked up by Gunicorn to guarantee correct binding to $PORT,
threading, and memory optimization for Render's 512 MB free tier.
"""
import os

# Dynamic port binding for Render and cloud environments
port = os.environ.get("PORT", "10000")
bind = f"0.0.0.0:{port}"

# Concurrency tuned for 512 MB RAM free tier
workers = 1
threads = 2
worker_class = "gthread"
timeout = 60
keepalive = 5

# Logging
accesslog = "-"
errorlog = "-"
loglevel = "info"
