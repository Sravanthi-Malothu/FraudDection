"""
Root app.py — Production WSGI Entrypoint for Render and Cloud Deployments.
Exposes both 'app' and 'application' so that:
  - gunicorn app:app
  - gunicorn app:application
  - gunicorn wsgi:app
  - python app.py
all work seamlessly out-of-the-box.
"""
import os
import sys
from pathlib import Path

# Ensure repository root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Import initialized Flask application
from app.app import app, application

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port, debug=False)
