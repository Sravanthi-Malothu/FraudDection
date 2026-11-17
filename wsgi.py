"""
wsgi.py — Web Server Gateway Interface entrypoint.
Standard WSGI entrypoint for Render, Gunicorn, and cloud production servers.
"""
import sys
from pathlib import Path

# Ensure project root is in sys.path
_ROOT = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from app.app import app

# Provide both standard WSGI variable names
application = app

if __name__ == "__main__":
    import os
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)
