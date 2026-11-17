"""
wsgi.py — Web Server Gateway Interface entrypoint.
Enables 'gunicorn wsgi:app' and standard WSGI runners.
"""

from app.app import app

if __name__ == "__main__":
    import os
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)
