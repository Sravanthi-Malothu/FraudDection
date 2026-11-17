"""
app/__init__.py — Package entrypoint for the Sentra Web Application.
Exposes the Flask 'app' instance so that both:
  1. gunicorn app:app
  2. gunicorn app.app:app
resolve cleanly in deployment environments like Render.
"""

from app.app import app

__all__ = ["app"]
