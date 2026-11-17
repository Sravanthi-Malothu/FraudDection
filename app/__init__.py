"""
app package initialization.
Exposes 'app' and 'application' so that both 'gunicorn app:app' and 'gunicorn app.app:app' work.
"""
import sys
from pathlib import Path

# Ensure project root is in sys.path
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from .app import app

application = app

__all__ = ["app", "application"]
