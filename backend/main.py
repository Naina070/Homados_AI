"""Compatibility shim. Prefer `uvicorn app.main:app --app-dir backend`."""
from app.main import app

__all__ = ["app"]
