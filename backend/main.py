"""FastAPI backend entry point for the ISL Emergency Gesture Recognition system.
This module re-exports the app from app/main.py so you can run the server from either location.

Usage:
    # From project root (recommended — matches the .bat launchers):
    uvicorn app.main:app --host 127.0.0.1 --port 8000

    # Alternatively, from model_training folder perspective:
    uvicorn backend.main:app --host 127.0.0.1 --port 8000
"""
from app.main import app  # noqa: F401

__all__ = ['app']
