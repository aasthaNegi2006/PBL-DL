"""Evaluate a trained checkpoint on the held-out signer-independent test set.
Entry point for model_training folder — delegates to src.evaluate (source of truth).

Usage:
    python -m model_training.evaluate
    python -m model_training.evaluate --checkpoint models/best_isl_model.pth
"""
from src.evaluate import *  # noqa: F401, F403
from src.evaluate import main

if __name__ == '__main__':
    main()
