"""Run leave-one-signer-out cross-validation.
Entry point for model_training folder — delegates to src.cross_validate (source of truth).

Usage:
    python -m model_training.cross_validate
"""
from src.cross_validate import *  # noqa: F401, F403
from src.cross_validate import main

if __name__ == '__main__':
    main()
