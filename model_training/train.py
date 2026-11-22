"""Train the ISL model with a signer-independent split.
Entry point for model_training folder — delegates to src.train (source of truth).

Usage:
    python -m model_training.train
    python -m model_training.train --epochs 40 --test-subjects 022 023 024 025 026
"""
# Re-export everything from src.train so this module works identically.
from src.train import *  # noqa: F401, F403
from src.train import main

if __name__ == '__main__':
    main()
