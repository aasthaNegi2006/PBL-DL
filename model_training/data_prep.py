"""Build the frame cache from raw videos.
Entry point for model_training folder — delegates to src.data_prep (source of truth).

Usage:
    python -m model_training.data_prep --raw-dir data/raw
"""
from src.data_prep import *  # noqa: F401, F403
from src.data_prep import main

if __name__ == '__main__':
    main()
