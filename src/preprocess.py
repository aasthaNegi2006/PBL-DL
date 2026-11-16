"""Single source of truth for turning video frames into model input.

Data preparation, training, evaluation, the desktop webcam app and the FastAPI
server all use these functions, so the model sees the same kind of input
everywhere. If you change anything here, rebuild the frame cache
(`python -m src.data_prep`) and retrain.
"""
import random

import cv2
import numpy as np
import torch

IMG_SIZE = 112
NUM_FRAMES = 16
MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)

# Live camera framing. Training clips are the dataset's pre-cropped videos, so
# live frames are cropped to a centred square to approximate that framing
# (signer's upper body and hands). `size` is a fraction of the shorter side.
# Dataset clips (uploads, --source file.avi) are already cropped: don't crop them again.
DEFAULT_ROI = {'cx': 0.5, 'cy': 0.5, 'size': 0.9}

# Used when a checkpoint does not record how long training clips were.
DEFAULT_WINDOW_SECONDS = 2.5
DEFAULT_FPS = 30.0

# Mean absolute pixel change (0-255) between sampled frames below which the
# input is treated as "no gesture". Camera-dependent; a "none" class is the
# better long-term fix (see `python -m src.data_prep --none-from`).
DEFAULT_MOTION_THRESHOLD = 2.5

_MEAN_T = torch.tensor(MEAN).view(1, 3, 1, 1)
_STD_T = torch.tensor(STD).view(1, 3, 1, 1)


def roi_box(width, height, roi=None):
    """Return (x0, y0, side) of the square crop region inside a width x height frame."""
    roi = roi or DEFAULT_ROI
    side = int(round(roi['size'] * min(width, height)))
    side = max(1, min(side, width, height))
    x0 = int(round(roi['cx'] * width - side / 2))
    y0 = int(round(roi['cy'] * height - side / 2))
    x0 = min(max(0, x0), width - side)
    y0 = min(max(0, y0), height - side)
    return x0, y0, side


def prepare_frame(frame_bgr, crop=False, roi=None):
    """OpenCV BGR frame -> uint8 RGB array of shape (IMG_SIZE, IMG_SIZE, 3)."""
    if crop:
        h, w = frame_bgr.shape[:2]
        x0, y0, side = roi_box(w, h, roi)
        frame_bgr = frame_bgr[y0:y0 + side, x0:x0 + side]
    resized = cv2.resize(frame_bgr, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_AREA)
    return cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)


def uniform_indices(total_frames, num_frames=NUM_FRAMES):
    """Evenly spaced frame indices across the whole clip (eval and live inference)."""
    return np.linspace(0, total_frames - 1, num_frames, dtype=int)


def jitter_indices(total_frames, num_frames=NUM_FRAMES, rng=random):
    """Training-time sampling: one random frame from each of num_frames equal segments."""
    if total_frames <= num_frames:
        return uniform_indices(total_frames, num_frames)
    segment_len = total_frames / float(num_frames)
    indices = []
    for i in range(num_frames):
        start = int(i * segment_len)
        end = int((i + 1) * segment_len)
        indices.append(start if end <= start else rng.randint(start, end - 1))
    return np.array(indices, dtype=int)


def to_model_tensor(frames):
    """(T, H, W, 3) uint8 or float-in-[0,1] RGB frames -> normalised (T, 3, H, W) float tensor."""
    arr = np.asarray(frames)
    if arr.dtype == np.uint8:
        arr = arr.astype(np.float32) / 255.0
    tensor = torch.from_numpy(np.ascontiguousarray(arr, dtype=np.float32)).permute(0, 3, 1, 2)
    return (tensor - _MEAN_T) / _STD_T


def motion_energy(frames):
    """Mean absolute change between consecutive frames, on a 0-255 scale."""
    arr = np.asarray(frames)
    if len(arr) < 2:
        return 0.0
    return float(np.abs(np.diff(arr.astype(np.int16), axis=0)).mean())
