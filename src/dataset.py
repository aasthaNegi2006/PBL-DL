import os
import pickle
import random
import re

import numpy as np
import torch
from torch.utils.data import Dataset

from src import preprocess as P

CLASSES = ['accident', 'call', 'doctor', 'help', 'pain']
NONE_CLASS = 'none'

# Dataset filenames end in <subject>_<take>, e.g. help_Crop_001_01.avi -> subject 001, take 01.
_SUBJECT_TAKE_RE = re.compile(r'(\d{3})_(\d{2})$')

DEFAULT_CACHE = os.path.join('data', 'processed', 'cached_all_frames.pt')


def parse_subject_take(video_name):
    stem = os.path.splitext(os.path.basename(video_name))[0]
    match = _SUBJECT_TAKE_RE.search(stem)
    if not match:
        raise ValueError(f'Cannot read subject/take from video name "{video_name}" '
                         '(expected it to end in _<3-digit subject>_<2-digit take>)')
    return match.group(1), match.group(2)


def subject_of(sample):
    return sample.get('subject') or parse_subject_take(sample['video_name'])[0]


def load_cache(path=DEFAULT_CACHE):
    """Load the frame cache built by src.data_prep. Returns (samples, classes, meta)."""
    if not os.path.exists(path):
        raise FileNotFoundError(f'Frame cache not found at {path}. Build it with: python -m src.data_prep')
    try:
        data = torch.load(path, weights_only=True)
    except pickle.UnpicklingError:
        # Caches made by the original project stored numpy arrays, which need the
        # full unpickler. Only do this for a cache you built yourself.
        print(f'Note: {path} uses the old cache format. Rebuild it with python -m src.data_prep.')
        data = torch.load(path, weights_only=False)
    samples = data['samples']
    classes = list(data.get('classes', CLASSES))
    for s in samples:
        s.setdefault('subject', subject_of(s))
        s.setdefault('fps', P.DEFAULT_FPS)
    return samples, classes, data.get('meta', {})


def clip_seconds(samples):
    """Median clip length in seconds; the live window uses this so it matches training."""
    durations = [s['total_frames'] / float(s.get('fps') or P.DEFAULT_FPS) for s in samples]
    return float(np.median(durations)) if durations else P.DEFAULT_WINDOW_SECONDS


def all_subjects(samples):
    return sorted({subject_of(s) for s in samples})


def choose_val_subjects(candidates, n_val, seed):
    rng = random.Random(seed)
    candidates = sorted(candidates)
    return sorted(rng.sample(candidates, n_val))


def split_by_subject(samples, test_subjects, val_subjects):
    """Split samples so that each signer appears in exactly one of train / val / test.

    Every signer recorded each word twice. Splitting clips at random (as the
    original code did) puts a signer's second take in training while the first
    is used for validation, which inflates accuracy.
    """
    test_subjects, val_subjects = set(test_subjects), set(val_subjects)
    overlap = test_subjects & val_subjects
    if overlap:
        raise ValueError(f'Subjects in both val and test: {sorted(overlap)}')
    known = set(all_subjects(samples))
    missing = (test_subjects | val_subjects) - known
    if missing:
        raise ValueError(f'Unknown subjects {sorted(missing)}; available: {sorted(known)}')

    train, val, test = [], [], []
    for s in samples:
        subj = subject_of(s)
        if subj in test_subjects:
            test.append(s)
        elif subj in val_subjects:
            val.append(s)
        else:
            train.append(s)
    return train, val, test


def samples_for_subjects(samples, subjects):
    subjects = set(subjects)
    return [s for s in samples if subject_of(s) in subjects]


class ISLVideoDataset(Dataset):
    def __init__(self, samples, num_frames=P.NUM_FRAMES, is_train=True):
        """
        samples: list of sample dicts from the frame cache
        num_frames: number of frames to sample per sequence
        is_train: random temporal jitter and spatial augmentation if True
        """
        self.samples = samples
        self.num_frames = num_frames
        self.is_train = is_train

    def __len__(self):
        return len(self.samples)

    def _sample_frame_indices(self, total_frames):
        if self.is_train:
            return P.jitter_indices(total_frames, self.num_frames)
        return P.uniform_indices(total_frames, self.num_frames)

    def _apply_spatial_augment(self, frames):
        # frames: (T, H, W, C) float32 in [0, 1]
        T, H, W, C = frames.shape

        # 1. Random brightness jitter
        frames = frames * random.uniform(0.85, 1.15)

        # 2. Random contrast jitter
        contrast_factor = random.uniform(0.85, 1.15)
        mean = np.mean(frames)
        frames = (frames - mean) * contrast_factor + mean

        # 3. Random small translation (shift by up to 6 pixels)
        dx = random.randint(-6, 6)
        dy = random.randint(-6, 6)
        shifted = np.zeros_like(frames)
        src_x_start, src_x_end = max(0, -dx), min(W, W - dx)
        dst_x_start, dst_x_end = max(0, dx), min(W, W + dx)
        src_y_start, src_y_end = max(0, -dy), min(H, H - dy)
        dst_y_start, dst_y_end = max(0, dy), min(H, H + dy)
        shifted[:, dst_y_start:dst_y_end, dst_x_start:dst_x_end, :] = \
            frames[:, src_y_start:src_y_end, src_x_start:src_x_end, :]

        return np.clip(shifted, 0.0, 1.0)

    def __getitem__(self, idx):
        item = self.samples[idx]
        raw_frames = item['frames']  # uint8 (N, 112, 112, 3), numpy array or torch tensor
        if torch.is_tensor(raw_frames):
            raw_frames = raw_frames.numpy()

        indices = self._sample_frame_indices(item['total_frames'])
        sampled = raw_frames[indices]

        if self.is_train:
            sampled = self._apply_spatial_augment(sampled.astype(np.float32) / 255.0)

        return P.to_model_tensor(sampled), item['label'], item['video_name']
