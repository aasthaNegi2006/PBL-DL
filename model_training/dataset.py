import os
import random
import re

import numpy as np
import torch
from torch.utils.data import Dataset

from model_training import preprocess as P

CLASSES = ['accident', 'call', 'doctor', 'help', 'pain']
NONE_CLASS = 'none'

# Dataset filenames end in <subject>_<take>, e.g. help_Crop_001_01.avi -> subject 001, take 01.
_SUBJECT_TAKE_RE = re.compile(r'(\d{3})_(\d{2})$')
_SUBJECT_RE = re.compile(r'^\d{3}$')

DEFAULT_CACHE = os.path.join('data', 'processed', 'cached_all_frames.pt')


def parse_subject_take(video_name):
    stem = os.path.splitext(os.path.basename(video_name))[0]
    match = _SUBJECT_TAKE_RE.search(stem)
    if not match:
        raise ValueError(f'Cannot read subject/take from video name "{video_name}" '
                         '(expected it to end in _<3-digit subject>_<2-digit take>)')
    subject, take = match.group(1), match.group(2)
    if not _SUBJECT_RE.match(subject):
        raise ValueError(f'Invalid subject ID "{subject}" parsed from "{video_name}" (expected 3 digits)')
    return subject, take


def subject_of(sample):
    subj = sample.get('subject') or parse_subject_take(sample['video_name'])[0]
    if not _SUBJECT_RE.match(subj):
        raise ValueError(f'Invalid subject ID "{subj}" in sample (expected 3 digits)')
    return subj


def load_cache(path=DEFAULT_CACHE):
    """Load the frame cache built by model_training.data_prep. Returns (samples, classes, meta).
    Always enforces safe tensor loading (weights_only=True) to eliminate arbitrary code execution.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f'Frame cache not found at {path}. Build it with: python -m model_training.data_prep')
    try:
        data = torch.load(path, weights_only=True)
    except Exception as e:
        raise RuntimeError(f'Failed to load cache safely from {path}: {e}. Rebuild it with: python -m model_training.data_prep')

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

    Enforces strict mathematical disjointness:
        train_signers ∩ val_signers = ∅
        train_signers ∩ test_signers = ∅
        val_signers ∩ test_signers = ∅

    Every signer recorded each word twice. Splitting clips at random puts a signer's
    second take in training while the first is in test, inflating accuracy.
    """
    test_set = set(test_subjects)
    val_set = set(val_subjects)
    known_set = set(all_subjects(samples))

    for s in test_set | val_set:
        if not _SUBJECT_RE.match(s):
            raise ValueError(f'Invalid subject ID format: {s!r} (expected 3 digits)')

    overlap = test_set & val_set
    if overlap:
        raise ValueError(f'Subjects in both val and test: {sorted(overlap)}')

    missing = (test_set | val_set) - known_set
    if missing:
        raise ValueError(f'Unknown subjects {sorted(missing)}; available: {sorted(known_set)}')

    train, val, test = [], [], []
    for s in samples:
        subj = subject_of(s)
        if subj in test_set:
            test.append(s)
        elif subj in val_set:
            val.append(s)
        else:
            train.append(s)

    train_set = {subject_of(s) for s in train}

    # Strict disjointness assertions
    assert not (train_set & val_set), f'Leakage detected: train & val share subjects {train_set & val_set}'
    assert not (train_set & test_set), f'Leakage detected: train & test share subjects {train_set & test_set}'
    assert not (val_set & test_set), f'Leakage detected: val & test share subjects {val_set & test_set}'

    # Disallow duplicate-recording leakage across splits
    train_recs = {s.get('recording_id') for s in train if s.get('recording_id')}
    val_recs = {s.get('recording_id') for s in val if s.get('recording_id')}
    test_recs = {s.get('recording_id') for s in test if s.get('recording_id')}
    assert not (train_recs & val_recs), f'Duplicate recording leakage: train & val share {train_recs & val_recs}'
    assert not (train_recs & test_recs), f'Duplicate recording leakage: train & test share {train_recs & test_recs}'
    assert not (val_recs & test_recs), f'Duplicate recording leakage: val & test share {val_recs & test_recs}'

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
