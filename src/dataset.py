import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import random
import os

CLASSES = ['accident', 'call', 'doctor', 'help', 'pain']
CLASS_TO_IDX = {c: i for i, c in enumerate(CLASSES)}

class ISLVideoDataset(Dataset):
    def __init__(self, samples, num_frames=16, is_train=True):
        """
        samples: list of sample dicts from cached_all_frames.pt
        num_frames: number of frames to sample per sequence (e.g. 16)
        is_train: apply augmentations if True
        """
        self.samples = samples
        self.num_frames = num_frames
        self.is_train = is_train

    def __len__(self):
        return len(self.samples)

    def _sample_frame_indices(self, total_frames):
        if total_frames <= self.num_frames:
            return np.linspace(0, total_frames - 1, self.num_frames, dtype=int)
        
        if self.is_train:
            # Random temporal jitter: divide into num_frames segments and pick one frame randomly in each
            segment_len = total_frames / float(self.num_frames)
            indices = []
            for i in range(self.num_frames):
                start = int(i * segment_len)
                end = int((i + 1) * segment_len)
                if end <= start:
                    idx = start
                else:
                    idx = random.randint(start, end - 1)
                indices.append(idx)
            return np.array(indices, dtype=int)
        else:
            # Deterministic uniform sampling
            return np.linspace(0, total_frames - 1, self.num_frames, dtype=int)

    def _apply_spatial_augment(self, frames):
        # frames: (T, H, W, C) float32 in [0, 1]
        T, H, W, C = frames.shape
        
        # 1. Random brightness jitter
        brightness_factor = random.uniform(0.85, 1.15)
        frames = frames * brightness_factor
        
        # 2. Random contrast jitter
        contrast_factor = random.uniform(0.85, 1.15)
        mean = np.mean(frames)
        frames = (frames - mean) * contrast_factor + mean
        
        # 3. Random small translation / crop (shift by up to 6 pixels)
        dx = random.randint(-6, 6)
        dy = random.randint(-6, 6)
        
        shifted = np.zeros_like(frames)
        src_x_start = max(0, -dx)
        src_x_end = min(W, W - dx)
        dst_x_start = max(0, dx)
        dst_x_end = min(W, W + dx)
        
        src_y_start = max(0, -dy)
        src_y_end = min(H, H - dy)
        dst_y_start = max(0, dy)
        dst_y_end = min(H, H + dy)
        
        shifted[:, dst_y_start:dst_y_end, dst_x_start:dst_x_end, :] = \
            frames[:, src_y_start:src_y_end, src_x_start:src_x_end, :]
            
        return np.clip(shifted, 0.0, 1.0)

    def __getitem__(self, idx):
        item = self.samples[idx]
        raw_frames = item['frames'] # uint8 array (N, 112, 112, 3)
        total_frames = item['total_frames']
        label = item['label']
        
        # 1. Sample frame indices
        indices = self._sample_frame_indices(total_frames)
        sampled = raw_frames[indices].astype(np.float32) / 255.0 # (T, 112, 112, 3)
        
        # 2. Augment if train
        if self.is_train:
            sampled = self._apply_spatial_augment(sampled)
            
        # 3. Transpose to (T, C, H, W)
        tensor_frames = torch.from_numpy(sampled).permute(0, 3, 1, 2).float()
        
        # Standardize using ImageNet statistics
        mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        tensor_frames = (tensor_frames - mean) / std
        
        return tensor_frames, label, item['video_name']


def create_splits(cached_path='data/processed/cached_all_frames.pt', val_ratio=0.2, seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    data = torch.load(cached_path, weights_only=False)
    samples = data['samples']
    
    by_class = {c: [] for c in CLASSES}
    for s in samples:
        by_class[s['class_name']].append(s)
        
    train_samples = []
    val_samples = []
    
    for cls, items in by_class.items():
        random.shuffle(items)
        n_val = int(len(items) * val_ratio) # 10 val per class out of 52 = 50 val, 210 train
        val_samples.extend(items[:n_val])
        train_samples.extend(items[n_val:])
        
    print(f'Total samples: {len(samples)} | Train: {len(train_samples)} | Val: {len(val_samples)}')
    return train_samples, val_samples
