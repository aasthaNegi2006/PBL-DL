"""Model loading and prediction shared by evaluation, the webcam app and the API."""
import collections
from dataclasses import dataclass, field

import numpy as np
import torch

from src import preprocess as P
from src.dataset import CLASSES, NONE_CLASS
from src.model import ISLGestureRecognizer

# A sign is only reported when the top class reaches this probability.
CONFIDENCE_THRESHOLD = 0.75


@dataclass
class LoadedModel:
    model: torch.nn.Module
    classes: list
    num_frames: int
    window_seconds: float
    device: torch.device
    meta: dict = field(default_factory=dict)

    @property
    def has_none_class(self):
        return NONE_CLASS in self.classes


def build_model(checkpoint, device):
    classes = list(checkpoint.get('classes', CLASSES))
    model = ISLGestureRecognizer(
        num_classes=len(classes),
        num_frames=checkpoint.get('num_frames', P.NUM_FRAMES),
        in_channels=3,
        feature_dim=checkpoint.get('feature_dim', 256),
        hidden_dim=checkpoint.get('hidden_dim', 128),
    )
    model.load_state_dict(checkpoint['model_state_dict'])
    return model.to(device).eval()


def load_model(path, device=None):
    """Load a checkpoint saved by src.train. Uses weights_only=True, so a
    tampered .pth file cannot run code when it is loaded."""
    device = torch.device(device) if device else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    checkpoint = torch.load(path, map_location=device, weights_only=True)
    model = build_model(checkpoint, device)
    meta = {k: checkpoint[k] for k in ('epoch', 'val_acc', 'val_loss', 'split', 'selection_metric') if k in checkpoint}
    return LoadedModel(
        model=model,
        classes=list(checkpoint.get('classes', CLASSES)),
        num_frames=int(checkpoint.get('num_frames', P.NUM_FRAMES)),
        window_seconds=float(checkpoint.get('window_seconds', P.DEFAULT_WINDOW_SECONDS)),
        device=device,
        meta=meta,
    )


def sample_clip(frames, num_frames):
    """Pick num_frames evenly spaced frames from a list of prepared (uint8 RGB) frames."""
    frames = np.asarray(frames)
    return frames[P.uniform_indices(len(frames), num_frames)]


@torch.no_grad()
def predict_clip(lm, sampled_frames):
    """sampled_frames: (num_frames, H, W, 3) uint8 RGB. Returns (probs, attention_weights)."""
    x = P.to_model_tensor(sampled_frames).unsqueeze(0).to(lm.device)
    logits, attn = lm.model(x)
    probs = torch.softmax(logits, dim=1)[0].cpu().numpy()
    return probs, attn[0, :, 0].cpu().numpy()


class FrameWindow:
    """Rolling buffer holding the last `seconds` of prepared frames.

    Training clips cover a whole gesture (2-4 s) squeezed into 16 evenly spaced
    frames, so live input must be sampled the same way: keep a gesture-length
    window and pick 16 frames spread across it, rather than feeding the last 16
    consecutive frames (about 0.5 s at 30 FPS).
    """

    def __init__(self, seconds, max_frames=600):
        self.seconds = float(seconds)
        self._buf = collections.deque(maxlen=max_frames)

    def add(self, frame, t):
        self._buf.append((t, frame))
        while self._buf and t - self._buf[0][0] > self.seconds:
            self._buf.popleft()

    def span(self):
        return self._buf[-1][0] - self._buf[0][0] if len(self._buf) > 1 else 0.0

    def ready(self, num_frames):
        return len(self._buf) >= num_frames and self.span() >= 0.9 * self.seconds

    def frames(self):
        return [f for _, f in self._buf]

    def clear(self):
        self._buf.clear()

    def __len__(self):
        return len(self._buf)
