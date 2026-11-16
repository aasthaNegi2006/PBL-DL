"""Training and prediction loops shared by train.py, evaluate.py and cross_validate.py."""
import copy
import random
import time

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src import preprocess as P
from src.dataset import ISLVideoDataset
from src.model import ISLGestureRecognizer

DEFAULT_HPARAMS = {
    'epochs': 40,
    'batch_size': 8,
    'lr': 5e-4,
    'weight_decay': 1e-3,
    'label_smoothing': 0.05,
    'num_frames': P.NUM_FRAMES,
    'feature_dim': 256,
    'hidden_dim': 128,
}


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device():
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def make_model(num_classes, hp):
    return ISLGestureRecognizer(
        num_classes=num_classes,
        num_frames=hp['num_frames'],
        in_channels=3,
        feature_dim=hp['feature_dim'],
        hidden_dim=hp['hidden_dim'],
    )


def train_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    running_loss, correct, total = 0.0, 0, 0
    for frames, labels, _ in dataloader:
        frames, labels = frames.to(device), labels.to(device)
        optimizer.zero_grad()
        logits, _ = model(frames)
        loss = criterion(logits, labels)
        loss.backward()
        # Gradient clipping to stabilise the recurrent layers
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
        optimizer.step()

        running_loss += loss.item() * frames.size(0)
        correct += (logits.argmax(dim=1) == labels).sum().item()
        total += labels.size(0)
    return running_loss / total, 100.0 * correct / total


@torch.no_grad()
def evaluate_loader(model, dataloader, criterion, device):
    model.eval()
    running_loss, correct, total = 0.0, 0, 0
    for frames, labels, _ in dataloader:
        frames, labels = frames.to(device), labels.to(device)
        logits, _ = model(frames)
        running_loss += criterion(logits, labels).item() * frames.size(0)
        correct += (logits.argmax(dim=1) == labels).sum().item()
        total += labels.size(0)
    if total == 0:
        return 0.0, 0.0
    return running_loss / total, 100.0 * correct / total


@torch.no_grad()
def predict_samples(model, samples, device, num_frames=P.NUM_FRAMES, batch_size=8):
    """Returns (preds, targets, probs, video_names) for a list of cached samples."""
    model.eval()
    loader = DataLoader(ISLVideoDataset(samples, num_frames=num_frames, is_train=False),
                        batch_size=batch_size, shuffle=False)
    preds, targets, probs, names = [], [], [], []
    for frames, labels, batch_names in loader:
        logits, _ = model(frames.to(device))
        p = torch.softmax(logits, dim=1).cpu().numpy()
        probs.extend(p)
        preds.extend(p.argmax(axis=1))
        targets.extend(labels.numpy())
        names.extend(batch_names)
    return np.array(preds), np.array(targets), np.array(probs), list(names)


def train_model(train_samples, val_samples, num_classes, hp, device, seed=42, log=print):
    """Train on train_samples and keep the epoch with the lowest validation loss.

    The original code kept the first epoch to reach the best validation
    accuracy (`>`), which froze the checkpoint at epoch 13 once accuracy hit
    100% even though validation loss kept improving until epoch 40.
    Validation loss is a smoother signal on a 40-50 clip validation set.

    Returns (best_state_dict, history, best_info).
    """
    set_seed(seed)
    train_loader = DataLoader(ISLVideoDataset(train_samples, hp['num_frames'], is_train=True),
                              batch_size=hp['batch_size'], shuffle=True, num_workers=0)
    val_loader = DataLoader(ISLVideoDataset(val_samples, hp['num_frames'], is_train=False),
                            batch_size=hp['batch_size'], shuffle=False, num_workers=0)

    model = make_model(num_classes, hp).to(device)
    log(f'Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}')

    criterion = nn.CrossEntropyLoss(label_smoothing=hp['label_smoothing'])
    optimizer = torch.optim.AdamW(model.parameters(), lr=hp['lr'], weight_decay=hp['weight_decay'])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=hp['epochs'], eta_min=1e-6)

    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
    best = {'val_loss': float('inf'), 'val_acc': 0.0, 'epoch': 0}
    best_state = None

    for epoch in range(1, hp['epochs'] + 1):
        t0 = time.time()
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = evaluate_loader(model, val_loader, criterion, device)
        scheduler.step()

        for key, value in zip(history, (train_loss, train_acc, val_loss, val_acc)):
            history[key].append(value)

        marker = ''
        if val_loss < best['val_loss']:
            best = {'val_loss': val_loss, 'val_acc': val_acc, 'epoch': epoch}
            best_state = copy.deepcopy(model.state_dict())
            marker = '  <- best'
        log(f'Epoch [{epoch:02d}/{hp["epochs"]:02d}] ({time.time() - t0:.1f}s) '
            f'Train Loss {train_loss:.4f} Acc {train_acc:.1f}% | '
            f'Val Loss {val_loss:.4f} Acc {val_acc:.1f}%{marker}')

    return best_state, history, best
