# 🧠 Model Training

This folder contains **everything needed to train, evaluate, and validate** the ISL Emergency Gesture Recognition model from scratch.

---

## What Model Was Used?

### Architecture: Spatio-Temporal Residual CNN + Bidirectional GRU + Temporal Attention

| Component | Details |
|-----------|---------|
| **Input** | 16 frames × 112×112 RGB, sampled uniformly from a ~2.5s video clip |
| **Spatial Extractor** | 4-stage Residual CNN (32→64→128→256 channels) with skip connections |
| **Temporal Encoder** | 2-layer Bidirectional GRU (hidden=128 each direction = 256 combined) |
| **Attention** | Bahdanau-style temporal attention over all 16 GRU outputs |
| **Classifier** | FC(256→128) → Dropout(0.4) → FC(128→num_classes) |
| **Total params** | ~2.3 million |

### Training Setup

| Setting | Value |
|---------|-------|
| **Dataset** | Mendeley ISL Emergency Gestures (CC BY 4.0) |
| **Classes** | accident, call, doctor, help, pain (+ optional `none`) |
| **Signers** | 26 participants, 2 takes each |
| **Split** | Signer-independent: test=022–026, val=4 random, train=rest |
| **Optimizer** | AdamW (lr=5e-4, weight_decay=1e-3) |
| **Scheduler** | Cosine Annealing (T_max=40, eta_min=1e-6) |
| **Loss** | CrossEntropyLoss with label smoothing=0.05 |
| **Augmentation** | Temporal jitter + brightness/contrast jitter + ±6px translation |
| **Checkpoint** | Best epoch by lowest validation loss |
| **Hardware** | CPU / CUDA (auto-detected) |

### Verified Results (Held-Out Test: Signers 022–026)

| Metric | Value |
|--------|-------|
| Accuracy | **97.50%** (78/80 clips correct) |
| Macro F1 | **0.9778** |
| Weighted F1 | **0.9755** |

---

## Files

| File | Purpose |
|------|---------|
| `model.py` | Neural network architecture (CNN + BiGRU + Attention) |
| `preprocess.py` | Frame normalization, ROI cropping, motion energy |
| `dataset.py` | PyTorch Dataset with signer-independent splitting |
| `engine.py` | Training loop, evaluation loop, predict_samples |
| `train.py` | CLI entry point to train the model |
| `evaluate.py` | Evaluate a checkpoint on the held-out test set |
| `cross_validate.py` | 5-fold cross-validation across signers |
| `data_prep.py` | Preprocess raw .avi videos into a frame cache |

---

## Usage

```bash
# 1. Build frame cache from raw videos
python -m src.data_prep --raw-dir data/raw --out data/processed/cached_all_frames.pt

# 2. Train
python -m src.train --epochs 40

# 3. Evaluate on held-out signers
python -m src.evaluate --checkpoint models/best_isl_model.pth

# 4. Cross-validate
python -m src.cross_validate
```

---

## Why Bidirectional GRU?

Sign language is **temporal** — the same hand shape means different things in different motion contexts. A BiGRU reads the gesture **forwards AND backwards**, so it understands not just where a hand is but where it came from and where it's going.

**Temporal Attention** then learns which of the 16 frames carry the most discriminative information (e.g., the peak of the wrist-twist in "pain" vs. the contact point of hands in "help").
