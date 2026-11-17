# Indian Sign Language (ISL) Emergency Gesture Recognition System

[![Tests](https://github.com/aasthaNegi2006/PBL-DL/actions/workflows/tests.yml/badge.svg)](https://github.com/aasthaNegi2006/PBL-DL/actions/workflows/tests.yml)

An end-to-end Deep Learning system designed for real-time recognition of **Indian Sign Language (ISL)** emergency gestures from live webcam video streams, built using **PyTorch**, **OpenCV**, and **FastAPI**.

---

## 📌 1. Project Overview & Motivation
In emergency situations (accidents, acute pain, medical distress), Deaf and Hard-of-Hearing individuals face severe communication barriers when attempting to contact emergency services or communicate with first responders.

This project bridges that gap by providing:
- **Instant Video Gesture Recognition**: Automatically classifies dynamic ISL emergency signs from video streams.
- **Auditory Feedback (Text-to-Speech)**: Speaks the recognised word so bystanders and healthcare professionals hear the request immediately.
- **Signer-Independent Evaluation**: Accuracy is measured only on people the model never saw during training (see [Section 4](#-4-evaluation-protocol--results)).

---

## 🗂️ 2. Dataset & Vocabulary

This project uses the public dataset:

> Adithya V. and Rajesh R., **"A Video Dataset of the Hand Gestures of Indian Sign Language Words used in Emergency Situations"**, Mendeley Data, V1. <https://data.mendeley.com/datasets/2vfdm42337/1> — licensed **CC BY 4.0**.
> Paper: *Hand gestures for emergency situations: A video dataset based on words from Indian sign language*, Data in Brief, 2020. <https://pmc.ncbi.nlm.nih.gov/articles/PMC7378574/>

We use the **cropped** videos of 5 of the dataset's 8 words, recorded at 30 FPS by 26 signers (2 takes each):

| Class | Meaning / Context | Total Clips | Subjects | Takes per Subject |
| :--- | :--- | :---: | :---: | :---: |
| **`accident`** | Reporting a collision or vehicle accident | 52 | 26 | 2 |
| **`call`** | Urgently requesting a phone call / helpline | 52 | 26 | 2 |
| **`doctor`** | Requesting immediate medical / physician attention | 52 | 26 | 2 |
| **`help`** | General distress / calling for assistance | 52 | 26 | 2 |
| **`pain`** | Reporting severe physical pain / injury | 52 | 26 | 2 |
| **TOTAL** | **5 Dynamic ISL Emergency Words** | **260** | **26** | **2** |

The other three words (`hot`, `lose`, `thief`) can be used as a **`none`** class so the model learns to say "this movement is not one of the emergency signs" instead of always picking one of the five (see `--none-from` below).

---

## 🧠 3. System Architecture

Because sign language words are **dynamic motion gestures** (movements unfolding over time), standard single-image classifiers fail to capture motion trajectories. This system uses a hierarchical **Spatio-Temporal Deep Learning Architecture**:

```
Input Video Stream (Webcam or File)
                │
                ▼
Gesture window (one whole sign, ~2-3 s) → 16 evenly spaced frames → 16 x 112 x 112 x 3
                │
                ▼
Spatial Feature Extractor (Residual ConvNet, trained from scratch)
    - Stem: 5x5 conv, 32 filters + max-pool (112 → 28)
    - Residual stage 1: 64 filters (14x14)
    - Residual stage 2: 128 filters (7x7)
    - Residual stage 3: 256 filters (4x4) → Adaptive AvgPool → 256-D feature vector per frame
                │
                ▼
Temporal Sequence Modeling (2-Layer Bidirectional GRU)
    - Hidden Size: 128 (Bidirectional -> 256 output dimensions)
                │
                ▼
Temporal Self-Attention (weights the 16 timesteps)
                │
                ▼
Dense Classifier Head (Dropout 0.4 -> Linear 128 -> ReLU -> Dropout 0.3 -> Linear N)
                │
                ▼
Predicted ISL Class + Softmax Probabilities + Voice Output
```

All code paths (data preparation, training, evaluation, desktop webcam, web API) share one preprocessing module, `src/preprocess.py`, so the model sees the same kind of input everywhere.

---

## 📊 4. Evaluation Protocol & Results

### Why the earlier 100% was not a real result
The first version split the 260 clips at random within each class. Every signer recorded each word **twice**, so for most validation clips the same person's other take was in the training set. The model was also selected and scored on the same 50 validation clips. Both inflate accuracy, so the earlier "100% validation / 98% unseen subjects" numbers have been withdrawn.

### Current protocol (signer-independent)
- **Held-out test signers**: `src.train` holds out signers `022`–`026` by default. They are never used for training or for choosing the best epoch. `src.evaluate` scores the model on them.
- **Validation signers**: a few other signers are used only to choose the best epoch (lowest validation loss).
- **Cross-validation**: `src.cross_validate` runs 5-fold `GroupKFold` by signer, so every clip is scored once by a model that never saw that person. Report the **mean ± std** from this.

### Results
| Protocol | Accuracy | Macro F1 |
| :--- | :---: | :---: |
| 5-fold signer-independent CV (`eval_results/cross_validation.json`) | _run `python -m src.cross_validate`_ | |
| Held-out signers 022–026 (`eval_results/summary.json`) | _run `python -m src.evaluate`_ | |

For reference, the dataset paper reports 90–100% per-class accuracy for its deep-learning baseline on all 8 words.

### To do: produce the results
The code has only been tested end to end on synthetic videos, so the table above is still empty. With the dataset downloaded:

- [ ] Download the dataset and build the frame cache ([Step 1](#step-1-download-the-dataset-and-build-the-frame-cache))
- [ ] `python -m src.train` then `python -m src.evaluate` → copy `accuracy_percent` and `macro_avg_f1` from `eval_results/summary.json` into the table
- [ ] `python -m src.cross_validate` (about 40 minutes on a CPU) → copy `accuracy_mean ± accuracy_std` and `macro_f1_mean ± macro_f1_std` from `eval_results/cross_validation.json`
- [ ] Try the live demo (`python -m src.real_time_webcam --source 0`). If real signs are reported as idle, lower `--motion-threshold` (default 2.5)
- [ ] Commit the new `eval_results/` files so the dashboard shows the real numbers, and add the trained model to `models/` (about 8 MB)

---

## 🚀 5. How to Run

### Setup
```bash
python -m venv .venv
# Windows: .\.venv\Scripts\Activate.ps1      macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
```

### Step 1: Download the dataset and build the frame cache
The videos are not stored in this repository.

1. Open <https://data.mendeley.com/datasets/2vfdm42337/1> and click **Download All** (about 3.14 GB).
2. Unzip it and copy the **cropped** video folders into `data/raw/`. The folder layout does not matter: files are found by name (`<word>_crop_<signer>_<take>`, any capitalisation, e.g. `help_Crop_001_01.avi`). The uncropped "Raw" videos are not needed.
   - Needed: `accident`, `call`, `doctor`, `help`, `pain`
   - Also copy `hot`, `lose`, `thief` if you want the "none" class (recommended)
3. Build the frame cache:
```bash
python -m src.data_prep --none-from hot lose thief
# or, without the "none" class:
python -m src.data_prep
```
It prints the number of clips per class (52 each; 50 for `lose` and `thief`) and the clip lengths. If it finds no videos, check the file names.

### Step 2: Train and evaluate
```bash
python -m src.train              # holds out signers 022-026, saves models/best_isl_model.pth
python -m src.evaluate           # scores the held-out signers → eval_results/
python -m src.cross_validate     # 5-fold signer-independent CV → eval_results/cross_validation.json
```

### Option A: Live Webcam Application (OpenCV HUD + Voice)
```bash
python -m src.real_time_webcam --source 0
```
- Sign inside the yellow box. The model looks at the last ~2-3 seconds (one whole sign), so finish the full movement.
- **Hotkeys**: `Q`/`ESC` quit · `S` toggle speech · `C` clear history
- Test with a dataset clip (already cropped, so no box is applied): `python -m src.real_time_webcam --source data/raw/help/help_Crop_001_01.avi`
- Speech uses the system voice: Windows SAPI, macOS `say`, Linux `espeak-ng`.

### Option B: FastAPI Web Dashboard
```bash
uvicorn app.main:app --reload --host 127.0.0.1 --port 8000
```
Then open `http://127.0.0.1:8000`.

- Browser webcam with the same gesture window and crop as the desktop app
- Drag-and-drop video tester
- REST endpoints: `POST /api/predict` (one clip), `POST /api/predict-frames` (live frames), `GET /api/info`

### Tests
```bash
pip install -r requirements-dev.txt
pytest
```
GitHub Actions runs the same tests on Python 3.10 and 3.13 for every pull request and every push to `main` (`.github/workflows/tests.yml`). The tests use synthetic videos, so they do not need the dataset or a trained model.

---

## 📁 6. Project Directory Structure

```
PBL-DL/
├── .github/workflows/
│   └── tests.yml                # CI: runs pytest on every PR
├── app/
│   ├── main.py                  # FastAPI REST server & prediction routes
│   └── templates/
│       └── index.html           # Interactive HTML5/JS dashboard
├── data/                        # not committed
│   ├── raw/                     # dataset videos
│   └── processed/
│       └── cached_all_frames.pt # frame cache built by src.data_prep
├── eval_results/                # written by src.evaluate and src.cross_validate
├── models/
│   └── best_isl_model.pth       # trained weights + config + signer split
├── src/
│   ├── preprocess.py            # shared frame preparation, sampling, normalisation, crop box
│   ├── data_prep.py             # builds the frame cache from raw videos
│   ├── dataset.py               # PyTorch Dataset, signer-based splitting
│   ├── model.py                 # Spatio-Temporal Residual CNN + BiGRU + Attention
│   ├── engine.py                # training / prediction loops
│   ├── train.py                 # training with held-out test signers
│   ├── evaluate.py              # held-out signer evaluation, plots and reports
│   ├── cross_validate.py        # k-fold signer-independent cross-validation
│   ├── inference.py             # model loading, gesture window for live input
│   ├── real_time_webcam.py      # OpenCV desktop app with HUD & speech
│   └── tts_helper.py            # cross-platform non-blocking text-to-speech
├── tests/
├── requirements.txt
└── README.md
```

---

