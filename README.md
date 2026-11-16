# Indian Sign Language (ISL) Emergency Gesture Recognition System

An end-to-end Deep Learning system designed for real-time recognition of **Indian Sign Language (ISL)** emergency gestures from live webcam video streams, built using **PyTorch**, **OpenCV**, and **FastAPI**.

---

## 📌 1. Project Overview & Motivation
In emergency situations (accidents, acute pain, medical distress), Deaf and Hard-of-Hearing individuals face severe communication barriers when attempting to contact emergency services or communicate with first responders. 

This project bridges that gap by providing:
- **Instant Video Gesture Recognition**: Automatically classifies dynamic ISL emergency signs from video streams.
- **Auditory Feedback (Text-to-Speech)**: Synthesizes spoken voice alerts in real time so bystanders and healthcare professionals hear the request immediately.
- **High Generalization**: Evaluated on completely unseen individuals with **98.0% accuracy** and **100.0% validation accuracy**, surpassing the academic benchmark threshold ($\ge 85\%$).

---

## 🗂️ 2. Dataset & Vocabulary

The dataset comprises **260 cropped video clips** recorded at 30 FPS, covering 5 emergency and healthcare classes:

| Class | Meaning / Context | Total Clips | Subjects | Takes per Subject |
| :--- | :--- | :---: | :---: | :---: |
| **`accident`** | Reporting a collision or vehicle accident | 52 | 26 | 2 |
| **`call`** | Urgently requesting a phone call / helpline | 52 | 26 | 2 |
| **`doctor`** | Requesting immediate medical / physician attention | 52 | 26 | 2 |
| **`help`** | General distress / calling for assistance | 52 | 26 | 2 |
| **`pain`** | Reporting severe physical pain / injury | 52 | 26 | 2 |
| **TOTAL** | **5 Dynamic ISL Emergency Words** | **260** | **26** | **2** |

---

## 🧠 3. System Architecture

Because sign language words are **dynamic motion gestures** (movements unfolding over time), standard single-image classifiers fail to capture motion trajectories. This system uses a hierarchical **Spatio-Temporal Deep Learning Architecture**:

```
Input Video Stream (Webcam or File)
                │
                ▼
Uniform Sampling (16 Frames Sequence: 16 x 112 x 112 x 3)
                │
                ▼
Spatial Feature Extractor (4-Stage Residual ConvNet)
    - Conv Block 1: 32 filters (56x56 -> 28x28)
    - Conv Block 2: 64 filters (14x14)
    - Conv Block 3: 128 filters (7x7)
    - Conv Block 4: 256 filters (Adaptive AvgPool -> 256-D feature vector per frame)
                │
                ▼
Temporal Sequence Modeling (2-Layer Bidirectional GRU)
    - Hidden Size: 128 (Bidirectional -> 256 output dimensions)
    - Captures forward and reverse temporal dynamics of hand motion
                │
                ▼
Temporal Self-Attention Mechanism
    - Dynamically computes attention weights across the 16 timesteps
    - Focuses peak attention on the stroke/impact phase of the gesture
                │
                ▼
Dense Classifier Head (Dropout 0.4 -> Linear 128 -> ReLU -> Linear 5)
                │
                ▼
Predicted ISL Class + Softmax Probabilities + Real-Time Voice Synthesis
```

---

## 📊 4. Experimental Results

The model was trained and evaluated under two rigorous validation protocols:

### A. Stratified 80/20 Validation
- **Validation Accuracy**: **100.00%**
- **Macro Average F1-Score**: **1.0000**
- **Training Time**: ~8 minutes on CPU (40 epochs)

#### Classification Report:
```
              precision    recall  f1-score   support

    accident     1.0000    1.0000    1.0000        10
        call     1.0000    1.0000    1.0000        10
      doctor     1.0000    1.0000    1.0000        10
        help     1.0000    1.0000    1.0000        10
        pain     1.0000    1.0000    1.0000        10

    accuracy                         1.0000        50
   macro avg     1.0000    1.0000    1.0000        50
weighted avg     1.0000    1.0000    1.0000        50
```

### B. Subject-Independent Generalization (Zero-Shot Signers)
To prove the model generalizes to new people (and does not simply memorize clothing, lighting, or facial appearance), subjects `022`, `023`, `024`, `025`, and `026` were completely held out from training:
- **Test Accuracy on completely unseen subjects**: **98.00%** (49 / 50 correct).

Visual performance curves and the confusion matrix are stored in `eval_results/`:
- `eval_results/confusion_matrix.png`
- `eval_results/training_curves.png`
- `eval_results/classification_report.txt`

---

## 🚀 5. How to Run the Applications

### Prerequisites
Make sure your virtual environment is active:
```powershell
.\.venv\Scripts\Activate.ps1
```

### Option A: Run Live Webcam Application (OpenCV HUD + Audio Voice)
Launch the real-time desktop window with webcam feed, gesture HUD overlay, confidence meters, and voice synthesis:
```powershell
.\.venv\Scripts\python.exe -m src.real_time_webcam --source 0
```
- **Hotkeys in window**:
  - `Q` or `ESC`: Quit application
  - `S`: Toggle voice speech feedback (ON / OFF)
  - `C`: Clear recent gesture history

*(Tip: You can also test with a saved video file instead of a live camera: `python -m src.real_time_webcam --source data/raw/help/help_Crop_001_01.avi`)*

---

### Option B: Run FastAPI Web Dashboard
Start the web server to access the browser-based dashboard and REST API:
```powershell
.\.venv\Scripts\uvicorn.exe app.main:app --reload --host 127.0.0.1 --port 8000
```
Then open your browser at:
`http://127.0.0.1:8000`

- **Interactive features**:
  - Drag-and-drop video gesture tester
  - Live animated confidence bars for all 5 classes
  - REST endpoint: `POST /api/predict`
  - System metadata endpoint: `GET /api/info`

---

## 📁 6. Project Directory Structure

```
SignLanguageProject/
├── app/
│   ├── main.py                  # FastAPI REST server & prediction router
│   └── templates/
│       └── index.html           # Interactive HTML5/JS dashboard
├── data/
│   ├── raw/                     # 260 extracted ISL videos organized by class
│   │   ├── accident/
│   │   ├── call/
│   │   ├── doctor/
│   │   ├── help/
│   │   └── pain/
│   └── processed/
│       └── cached_all_frames.pt # Preprocessed frames cache for fast loading
├── eval_results/
│   ├── classification_report.txt# Detailed precision/recall/F1 metrics
│   ├── confusion_matrix.png     # Visual confusion matrix plot
│   ├── history.json             # Epoch-by-epoch loss & accuracy log
│   ├── summary.json             # Benchmark summary metadata
│   └── training_curves.png      # Loss and accuracy vs epoch plot
├── models/
│   └── best_isl_model.pth       # Trained PyTorch model weights & hyperparameters
├── src/
│   ├── __init__.py
│   ├── dataset.py               # Custom PyTorch Dataset with temporal sampling & spatial jitter
│   ├── evaluate.py              # Evaluation script generating plots and reports
│   ├── model.py                 # PyTorch Spatio-Temporal Residual CNN + BiGRU + Attention
│   ├── real_time_webcam.py      # Real-time OpenCV runner with HUD overlay & speech
│   ├── train.py                 # End-to-end training loop with Cosine Annealing LR
│   └── tts_helper.py            # Non-blocking Windows SAPI Text-to-Speech engine
├── requirements.txt
└── README.md
```

---

## 🎓 7. UPES Internals / Viva Voce Preparation Guide

### Frequently Asked Questions & Model Answers:

**Q1: Why did you choose a Spatio-Temporal model (CNN + BiGRU) instead of a simple 2D CNN?**
> **Answer**: ISL emergency words like *accident*, *call*, *doctor*, *help*, and *pain* are dynamic movements that take 2 to 4 seconds to perform. A 2D CNN only classifies a single still frame, ignoring the direction, speed, and trajectory of hand movement. Our architecture uses a 2D Residual CNN to extract spatial hand/body features from 16 sampled frames, and a 2-layer Bidirectional GRU with Temporal Attention to model the chronological sequence of the gesture.

**Q2: How did you ensure your model does not overfit on 260 video samples?**
> **Answer**: We implemented four regularization and generalization techniques:
> 1. **Temporal Jitter**: Random temporal segment sampling during training so the network sees varied phase alignments.
> 2. **Spatial Augmentation**: Random small translations, brightness jitter, and contrast scaling.
> 3. **Regularization**: Dropout (0.3 in GRU, 0.4 and 0.3 in classifier head), Weight Decay ($10^{-3}$ in AdamW), and Label Smoothing ($0.05$).
> 4. **Subject-Independent Cross-Validation**: Testing on unseen individuals, achieving 98.0% accuracy.

**Q3: Why use GRU instead of LSTM or 3D CNN?**
> **Answer**: GRU has fewer parameters than standard LSTM (3 gates vs 4), training faster and reducing overfitting on medium-sized sequence datasets. Furthermore, full 3D CNNs (e.g. C3D, SlowFast) have massive computational overhead and struggle to run smoothly in real time on CPU. MobileNet/Residual CNN + BiGRU provides real-time 30 FPS inference on CPU without needing a dedicated GPU.

**Q4: How does the real-time webcam inference pipeline work?**
> **Answer**: The webcam captures live frames into a sliding queue (FIFO buffer) of 16 frames. Once 16 frames are accumulated, the tensor is preprocessed (resized to $112 \times 112$ and standardized) and passed through the neural network every 2-3 frames. An Exponential Moving Average filter smooths probability spikes, and when a gesture maintains $\ge 85\%$ confidence for consecutive frames, the HUD turns green and the voice synthesizer speaks the word.
#   P B L - D L  
 