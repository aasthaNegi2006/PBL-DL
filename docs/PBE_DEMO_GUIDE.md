# 8-Minute Project Demonstration Guide

**Project:** Indian Sign Language (ISL) Emergency Gesture Recognition System  
**Presenter:** Aastha Negi (Solo Viva Defense)  
**Academic Context:** UPES Semester V Deep Learning — Project-Based Evaluation (PBL-DL)  
**Total Allotted Time:** 8 minutes demo + 4 minutes technical viva voce  

---

## 1. Demo Timing & Agenda Breakdown

```
[00:00 - 02:00] Problem Definition, ISL Emergency Vocabulary & Dataset Leakage Trap
[02:00 - 04:00] Spatio-Temporal Deep Learning Architecture & Mathematical Flow
[04:00 - 06:00] Working Demonstration: Real-Time Webcam & Video Pipeline
[06:00 - 08:00] Scientific Results, Confusion Matrix, Error Analysis & Real-World Domain Shift
[08:00 - 12:00] Technical Viva Voce Defense
```

---

## 2. Minute-by-Minute Speaking Script

### Minute 0:00 – 02:00: Problem & Scientific Integrity
- **What to say:**
  > "Good morning, respected evaluators. My project is an Indian Sign Language Emergency Gesture Recognition System designed to bridge the communication gap for Deaf and Hard-of-Hearing individuals during medical and distress situations.
  > The system recognizes 5 high-priority emergency gestures: **Accident, Call, Doctor, Help, and Pain**, alongside a 6th **None** rejection class.
  > A central contribution of my work is ensuring scientific validity: earlier models in this domain suffered from the '100% accuracy trap' caused by random video splitting, where the same signer's Take 1 was in train and Take 2 in test. This allowed networks to memorize clothing and background rather than gestures.
  > In this project, I enforced a strict **Signer-Independent protocol**: 5 signers (80 clips) were completely held out and never seen during training or tuning. Our true, honest test accuracy is **97.50%**, comfortably surpassing the UPES 85% target threshold."

### Minute 02:00 – 04:00: Architecture & Mathematical Justification
- **What to say:**
  > "Sign gestures are dynamic spatio-temporal events. A 2D-CNN alone cannot capture hand trajectory over time, while a 3D-CNN is too computationally expensive for real-time CPU laptops.
  > Therefore, I implemented a hybrid **4-stage Residual 2D-CNN + 2-layer Bidirectional GRU + Bahdanau Additive Attention**:
  > 1. **Spatial Encoder:** A 4-stage ResNet processes $16$ uniformly sampled $112 \times 112$ frames, extracting a 256-dimensional spatial embedding per frame.
  > 2. **Temporal BiGRU:** A 2-layer Bidirectional GRU (hidden size 128 per direction) processes forward and backward motion dynamics, yielding 256-dimensional temporal states.
  > 3. **Bahdanau Attention:** Instead of relying only on the final frame, attention computes scalar alignment energies $e_t = v_a^\top \tanh(W_a h_t + b_a)$ to softly weigh the most informative contact points—such as the two fists colliding in 'Accident' or the pulse tap in 'Doctor'.
  > 4. **Linear Head:** Maps the 256-dim context vector to 6 logits trained via Cross-Entropy Loss with Cosine Annealing learning rate decay."

### Minute 04:00 – 06:00: Live Working Demonstration
- **Action:** Open the dashboard at `http://127.0.0.1:8000` or run `run_web_dashboard.bat`.
- **What to say & show:**
  > "Here is our running dashboard. Notice that performance metrics and badges are dynamically verified against the model checkpoint SHA-256 digest to prevent unverified claims.
  > I will now demonstrate live inference:
  > 1. When resting, our **calibrated motion energy gate** ($\theta = 2.4$) classifies the video stream as **Waiting (idle)** without false alarms.
  > 2. Now I will perform the **Help** sign: flat left palm with right thumb-up fist brought firmly down. Notice the bounding box around the chest and hands. The system detects the sign, accumulates consecutive hits, and speaks: *'HELP'* via Text-to-Speech!
  > 3. If I make random movements like waving or adjusting glasses, the 6th **None** rejection class activates with over 96% precision, suppressing false alerts."
- **Alternative (if webcam permission is unavailable):**
  > Double-click `run_video_demo.bat` to demonstrate the exact same pipeline running on an unseen test signer clip.

### Minute 06:00 – 08:00: Experimental Results & Limitations
- **What to say:**
  > "Looking at our official evaluation report on 80 held-out clips from unseen signers 022–026:
  > - **Overall Accuracy:** 97.50% (78 / 80 correct)
  > - **Macro F1-Score:** 0.9778
  > - **Per-Class Breakdown:** `accident`, `doctor`, `help`, and `pain` all achieved 100% recall. `call` achieved 90% (1 clip confused with `none` due to subtle finger phone-curve motion).
  > - **Domain Shift Analysis:** The primary challenge when transitioning from studio clips to live webcams is lighting and background noise. We engineered motion gating and an ambiguity margin filter ($\ge 0.12$) to bridge this domain gap.
  > The system is ready for technical viva questions."

---

## 3. Demonstration Checklist

- [ ] Laptop charger plugged in; Windows display sleep disabled.
- [ ] Virtual environment verified (`.\.venv\Scripts\activate`).
- [ ] Model weights verified at `models/best_isl_model.pth`.
- [ ] Evaluation summary present at `eval_results/summary.json`.
- [ ] FastAPI dashboard tested at `http://127.0.0.1:8000`.
- [ ] Audio volume turned up for Text-to-Speech demonstration.
