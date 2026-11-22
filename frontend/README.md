# 🖥️ Frontend

This folder contains the **HTML5/CSS/JavaScript web dashboard** for the ISL Emergency Gesture Recognition system.

---

## File

| File | Purpose |
|------|---------|
| `index.html` | Full single-page application — webcam feed, video upload, live prediction, evaluation report |

The HTML is **served by the FastAPI backend** at `http://127.0.0.1:8000`. Open `run_web_dashboard.bat` to start the server and launch the browser automatically.

---

## Features

| Feature | How it works |
|---------|--------------|
| **Live Webcam** | Captures frames at ~15fps, sends base64 batches to `/api/predict-frames` |
| **Video Upload** | Drag-and-drop or file picker; sends to `/api/predict` |
| **ROI Overlay** | Cyan dashed box shows exactly what the model sees (matches server-side crop) |
| **Confidence Bars** | Per-class softmax probabilities shown in real time |
| **Text-to-Speech** | Web Speech API speaks the confirmed gesture aloud in the browser |
| **Evaluation Report** | Loads from `/api/info` — accuracy, F1, per-class precision/recall, confusion matrix |
| **Provenance Verification** | SHA-256 of the loaded model must match the evaluation record; mismatch shows a warning badge |

---

## Gesture Guide (Updated)

| Gesture | Correct Technique |
|---------|------------------|
| **Accident** | Both fists apart at chest → bring inward to collide at center |
| **Call** | Extend thumb & pinky (phone shape) → raise right hand **fully to ear/cheek** (keep whole arm in box) |
| **Doctor** | Hold left wrist horizontal → tap right index+middle fingers on pulse point |
| **Help** | Flat left palm up → place right thumb-up fist firmly on top |
| **Pain** | Curled claw-hand in front of right chest → **twist wrist twice slowly and deliberately** |

> [!TIP] For best results on **Call**: make sure your whole arm — from shoulder to hand — is visible in the dashed ROI box. The box now covers head-to-waist to prevent the raised hand from leaving the frame.

> [!TIP] For best results on **Pain**: perform the wrist-twist slowly and hold each position for a full half-second. Quick twitches may fall below the motion threshold.

---

## Technology Stack

| Component | Technology |
|-----------|-----------|
| Layout | Pure HTML5 / CSS3 (no frameworks) |
| Webcam capture | MediaDevices Web API |
| Frame encoding | Canvas 2D → JPEG → Base64 |
| Speech output | Web Speech API (`speechSynthesis`) |
| Data fetching | Fetch API (async/await) |
| Charts | Vanilla JS (custom progress bars) |
