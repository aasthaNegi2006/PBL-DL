# ⚙️ Backend

This folder contains the **FastAPI server**, **model inference engine**, **desktop webcam recognizer**, and **text-to-speech helper**.

---

## Files

| File | Purpose |
|------|---------|
| `inference.py` | Load model, run predictions, rolling frame window |
| `tts_helper.py` | Non-blocking OS text-to-speech (Windows SAPI / macOS say / Linux espeak) |
| `real_time_webcam.py` | OpenCV desktop app with live HUD and speech output |

The main FastAPI server entry point is in `app/main.py` (kept there for compatibility with one-click `.bat` launchers).

---

## API Routes

| Route | Method | Purpose |
|-------|--------|---------|
| `/` | GET | HTML dashboard (served from `frontend/`) |
| `/api/info` | GET | Model metadata, eval provenance, classes |
| `/api/predict` | POST | Upload a video file → get predicted gesture |
| `/api/predict-frames` | POST | Send base64-encoded webcam frames → real-time prediction |
| `/eval-results/{filename}` | GET | Download evaluation JSON artifacts |

### `/api/predict-frames` Payload
```json
{
  "frames": ["data:image/jpeg;base64,...", "..."],
  "crop": true
}
```

### `/api/info` Response (excerpt)
```json
{
  "status": "online",
  "model_loaded": true,
  "supported_classes": ["accident", "call", "doctor", "help", "pain", "none"],
  "evaluation_provenance_verified": true,
  "evaluation_summary": { "accuracy": 0.975, "macro_f1": 0.9778 }
}
```

---

## Desktop Webcam App

```bash
python -m src.real_time_webcam --source 0
# or double-click:
run_desktop_webcam.bat
```

**Controls:**
- `S` — toggle speech on/off
- `C` — clear history
- `Q` / `ESC` — quit

**Tuned thresholds** (as of this version):

| Setting | Value | Reason |
|---------|-------|--------|
| `DEFAULT_ROI` | cx=0.5, cy=0.50, size=0.88 | Full upper-body capture so raised 'call' hand stays in frame |
| `DEFAULT_MOTION_THRESHOLD` | 1.8 | Catches subtle 'pain' wrist-twist; filters true resting noise |
| `CONFIDENCE_THRESHOLD` | 0.70 | Accepts 'pain'/'call' peak at 0.75-0.82 live confidence |
| `SPEAK_THRESHOLD` | 0.70 | Trigger speech at same confidence level |
| `HITS_TO_CONFIRM` | 2 | 2 consecutive windows (not 3) for faster feedback |
| `MARGIN_THRESHOLD` | 0.08 | 'call'/'pain' sometimes have 0.09 margin; 0.12 blocked them |

---

## Text-to-Speech

Uses the OS native voice:
- **Windows**: PowerShell `SAPI.SpVoice` (no install needed)
- **macOS**: `say`
- **Linux**: `spd-say` / `espeak-ng` / `espeak`

A 2.5s cooldown prevents repeated announcements of the same gesture.
