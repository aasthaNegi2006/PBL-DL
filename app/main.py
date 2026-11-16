import base64
import binascii
import json
import os
import shutil
import tempfile
import threading
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated

import cv2
import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse
from pydantic import BaseModel, Field, StringConstraints

from src import preprocess as P
from src.dataset import NONE_CLASS
from src.inference import CONFIDENCE_THRESHOLD, load_model, predict_clip, sample_clip

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = Path(os.environ.get('ISL_MODEL_PATH', BASE_DIR / 'models' / 'best_isl_model.pth'))
EVAL_DIR = BASE_DIR / 'eval_results'
TEMPLATE_PATH = BASE_DIR / 'app' / 'templates' / 'index.html'

MAX_UPLOAD_BYTES = 50 * 1024 * 1024   # 50 MB
MAX_VIDEO_FRAMES = 900                # 30 s at 30 FPS; a single sign is 2-4 s
MAX_LIVE_FRAMES = 64
MAX_FRAME_B64_CHARS = 400_000         # ~300 KB JPEG per frame
MIN_LIVE_FRAMES = 8
VIDEO_EXTS = {'.avi', '.mp4', '.mov', '.mkv', '.webm'}
EVAL_FILES = {'confusion_matrix.png', 'training_curves.png', 'classification_report.txt',
              'cv_confusion_matrix.png', 'cv_classification_report.txt'}

_model_lock = threading.Lock()
_state = {'lm': None, 'error': None}


def get_model():
    """Load the model on first use. Returns 503 (not a crash) if it is missing."""
    if _state['lm'] is None:
        with _model_lock:
            if _state['lm'] is None:
                if not Path(MODEL_PATH).exists():
                    _state['error'] = f'Model file not found at {MODEL_PATH}. Train one with: python -m src.train'
                    raise HTTPException(status_code=503, detail=_state['error'])
                _state['lm'] = load_model(MODEL_PATH)
                _state['error'] = None
                print(f'Loaded ISL model from {MODEL_PATH} on {_state["lm"].device}.')
    return _state['lm']


@asynccontextmanager
async def lifespan(_app):
    try:
        get_model()
    except Exception as e:  # keep serving the dashboard; prediction endpoints return 503
        detail = getattr(e, 'detail', e)
        print(f'Warning on startup: {detail}')
    yield


app = FastAPI(
    title="Indian Sign Language (ISL) Recognition API",
    description="Emergency gesture recognition powered by PyTorch Spatio-Temporal Deep Learning",
    version="1.1.0",
    lifespan=lifespan,
)

# The dashboard is served from this app, so it needs no CORS. To call the API from
# another site, list its origins, e.g. ISL_CORS_ORIGINS=http://localhost:3000
_cors_origins = [o.strip() for o in os.environ.get('ISL_CORS_ORIGINS', '').split(',') if o.strip()]
if _cors_origins:
    app.add_middleware(CORSMiddleware, allow_origins=_cors_origins, allow_methods=['GET', 'POST'],
                       allow_headers=['Content-Type'])


def _read_json(path):
    if path.exists():
        with open(path) as f:
            return json.load(f)
    return {}


def _result(lm, probs, motion=None, extra=None, live=True):
    """Shared response shape for both prediction endpoints.

    Live predictions below the confidence threshold are reported as "detecting";
    an uploaded clip always reports its top class, flagged with low_confidence.
    """
    top_idx = int(np.argmax(probs))
    pred_class = lm.classes[top_idx]
    confidence = float(probs[top_idx])
    body = {
        'is_idle': False,
        'confidence_percentage': round(confidence * 100, 2),
        'low_confidence': confidence < CONFIDENCE_THRESHOLD,
        'class_probabilities': {c: float(p) for c, p in zip(lm.classes, probs)},
    }
    if motion is not None:
        body['avg_motion'] = round(motion, 2)
    if pred_class == NONE_CLASS:
        body.update(prediction='none', message='No emergency sign recognised.')
    elif live and confidence < CONFIDENCE_THRESHOLD:
        body.update(prediction='detecting', top_class=pred_class,
                    message=f'Detecting gesture... ({pred_class} {int(confidence * 100)}%)')
    else:
        body['prediction'] = pred_class
    body.update(extra or {})
    return body


@app.get("/api/info")
def get_info():
    lm = None
    try:
        lm = get_model()
    except HTTPException:
        pass
    return {
        'status': 'online',
        'model_loaded': lm is not None,
        'model_error': _state['error'],
        'device': str(lm.device) if lm else None,
        'supported_classes': lm.classes if lm else [],
        'model_architecture': 'Spatio-Temporal CNN + BiGRU + Temporal Attention',
        'num_frames_per_sequence': lm.num_frames if lm else P.NUM_FRAMES,
        'window_seconds': lm.window_seconds if lm else P.DEFAULT_WINDOW_SECONDS,
        'input_resolution': f'{P.IMG_SIZE}x{P.IMG_SIZE}',
        'roi': P.DEFAULT_ROI,
        'motion_threshold': P.DEFAULT_MOTION_THRESHOLD,
        'confidence_threshold': CONFIDENCE_THRESHOLD,
        'checkpoint': {k: v for k, v in (lm.meta if lm else {}).items() if k != 'split'},
        'evaluation_summary': _read_json(EVAL_DIR / 'summary.json'),
        'cross_validation': {k: v for k, v in _read_json(EVAL_DIR / 'cross_validation.json').items() if k != 'folds'},
    }


@app.get("/eval-results/{filename}")
def get_eval_file(filename: str):
    path = EVAL_DIR / filename
    if filename in EVAL_FILES and path.exists():
        return FileResponse(path)
    raise HTTPException(status_code=404, detail='File not found')


B64Frame = Annotated[str, StringConstraints(max_length=MAX_FRAME_B64_CHARS)]


class FrameSequencePayload(BaseModel):
    # Frames spread evenly across the last gesture-length window (see /api/info window_seconds).
    frames: list[B64Frame] = Field(min_length=1, max_length=MAX_LIVE_FRAMES)
    # Crop to the same centre box as the desktop app. Set false for already-cropped frames.
    crop: bool = True


@app.post("/api/predict-frames")
def predict_from_base64_frames(payload: FrameSequencePayload):
    lm = get_model()

    decoded = []
    for item in payload.frames:
        if ',' in item:
            item = item.split(',', 1)[1]
        try:
            img_bytes = base64.b64decode(item, validate=True)
        except (binascii.Error, ValueError):
            raise HTTPException(status_code=400, detail='A frame is not valid base64.')
        img = cv2.imdecode(np.frombuffer(img_bytes, np.uint8), cv2.IMREAD_COLOR)
        if img is not None:
            decoded.append(P.prepare_frame(img, crop=payload.crop))

    if len(decoded) < MIN_LIVE_FRAMES:
        raise HTTPException(status_code=400, detail=f'Need at least {MIN_LIVE_FRAMES} decodable frames, got {len(decoded)}.')

    sampled = sample_clip(decoded, lm.num_frames)
    motion = P.motion_energy(sampled)
    # Idle filter: if the hands are not moving, do not raise a false alarm
    if motion < P.DEFAULT_MOTION_THRESHOLD:
        return {
            'prediction': 'idle',
            'is_idle': True,
            'confidence_percentage': 0.0,
            'avg_motion': round(motion, 2),
            'message': 'Idle / no active gesture detected. Sign inside the box.',
            'class_probabilities': {c: 0.0 for c in lm.classes},
        }

    probs, _ = predict_clip(lm, sampled)
    return _result(lm, probs, motion)


@app.post("/api/predict")
def predict_gesture(file: UploadFile = File(...), crop: bool = False):
    """Classify one uploaded clip containing a single sign.

    Dataset clips are already cropped, so crop defaults to False. Pass
    ?crop=true for a full-frame recording, to crop it like the live camera.
    """
    lm = get_model()

    suffix = Path(file.filename or '').suffix.lower()
    suffix = suffix if suffix in VIDEO_EXTS else '.avi'
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp_path = tmp.name
        written = 0
        while chunk := file.file.read(1024 * 1024):
            written += len(chunk)
            if written > MAX_UPLOAD_BYTES:
                break
            tmp.write(chunk)

    try:
        if written > MAX_UPLOAD_BYTES:
            raise HTTPException(status_code=413, detail=f'Video is larger than {MAX_UPLOAD_BYTES // (1024 * 1024)} MB.')

        cap = cv2.VideoCapture(tmp_path)
        frames = []
        try:
            while len(frames) <= MAX_VIDEO_FRAMES:
                ok, frame = cap.read()
                if not ok:
                    break
                frames.append(P.prepare_frame(frame, crop=crop))
        finally:
            cap.release()

        if not frames:
            raise HTTPException(status_code=400, detail='Could not read frames from the uploaded video.')
        if len(frames) > MAX_VIDEO_FRAMES:
            raise HTTPException(status_code=413, detail=f'Video is longer than {MAX_VIDEO_FRAMES} frames. '
                                                        'Upload a clip containing a single sign.')

        probs, attn = predict_clip(lm, sample_clip(frames, lm.num_frames))
        return _result(lm, probs, live=False, extra={
            'total_frames_in_video': len(frames),
            'attention_weights': [round(float(w), 4) for w in attn],
        })
    finally:
        os.remove(tmp_path)


@app.get("/", response_class=HTMLResponse)
def index_page():
    if TEMPLATE_PATH.exists():
        return TEMPLATE_PATH.read_text(encoding='utf-8')
    return "<h1>ISL Recognition API is running.</h1>"
