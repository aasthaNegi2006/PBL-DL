import os
import io
import json
import tempfile
import cv2
import torch
import numpy as np
from fastapi import FastAPI, File, UploadFile, Request
from fastapi.responses import HTMLResponse, JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware

from src.model import ISLGestureRecognizer
from src.dataset import CLASSES

app = FastAPI(
    title="Indian Sign Language (ISL) Recognition API",
    description="Emergency gesture recognition powered by PyTorch Spatio-Temporal Deep Learning",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
MODEL_PATH = 'models/best_isl_model.pth'

model = None
checkpoint_meta = {}

def get_model():
    global model, checkpoint_meta
    if model is None:
        if os.path.exists(MODEL_PATH):
            checkpoint = torch.load(MODEL_PATH, map_location=DEVICE, weights_only=False)
            checkpoint_meta = {
                'epoch': checkpoint.get('epoch', 40),
                'val_acc': checkpoint.get('val_acc', 100.0),
                'num_frames': checkpoint.get('num_frames', 16),
                'classes': checkpoint.get('classes', CLASSES)
            }
            model = ISLGestureRecognizer(
                num_classes=len(CLASSES),
                num_frames=checkpoint.get('num_frames', 16),
                in_channels=3,
                feature_dim=checkpoint.get('feature_dim', 256),
                hidden_dim=checkpoint.get('hidden_dim', 128)
            ).to(DEVICE)
            model.load_state_dict(checkpoint['model_state_dict'])
            model.eval()
            print("Loaded PyTorch ISL Model successfully.")
        else:
            raise FileNotFoundError(f"Model file not found at {MODEL_PATH}")
    return model

@app.on_event("startup")
def startup_event():
    try:
        get_model()
    except Exception as e:
        print(f"Warning on startup: {e}")

@app.get("/api/info")
def get_info():
    summary_path = "eval_results/summary.json"
    summary_data = {}
    if os.path.exists(summary_path):
        with open(summary_path, "r") as f:
            summary_data = json.load(f)
            
    return {
        "status": "online",
        "device": str(DEVICE),
        "supported_classes": CLASSES,
        "model_architecture": "Spatio-Temporal CNN + BiGRU + Temporal Attention",
        "num_frames_per_sequence": 16,
        "input_resolution": "112x112",
        "evaluation_summary": summary_data
    }

@app.get("/eval-results/{filename}")
def get_eval_image(filename: str):
    safe_files = ["confusion_matrix.png", "training_curves.png", "classification_report.txt"]
    if filename in safe_files:
        filepath = os.path.join("eval_results", filename)
        if os.path.exists(filepath):
            return FileResponse(filepath)
    return JSONResponse(status_code=404, content={"error": "File not found"})

@app.post("/api/predict")
async def predict_gesture(file: UploadFile = File(...)):
    net = get_model()
    
    # Save uploaded file to temp file
    suffix = os.path.splitext(file.filename)[1] or ".avi"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        contents = await file.read()
        tmp.write(contents)
        tmp_path = tmp.name
        
    try:
        cap = cv2.VideoCapture(tmp_path)
        frames = []
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame_rgb = cv2.cvtColor(cv2.resize(frame, (112, 112)), cv2.COLOR_BGR2RGB)
            frames.append(frame_rgb)
        cap.release()
        
        if len(frames) == 0:
            return JSONResponse(status_code=400, content={"error": "Could not read frames from uploaded video."})
            
        # Sample 16 frames uniformly
        indices = np.linspace(0, len(frames) - 1, 16, dtype=int)
        sampled = np.array(frames)[indices].astype(np.float32) / 255.0
        
        # Standardize
        tensor = torch.from_numpy(sampled).permute(0, 3, 1, 2).float().unsqueeze(0).to(DEVICE)
        mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 1, 3, 1, 1).to(DEVICE)
        std = torch.tensor([0.229, 0.224, 0.225]).view(1, 1, 3, 1, 1).to(DEVICE)
        tensor = (tensor - mean) / std
        
        with torch.no_grad():
            logits, attn = net(tensor)
            probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
            attn_weights = attn.squeeze().cpu().numpy().tolist()
            
        top_idx = int(np.argmax(probs))
        pred_class = CLASSES[top_idx]
        confidence = float(probs[top_idx])
        
        breakdown = {CLASSES[i]: float(probs[i]) for i in range(len(CLASSES))}
        
        return {
            "prediction": pred_class,
            "confidence_percentage": round(confidence * 100, 2),
            "class_probabilities": breakdown,
            "total_frames_in_video": len(frames),
            "attention_weights": [round(w, 4) for w in attn_weights]
        }
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)

@app.get("/", response_class=HTMLResponse)
def index_page():
    template_path = os.path.join("app", "templates", "index.html")
    if os.path.exists(template_path):
        with open(template_path, "r", encoding="utf-8") as f:
            return f.read()
    return "<h1>ISL Recognition API is running.</h1>"
