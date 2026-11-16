import cv2
import torch
import numpy as np
import collections
import time
import argparse
import os

from src.model import ISLGestureRecognizer
from src.dataset import CLASSES
from src.tts_helper import TextToSpeechEngine

# Color Palette (BGR)
COLOR_BG = (20, 24, 30)
COLOR_PANEL = (35, 40, 50)
COLOR_ACCENT = (255, 150, 50)
COLOR_GREEN = (50, 205, 50)
COLOR_AMBER = (0, 165, 255)
COLOR_RED = (50, 50, 240)
COLOR_WHITE = (245, 245, 245)
COLOR_GRAY = (140, 140, 140)

def draw_hud(frame, current_pred, current_conf, class_probs, fps, speech_enabled, history):
    h, w = frame.shape[:2]
    overlay = frame.copy()
    
    # 1. Top Header Banner
    cv2.rectangle(overlay, (0, 0), (w, 55), COLOR_BG, -1)
    cv2.putText(overlay, "ISL Emergency Gesture Recognition System", (20, 36),
                cv2.FONT_HERSHEY_DUPLEX, 0.8, COLOR_WHITE, 2, cv2.LINE_AA)
    
    # Audio status pill
    status_str = "SPEECH: ON" if speech_enabled else "SPEECH: OFF"
    status_col = COLOR_GREEN if speech_enabled else COLOR_GRAY
    cv2.putText(overlay, f"[{status_str}]  FPS: {fps:.1f}", (w - 240, 35),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, status_col, 2, cv2.LINE_AA)
    
    # 2. Side Panel for Class Probabilities & Predictions
    panel_w = 280
    panel_x = w - panel_w - 15
    cv2.rectangle(overlay, (panel_x, 70), (w - 15, h - 20), COLOR_PANEL, -1)
    cv2.rectangle(overlay, (panel_x, 70), (w - 15, h - 20), COLOR_GRAY, 1)
    
    # Header in Side Panel
    cv2.putText(overlay, "LIVE PREDICTION", (panel_x + 15, 98),
                cv2.FONT_HERSHEY_DUPLEX, 0.55, COLOR_WHITE, 1, cv2.LINE_AA)
    
    # Current Pred Banner
    banner_col = COLOR_GREEN if current_conf >= 0.85 else (COLOR_AMBER if current_conf >= 0.60 else COLOR_GRAY)
    cv2.rectangle(overlay, (panel_x + 15, 115), (w - 30, 175), banner_col, -1)
    
    pred_text = current_pred.upper() if current_conf >= 0.55 else "WAITING..."
    cv2.putText(overlay, pred_text, (panel_x + 25, 155),
                cv2.FONT_HERSHEY_DUPLEX, 0.85, (10, 10, 10), 2, cv2.LINE_AA)
    
    conf_pct = int(current_conf * 100) if current_conf >= 0.55 else 0
    cv2.putText(overlay, f"Confidence: {conf_pct}%", (panel_x + 15, 205),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, COLOR_WHITE, 1, cv2.LINE_AA)
    
    # 3. Class probability bars
    cv2.putText(overlay, "CLASS CONFIDENCES", (panel_x + 15, 240),
                cv2.FONT_HERSHEY_DUPLEX, 0.5, COLOR_ACCENT, 1, cv2.LINE_AA)
    
    bar_y = 265
    for cls in CLASSES:
        prob = class_probs.get(cls, 0.0)
        # Class label
        cv2.putText(overlay, cls.capitalize(), (panel_x + 15, bar_y + 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, COLOR_WHITE, 1, cv2.LINE_AA)
        
        # Background bar
        bar_x = panel_x + 85
        bar_len = 160
        cv2.rectangle(overlay, (bar_x, bar_y + 2), (bar_x + bar_len, bar_y + 14), (60, 65, 75), -1)
        
        # Fill bar
        fill_len = int(bar_len * prob)
        bar_fill_col = COLOR_GREEN if prob >= 0.85 else (COLOR_AMBER if prob >= 0.5 else (180, 180, 180))
        if fill_len > 0:
            cv2.rectangle(overlay, (bar_x, bar_y + 2), (bar_x + fill_len, bar_y + 14), bar_fill_col, -1)
            
        cv2.putText(overlay, f"{int(prob * 100)}%", (bar_x + bar_len - 35, bar_y + 11),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, (20, 20, 20), 1, cv2.LINE_AA)
        bar_y += 30
        
    # 4. History Log
    cv2.putText(overlay, "RECENT SIGNS", (panel_x + 15, bar_y + 20),
                cv2.FONT_HERSHEY_DUPLEX, 0.5, COLOR_ACCENT, 1, cv2.LINE_AA)
    hist_y = bar_y + 45
    for item in list(history)[-3:]:
        cv2.putText(overlay, f"* {item}", (panel_x + 20, hist_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 220, 240), 1, cv2.LINE_AA)
        hist_y += 22
        
    # 5. Framing guide box on the left
    guide_size = min(h - 120, w - panel_w - 60)
    gx = 30
    gy = 75
    cv2.rectangle(overlay, (gx, gy), (gx + guide_size, gy + guide_size), (0, 255, 255), 2)
    cv2.putText(overlay, "Gesture Region of Interest (ROI)", (gx + 10, gy + 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1, cv2.LINE_AA)
    
    # 6. Bottom Controls Hint
    cv2.putText(overlay, "[Q] Quit  |  [S] Toggle Speech  |  [C] Clear History",
                (30, h - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.45, COLOR_GRAY, 1, cv2.LINE_AA)
                
    alpha = 0.88
    return cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)


def run_real_time_inference(source=0, model_path='models/best_isl_model.pth'):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Loading model from {model_path} on {device}...')
    
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    num_frames = checkpoint.get('num_frames', 16)
    
    model = ISLGestureRecognizer(
        num_classes=len(CLASSES),
        num_frames=num_frames,
        in_channels=3,
        feature_dim=checkpoint.get('feature_dim', 256),
        hidden_dim=checkpoint.get('hidden_dim', 128)
    ).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    print('Model ready for real-time inference.')
    
    tts = TextToSpeechEngine(cooldown=2.5)
    
    # Handle int source or string source
    try:
        src = int(source)
    except ValueError:
        src = source
        
    cap = cv2.VideoCapture(src)
    if not cap.isOpened():
        print(f'Error: Could not open video source "{source}".')
        print('Tip: Use --source <path_to_video.avi> to test with a video file.')
        return

    # Frame queue for model sequence
    frame_buffer = collections.deque(maxlen=num_frames)
    history = collections.deque(maxlen=5)
    
    # Prediction smoothing
    smoothed_probs = np.zeros(len(CLASSES), dtype=np.float32)
    current_pred = "Waiting"
    current_conf = 0.0
    consecutive_hits = 0
    
    prev_time = time.time()
    fps = 30.0
    step_counter = 0
    
    print('\nStarting Live Feed. Press "q" in the window to quit, "s" to toggle speech.\n')
    
    mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(device)
    std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(device)

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                # If reading from video file, loop back to start
                if isinstance(src, str):
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    continue
                else:
                    break
                    
            now = time.time()
            fps = 0.9 * fps + 0.1 * (1.0 / max(now - prev_time, 1e-4))
            prev_time = now
            step_counter += 1
            
            # Crop ROI box where hands perform the gesture
            h, w = frame.shape[:2]
            panel_w = 280
            guide_size = min(h - 120, w - panel_w - 60)
            gx, gy = 30, 75
            roi = frame[gy:gy+guide_size, gx:gx+guide_size]
            if roi.shape[0] < 20 or roi.shape[1] < 20:
                roi = frame
                
            frame_rgb = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
            frame_resized = cv2.resize(frame_rgb, (112, 112)).astype(np.float32) / 255.0
            frame_buffer.append(frame_resized)
            
            # Run inference every 2 frames when buffer is full
            if len(frame_buffer) == num_frames and (step_counter % 2 == 0):
                # 1. Motion gating: check if hands are actually moving
                diffs = [
                    np.mean(np.abs(frame_buffer[i] - frame_buffer[i-1])) * 255.0
                    for i in range(1, len(frame_buffer))
                ]
                avg_motion = float(np.mean(diffs)) if diffs else 0.0
                
                if avg_motion < 2.5:
                    current_pred = "Waiting (Idle)"
                    current_conf = 0.0
                    consecutive_hits = 0
                    smoothed_probs = np.zeros(len(CLASSES), dtype=np.float32)
                else:
                    # Prepare tensor (1, T, C, H, W)
                    tensor_seq = torch.from_numpy(np.array(frame_buffer)).permute(0, 3, 1, 2).float().unsqueeze(0).to(device)
                    tensor_seq = (tensor_seq - mean) / std
                    
                    with torch.no_grad():
                        logits, _ = model(tensor_seq)
                        raw_probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
                        
                    # Exponential moving average filter
                    smoothed_probs = 0.65 * raw_probs + 0.35 * smoothed_probs
                    top_idx = int(np.argmax(smoothed_probs))
                    top_conf = float(smoothed_probs[top_idx])
                    
                    if top_conf >= 0.75:
                        pred_name = CLASSES[top_idx]
                        current_pred = pred_name
                        current_conf = top_conf
                        consecutive_hits += 1
                        
                        if consecutive_hits >= 3 and top_conf >= 0.85:
                            entry = f"{pred_name.upper()} ({int(top_conf*100)}%)"
                            if len(history) == 0 or history[-1].split()[0] != pred_name.upper():
                                history.append(entry)
                                tts.speak(pred_name)
                    else:
                        consecutive_hits = max(0, consecutive_hits - 1)
                        if consecutive_hits == 0:
                            current_conf = top_conf
                            current_pred = "Detecting..."

            # Render HUD
            class_prob_dict = {CLASSES[i]: float(smoothed_probs[i]) for i in range(len(CLASSES))}
            display_frame = draw_hud(frame, current_pred, current_conf, class_prob_dict, fps, tts.enabled, history)
            
            cv2.imshow("Indian Sign Language Recognizer", display_frame)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == 27:
                break
            elif key == ord('s'):
                state = tts.toggle()
                print(f"Speech toggled: {'ON' if state else 'OFF'}")
            elif key == ord('c'):
                history.clear()
    finally:
        cap.release()
        cv2.destroyAllWindows()
        print('Real-time recognizer closed.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Real-Time Indian Sign Language Recognition")
    parser.add_argument('--source', type=str, default='0', help='Webcam index (0) or path to video file')
    parser.add_argument('--model', type=str, default='models/best_isl_model.pth', help='Path to saved model checkpoint')
    args = parser.parse_args()
    
    run_real_time_inference(source=args.source, model_path=args.model)
