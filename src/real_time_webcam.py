"""Real-time ISL recognition from a webcam or a video file, with an OpenCV HUD and speech.

Live input is prepared exactly like the training clips: frames are cropped to the
same region, kept in a rolling window as long as a typical training clip, and 16
frames spread evenly across that window are sent to the model.

Usage:
    python -m src.real_time_webcam --source 0
    python -m src.real_time_webcam --source data/raw/help/help_Crop_001_01.avi
    python -m src.real_time_webcam --source clip.avi --no-display --max-frames 300   # benchmark
"""
import argparse
import collections
import time

import cv2
import numpy as np

from src import preprocess as P
from src.dataset import NONE_CLASS
from src.inference import CONFIDENCE_THRESHOLD, FrameWindow, load_model, predict_clip, sample_clip
from src.tts_helper import TextToSpeechEngine

# Colour palette (BGR)
COLOR_BG = (20, 24, 30)
COLOR_PANEL = (35, 40, 50)
COLOR_ACCENT = (255, 150, 50)
COLOR_GREEN = (50, 205, 50)
COLOR_AMBER = (0, 165, 255)
COLOR_WHITE = (245, 245, 245)
COLOR_GRAY = (140, 140, 140)
COLOR_ROI = (0, 255, 255)

PANEL_W = 300
HEADER_H = 50
SPEAK_THRESHOLD = 0.85
HITS_TO_CONFIRM = 3


def draw_hud(frame, roi_rect, current_pred, current_conf, class_probs, fps, speech_enabled, history, window_fill):
    """Header on top, video below it on the left, prediction panel on the right (nothing covers the video)."""
    h, w = frame.shape[:2]
    canvas_h = max(HEADER_H + h, 300 + 26 * len(class_probs) + 110)
    canvas = np.full((canvas_h, w + PANEL_W, 3), COLOR_BG, np.uint8)
    canvas[HEADER_H:HEADER_H + h, :w] = frame
    cv2.rectangle(canvas, (w, HEADER_H), (w + PANEL_W, canvas_h), COLOR_PANEL, -1)

    cv2.putText(canvas, "ISL Emergency Gesture Recognition", (15, 33),
                cv2.FONT_HERSHEY_DUPLEX, 0.7, COLOR_WHITE, 1, cv2.LINE_AA)
    status = f"[SPEECH: {'ON' if speech_enabled else 'OFF'}]  FPS: {fps:.1f}"
    cv2.putText(canvas, status, (w + 15, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                COLOR_GREEN if speech_enabled else COLOR_GRAY, 1, cv2.LINE_AA)

    # Region the model sees
    if roi_rect is not None:
        x0, y0, side = roi_rect
        y0 += HEADER_H
        cv2.rectangle(canvas, (x0, y0), (x0 + side, y0 + side), COLOR_ROI, 2)
        cv2.putText(canvas, "Place chest & hands in box (face above)", (x0 + 8, y0 + 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, COLOR_ROI, 1, cv2.LINE_AA)

    px = w + 15
    cv2.putText(canvas, "LIVE PREDICTION", (px, 80), cv2.FONT_HERSHEY_DUPLEX, 0.55, COLOR_WHITE, 1, cv2.LINE_AA)
    banner_col = COLOR_GREEN if current_conf >= SPEAK_THRESHOLD else (COLOR_AMBER if current_conf >= 0.60 else COLOR_GRAY)
    cv2.rectangle(canvas, (px, 92), (w + PANEL_W - 15, 142), banner_col, -1)
    cv2.putText(canvas, current_pred.upper(), (px + 10, 126), cv2.FONT_HERSHEY_DUPLEX, 0.75, (10, 10, 10), 2, cv2.LINE_AA)
    cv2.putText(canvas, f"Confidence: {int(current_conf * 100)}%", (px, 165),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLOR_WHITE, 1, cv2.LINE_AA)

    # How full the gesture window is (it must span a whole sign before predicting)
    cv2.putText(canvas, "Window", (px, 192), cv2.FONT_HERSHEY_SIMPLEX, 0.45, COLOR_WHITE, 1, cv2.LINE_AA)
    bar_x, bar_len = px + 70, PANEL_W - 100
    cv2.rectangle(canvas, (bar_x, 182), (bar_x + bar_len, 194), (60, 65, 75), -1)
    cv2.rectangle(canvas, (bar_x, 182), (bar_x + int(bar_len * min(window_fill, 1.0)), 194), COLOR_ACCENT, -1)

    cv2.putText(canvas, "CLASS CONFIDENCES", (px, 225), cv2.FONT_HERSHEY_DUPLEX, 0.5, COLOR_ACCENT, 1, cv2.LINE_AA)
    bar_y = 240
    for cls, prob in class_probs.items():
        cv2.putText(canvas, cls.capitalize(), (px, bar_y + 12), cv2.FONT_HERSHEY_SIMPLEX, 0.45, COLOR_WHITE, 1, cv2.LINE_AA)
        bar_x, bar_len = px + 75, PANEL_W - 105
        cv2.rectangle(canvas, (bar_x, bar_y + 2), (bar_x + bar_len, bar_y + 14), (60, 65, 75), -1)
        fill = int(bar_len * prob)
        if fill > 0:
            col = COLOR_GREEN if prob >= SPEAK_THRESHOLD else (COLOR_AMBER if prob >= 0.5 else (180, 180, 180))
            cv2.rectangle(canvas, (bar_x, bar_y + 2), (bar_x + fill, bar_y + 14), col, -1)
        cv2.putText(canvas, f"{int(prob * 100)}%", (bar_x + bar_len - 35, bar_y + 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, (20, 20, 20), 1, cv2.LINE_AA)
        bar_y += 26

    cv2.putText(canvas, "RECENT SIGNS", (px, bar_y + 20), cv2.FONT_HERSHEY_DUPLEX, 0.5, COLOR_ACCENT, 1, cv2.LINE_AA)
    hist_y = bar_y + 42
    for item in list(history)[-3:]:
        cv2.putText(canvas, f"* {item}", (px + 5, hist_y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 220, 240), 1, cv2.LINE_AA)
        hist_y += 20

    cv2.putText(canvas, "[Q] Quit  [S] Speech  [C] Clear", (px, canvas_h - 12),
                cv2.FONT_HERSHEY_SIMPLEX, 0.42, COLOR_GRAY, 1, cv2.LINE_AA)
    return canvas


def run_real_time_inference(source='0', model_path='models/best_isl_model.pth', crop=None,
                            window_seconds=None, motion_threshold=P.DEFAULT_MOTION_THRESHOLD,
                            stride=2, display=True, max_frames=None, speech=True):
    lm = load_model(model_path)
    window_seconds = window_seconds or lm.window_seconds
    print(f'Model loaded on {lm.device}. Classes: {lm.classes}. Gesture window: {window_seconds:.2f}s')

    try:
        src, is_file = int(source), False
    except ValueError:
        src, is_file = source, True
    # Dataset clips are already cropped; live camera frames need the same framing.
    if crop is None:
        crop = not is_file

    cap = cv2.VideoCapture(src)
    if not cap.isOpened():
        print(f'Error: could not open video source "{source}".')
        print('Tip: use --source <path_to_video.avi> to test with a video file.')
        return []
    src_fps = (cap.get(cv2.CAP_PROP_FPS) or P.DEFAULT_FPS) if is_file else None

    tts = TextToSpeechEngine(cooldown=2.5)
    tts.enabled = speech
    window = FrameWindow(window_seconds)
    history = collections.deque(maxlen=5)
    spoken = []

    smoothed_probs = np.zeros(len(lm.classes), dtype=np.float32)
    current_pred, current_conf, consecutive_hits = 'Filling window...', 0.0, 0
    fps, prev_time = 30.0, time.time()
    frame_idx = 0
    infer_ms = []

    print('Starting. Press "q" to quit, "s" to toggle speech, "c" to clear history.')
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                if is_file and (display or max_frames) and frame_idx > 0:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # loop the clip
                    continue
                break
            frame_idx += 1
            now = time.time()
            fps = 0.9 * fps + 0.1 * (1.0 / max(now - prev_time, 1e-4))
            prev_time = now

            # Video files use their own clock so the window spans the same number of seconds.
            t = frame_idx / src_fps if is_file else now
            window.add(P.prepare_frame(frame, crop=crop), t)

            if frame_idx % stride == 0 and window.ready(lm.num_frames):
                sampled = sample_clip(window.frames(), lm.num_frames)
                if P.motion_energy(sampled) < motion_threshold:
                    current_pred, current_conf, consecutive_hits = 'Waiting (idle)', 0.0, 0
                    smoothed_probs[:] = 0
                else:
                    t0 = time.perf_counter()
                    raw_probs, _ = predict_clip(lm, sampled)
                    infer_ms.append(1000 * (time.perf_counter() - t0))

                    smoothed_probs = 0.65 * raw_probs + 0.35 * smoothed_probs
                    top_idx = int(np.argmax(smoothed_probs))
                    top_conf = float(smoothed_probs[top_idx])
                    pred_name = lm.classes[top_idx]
                    sorted_probs = np.sort(smoothed_probs)[::-1]
                    margin = sorted_probs[0] - sorted_probs[1] if len(sorted_probs) > 1 else 1.0

                    if pred_name == NONE_CLASS:
                        current_pred, current_conf, consecutive_hits = 'No sign', 0.0, 0
                    elif top_conf >= CONFIDENCE_THRESHOLD and margin >= 0.12:
                        current_pred, current_conf = pred_name, top_conf
                        consecutive_hits += 1
                        if consecutive_hits >= HITS_TO_CONFIRM and top_conf >= SPEAK_THRESHOLD:
                            if not history or history[-1].split()[0] != pred_name.upper():
                                history.append(f"{pred_name.upper()} ({int(top_conf * 100)}%)")
                                spoken.append(pred_name)
                                tts.speak(pred_name)
                                if not display:
                                    print(f'[{t:7.2f}s] {pred_name} ({top_conf:.0%})')
                    else:
                        consecutive_hits = max(0, consecutive_hits - 1)
                        if consecutive_hits == 0:
                            current_pred, current_conf = 'Detecting...', top_conf
            elif not window.ready(lm.num_frames):
                current_pred = 'Filling window...'

            if display:
                h, w = frame.shape[:2]
                probs = {c: float(smoothed_probs[i]) for i, c in enumerate(lm.classes)}
                canvas = draw_hud(frame, P.roi_box(w, h) if crop else None, current_pred, current_conf,
                                  probs, fps, tts.enabled, history, window.span() / window_seconds)
                cv2.imshow('Indian Sign Language Recognizer', canvas)
                delay = max(1, int(1000 / src_fps - 1000 * (time.time() - now))) if is_file else 1
                key = cv2.waitKey(delay) & 0xFF
                if key in (ord('q'), 27):
                    break
                if key == ord('s'):
                    print(f"Speech toggled: {'ON' if tts.toggle() else 'OFF'}")
                elif key == ord('c'):
                    history.clear()

            if max_frames and frame_idx >= max_frames:
                break
    finally:
        cap.release()
        if display:
            cv2.destroyAllWindows()
        if infer_ms:
            print(f'Inference: {np.mean(infer_ms):.1f} ms per window on {lm.device} '
                  f'({len(infer_ms)} windows, {frame_idx} frames)')
        print('Real-time recogniser closed.')
    return spoken


def main():
    parser = argparse.ArgumentParser(description='Real-Time Indian Sign Language Recognition')
    parser.add_argument('--source', default='0', help='Webcam index (0) or path to a video file')
    parser.add_argument('--model', default='models/best_isl_model.pth', help='Path to a checkpoint from src.train')
    crop = parser.add_mutually_exclusive_group()
    crop.add_argument('--crop', dest='crop', action='store_true', default=None,
                      help='Crop to the centre box (default for webcams)')
    crop.add_argument('--no-crop', dest='crop', action='store_false',
                      help='Use the full frame (default for dataset video files, which are already cropped)')
    parser.add_argument('--window-seconds', type=float, default=None,
                        help='Length of the gesture window (default: typical training clip length from the checkpoint)')
    parser.add_argument('--motion-threshold', type=float, default=P.DEFAULT_MOTION_THRESHOLD)
    parser.add_argument('--stride', type=int, default=2, help='Run the model every N frames')
    parser.add_argument('--no-display', action='store_true', help='No window; print detections (for testing)')
    parser.add_argument('--no-speech', action='store_true')
    parser.add_argument('--max-frames', type=int, default=None)
    args = parser.parse_args()

    run_real_time_inference(source=args.source, model_path=args.model, crop=args.crop,
                            window_seconds=args.window_seconds, motion_threshold=args.motion_threshold,
                            stride=args.stride, display=not args.no_display, max_frames=args.max_frames,
                            speech=not args.no_speech)


if __name__ == '__main__':
    main()
