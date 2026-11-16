"""Build the frame cache (data/processed/cached_all_frames.pt) from the raw videos.

Dataset: "A Video Dataset of the Hand Gestures of Indian Sign Language Words used
in Emergency Situations", Adithya V. and Rajesh R., Mendeley Data, CC BY 4.0.
https://data.mendeley.com/datasets/2vfdm42337/1

Put the cropped videos anywhere under --raw-dir (for example data/raw/help/help_Crop_001_01.avi).
Files are matched by name: <word>_crop_<subject>_<take>.<ext>, case-insensitive.

Usage:
    python -m src.data_prep
    python -m src.data_prep --none-from hot lose thief   # adds a "none" class (recommended)
"""
import argparse
import os
import re
from collections import Counter

import cv2
import numpy as np
import torch

from src import preprocess as P
from src.dataset import CLASSES, DEFAULT_CACHE, NONE_CLASS

VIDEO_EXTS = {'.avi', '.mp4', '.mov', '.mkv'}
_NAME_RE = re.compile(r'^(?P<word>[a-z]+)_(?P<crop>crop_)?(?P<subject>\d{3})_(?P<take>\d{2})$', re.IGNORECASE)


def read_video(path):
    cap = cv2.VideoCapture(path)
    fps = cap.get(cv2.CAP_PROP_FPS) or P.DEFAULT_FPS
    frames = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frames.append(P.prepare_frame(frame, crop=False))
    cap.release()
    return frames, float(fps)


def find_videos(raw_dir, include_uncropped=False):
    for root, _, files in os.walk(raw_dir):
        for name in sorted(files):
            stem, ext = os.path.splitext(name)
            if ext.lower() not in VIDEO_EXTS:
                continue
            match = _NAME_RE.match(stem)
            if not match:
                continue
            if not match.group('crop') and not include_uncropped:
                continue
            yield os.path.join(root, name), match.group('word').lower(), match.group('subject'), match.group('take')


def main():
    parser = argparse.ArgumentParser(description='Build the ISL frame cache from raw videos.')
    parser.add_argument('--raw-dir', default=os.path.join('data', 'raw'))
    parser.add_argument('--out', default=DEFAULT_CACHE)
    parser.add_argument('--classes', nargs='+', default=CLASSES, help='Words to recognise.')
    parser.add_argument('--none-from', nargs='*', default=[],
                        help='Other words (e.g. hot lose thief) to label as a single "none" class, '
                             'so the model can say "not one of these signs".')
    parser.add_argument('--include-uncropped', action='store_true',
                        help='Also use the raw (uncropped) videos. Off by default because the '
                             'two versions are the same recordings.')
    args = parser.parse_args()

    classes = [c.lower() for c in args.classes]
    none_words = {w.lower() for w in args.none_from}
    if none_words & set(classes):
        parser.error(f'--none-from overlaps --classes: {sorted(none_words & set(classes))}')
    if none_words:
        classes = classes + [NONE_CLASS]
    class_to_idx = {c: i for i, c in enumerate(classes)}

    samples = []
    for path, word, subject, take in find_videos(args.raw_dir, args.include_uncropped):
        if word in none_words:
            class_name = NONE_CLASS
        elif word in class_to_idx:
            class_name = word
        else:
            continue
        frames, fps = read_video(path)
        if len(frames) < 2:
            print(f'Skipping {path}: could not read frames')
            continue
        samples.append({
            'frames': torch.from_numpy(np.stack(frames)),  # uint8 (N, 112, 112, 3)
            'total_frames': len(frames),
            'fps': fps,
            'label': class_to_idx[class_name],
            'class_name': class_name,
            'word': word,
            'subject': subject,
            'take': take,
            'video_name': os.path.basename(path),
        })

    if not samples:
        raise SystemExit(f'No matching videos found under {args.raw_dir}. '
                         'Expected names like help_Crop_001_01.avi.')

    counts = Counter(s['class_name'] for s in samples)
    subjects = sorted({s['subject'] for s in samples})
    durations = [s['total_frames'] / s['fps'] for s in samples]
    print(f'{len(samples)} clips, {len(subjects)} subjects')
    for c in classes:
        print(f'  {c:10s} {counts.get(c, 0)}')
    print(f'Clip length: median {np.median(durations):.2f}s, min {min(durations):.2f}s, max {max(durations):.2f}s')

    os.makedirs(os.path.dirname(args.out) or '.', exist_ok=True)
    torch.save({
        'samples': samples,
        'classes': classes,
        'meta': {'img_size': P.IMG_SIZE, 'none_from': sorted(none_words), 'subjects': subjects},
    }, args.out)
    print(f'Saved {args.out}')


if __name__ == '__main__':
    main()
