"""Score a trained model on the signers it never saw (the held-out test split).

Usage:
    python -m src.evaluate
    python -m src.evaluate --split val
"""
import argparse
import json
import os
from collections import defaultdict

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix

from src.dataset import DEFAULT_CACHE, load_cache, samples_for_subjects, subject_of
from src.engine import predict_samples
from src.inference import load_model

TARGET_ACCURACY = 85.0


def plot_confusion_matrix(cm, classes, save_path, title='Confusion Matrix - Indian Sign Language (ISL)'):
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)
    ax.set(xticks=np.arange(cm.shape[1]), yticks=np.arange(cm.shape[0]),
           xticklabels=classes, yticklabels=classes, title=title,
           ylabel='True Gesture', xlabel='Predicted Gesture')
    plt.setp(ax.get_xticklabels(), rotation=30, ha='right', rotation_mode='anchor')

    thresh = cm.max() / 2.0 if cm.size else 0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], 'd'), ha='center', va='center',
                    color='white' if cm[i, j] > thresh else 'black', fontweight='bold')
    fig.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved {save_path}')


def plot_training_history(history_path, save_path):
    with open(history_path) as f:
        history = json.load(f)
    epochs = range(1, len(history['train_loss']) + 1)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    ax1.plot(epochs, history['train_loss'], 'b-o', markersize=4, label='Train Loss')
    ax1.plot(epochs, history['val_loss'], 'r-s', markersize=4, label='Val Loss')
    ax1.set(title='Cross-Entropy Loss Curve', xlabel='Epoch', ylabel='Loss')
    ax1.grid(True, linestyle='--', alpha=0.6)
    ax1.legend()

    ax2.plot(epochs, history['train_acc'], 'b-o', markersize=4, label='Train Accuracy (%)')
    ax2.plot(epochs, history['val_acc'], 'r-s', markersize=4, label='Val Accuracy (%)')
    ax2.axhline(y=TARGET_ACCURACY, color='green', linestyle='--', label=f'Target Threshold ({TARGET_ACCURACY:.0f}%)')
    ax2.set(title='Model Accuracy Curve (validation signers)', xlabel='Epoch', ylabel='Accuracy (%)')
    ax2.grid(True, linestyle='--', alpha=0.6)
    ax2.legend()

    fig.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved {save_path}')


def per_subject_accuracy(preds, targets, names):
    hits = defaultdict(list)
    for p, t, n in zip(preds, targets, names):
        hits[subject_of({'video_name': n})].append(p == t)
    return {s: round(100.0 * float(np.mean(v)), 2) for s, v in sorted(hits.items())}


def write_report(path, title, report_str, n, accuracy):
    with open(path, 'w') as f:
        f.write(f'================ {title} ================\n')
        f.write(report_str)
        f.write(f'\nTotal Evaluation Samples: {n}\n')
        f.write(f'Overall Accuracy: {accuracy:.2f}%\n')


def evaluate_model(model_path, cache_path, split, out_dir):
    lm = load_model(model_path)
    split_info = lm.meta.get('split')
    if not split_info:
        raise SystemExit(
            f'{model_path} has no signer split recorded. It was trained with the old random '
            'clip-level split, where the same signers appear in training and validation, so it '
            'cannot be scored fairly. Retrain with: python -m src.train')

    subjects = split_info[f'{split}_subjects']
    samples, classes, _ = load_cache(cache_path)
    if classes != lm.classes:
        raise SystemExit(f'Cache classes {classes} do not match the model classes {lm.classes}.')
    eval_samples = samples_for_subjects(samples, subjects)
    print(f'Evaluating on {len(eval_samples)} clips from {split} signers {subjects}')

    preds, targets, _, names = predict_samples(lm.model, eval_samples, lm.device, lm.num_frames)
    labels = list(range(len(classes)))
    report_dict = classification_report(targets, preds, labels=labels, target_names=classes,
                                        output_dict=True, zero_division=0)
    report_str = classification_report(targets, preds, labels=labels, target_names=classes,
                                       digits=4, zero_division=0)
    accuracy = 100.0 * report_dict['accuracy']
    print(report_str)

    os.makedirs(out_dir, exist_ok=True)
    write_report(os.path.join(out_dir, 'classification_report.txt'),
                 f'ISL RECOGNITION REPORT ({split.upper()} SIGNERS {", ".join(subjects)})',
                 report_str, len(targets), accuracy)
    plot_confusion_matrix(confusion_matrix(targets, preds, labels=labels), classes,
                          os.path.join(out_dir, 'confusion_matrix.png'),
                          title=f'Confusion Matrix - unseen {split} signers')

    history_path = os.path.join(out_dir, 'history.json')
    if os.path.exists(history_path):
        plot_training_history(history_path, os.path.join(out_dir, 'training_curves.png'))

    summary = {
        'model': 'Spatio-Temporal CNN + BiGRU + Temporal Attention',
        'protocol': 'signer-independent: test signers never used for training or model selection',
        'classes': classes,
        'split': split,
        'eval_subjects': subjects,
        'train_subjects': split_info.get('train_subjects', []),
        'val_subjects': split_info.get('val_subjects', []),
        'accuracy_percent': round(accuracy, 2),
        'macro_avg_f1': round(report_dict['macro avg']['f1-score'], 4),
        'per_subject_accuracy': per_subject_accuracy(preds, targets, names),
        'total_samples': len(targets),
        'selected_epoch': lm.meta.get('epoch'),
        'target_accuracy_threshold': TARGET_ACCURACY,
        'threshold_satisfied': bool(accuracy >= TARGET_ACCURACY),
    }
    with open(os.path.join(out_dir, 'summary.json'), 'w') as f:
        json.dump(summary, f, indent=2)

    print(f'{split.capitalize()} accuracy on unseen signers: {accuracy:.2f}% (target >= {TARGET_ACCURACY:.0f}%)')
    return summary


def main():
    parser = argparse.ArgumentParser(description='Evaluate the ISL model on held-out signers.')
    parser.add_argument('--model', default=os.path.join('models', 'best_isl_model.pth'))
    parser.add_argument('--cache', default=DEFAULT_CACHE)
    parser.add_argument('--split', choices=['test', 'val'], default='test')
    parser.add_argument('--out-dir', default='eval_results')
    args = parser.parse_args()
    evaluate_model(args.model, args.cache, args.split, args.out_dir)


if __name__ == '__main__':
    main()
