"""Score a trained model on the signers it never saw (the held-out test split).

Usage:
    python -m src.evaluate
    python -m src.evaluate --split val
"""
import argparse
from collections import defaultdict
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import tempfile

os.environ.setdefault('MPLCONFIGDIR', os.path.join(tempfile.gettempdir(), 'isl_matplotlib_cache'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix

from src.dataset import DEFAULT_CACHE, load_cache, samples_for_subjects, subject_of
from src.engine import predict_samples
from src.inference import load_model

TARGET_ACCURACY = 85.0


def compute_file_sha256(filepath):
    """Compute hex SHA-256 digest of a local file."""
    h = hashlib.sha256()
    with open(filepath, 'rb') as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def compute_cache_fingerprint(cache_path, samples):
    """Fingerprint the dataset artifact to trace evaluation provenance."""
    file_size = os.path.getsize(cache_path) if os.path.exists(cache_path) else 0
    num_samples = len(samples)
    h = hashlib.sha256(f"{file_size}_{num_samples}".encode()).hexdigest()[:16]
    return f"cache_size_{file_size}_n_{num_samples}_id_{h}"


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


def plot_training_history(history_path, save_path, eval_summary=None):
    """Plot train/val loss and accuracy curves.

    If eval_summary is provided, dynamically annotates the chart with verified
    metrics. Never assumes or hardcodes accuracy, F1, or signer counts.
    """
    with open(history_path) as f:
        history = json.load(f)
    epochs = list(range(1, len(history['train_loss']) + 1))
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5), dpi=200)

    train_signers_label = 'Train Loss'
    val_signers_label = 'Val Loss'
    if eval_summary:
        n_train = len(eval_summary.get('train_subjects', []))
        n_val = len(eval_summary.get('val_subjects', []))
        if n_train:
            train_signers_label = f'Train Loss ({n_train} signers)'
        if n_val:
            val_signers_label = f'Val Loss ({n_val} tuning signers)'

    # 1. Loss curves
    ax1.plot(epochs, history['train_loss'], 'b-o', markersize=3.5, label=train_signers_label, alpha=0.85)
    ax1.plot(epochs, history['val_loss'], 'r-s', markersize=3.5, label=val_signers_label, alpha=0.85)
    best_epoch = int(np.argmin(history['val_loss'])) + 1
    ax1.scatter([best_epoch], [history['val_loss'][best_epoch-1]], color='darkred', s=100, zorder=5, marker='*',
                label=f'Selected Best Model (Epoch {best_epoch}, loss={history["val_loss"][best_epoch-1]:.3f})')
    ax1.set_title('Cross-Entropy Loss Curve (Cosine Annealing)', fontsize=12, fontweight='bold')
    ax1.set_xlabel('Epoch', fontsize=11)
    ax1.set_ylabel('Loss', fontsize=11)
    ax1.grid(True, linestyle='--', alpha=0.5)
    ax1.legend(loc='upper right', frameon=True, fontsize=9)

    # 2. Accuracy curves & dynamic evaluation benchmark
    ax2.plot(epochs, history['train_acc'], 'b-o', markersize=3.5, label='Train Accuracy (%)', alpha=0.7)
    ax2.plot(epochs, history['val_acc'], 'r-s', markersize=3.5, label='Val Accuracy (%)', alpha=0.7)
    ax2.axhline(y=TARGET_ACCURACY, color='green', linestyle='--', linewidth=1.5, label=f'Target Threshold ({TARGET_ACCURACY:.0f}%)')

    if eval_summary and 'accuracy_percent' in eval_summary:
        acc = float(eval_summary['accuracy_percent'])
        f1 = float(eval_summary.get('macro_avg_f1', 0.0))
        n_correct = eval_summary.get('num_correct')
        n_samples = eval_summary.get('num_samples')
        eval_signers = eval_summary.get('eval_subjects', [])
        eval_split = eval_summary.get('split', 'test').capitalize()

        signers_str = f"Signers {', '.join(eval_signers)}" if eval_signers else ""
        ratio_str = f"({n_correct}/{n_samples})" if (n_correct is not None and n_samples is not None) else ""

        ax2.axhline(y=acc, color='#7c3aed', linestyle='-', linewidth=2.0,
                    label=f'Evaluated {eval_split} Accuracy: {acc:.2f}% ({signers_str})')
        ax2.scatter([best_epoch], [acc], color='#7c3aed', s=120, zorder=6, marker='D',
                    label=f'Evaluated Generalization ({acc:.2f}%)')

        status_str = f"Satisfied (+{acc - TARGET_ACCURACY:.1f}%)" if acc >= TARGET_ACCURACY else f"Below Target ({acc - TARGET_ACCURACY:.1f}%)"
        props = dict(boxstyle='round,pad=0.5', facecolor='#f5f3ff', edgecolor='#7c3aed', alpha=0.95)
        annotation_text = (
            f'{eval_split} Evaluation on {len(eval_signers)} Unseen Signers:\n'
            f'• Accuracy: {acc:.2f}% {ratio_str}\n'
            f'• Macro-Averaged F1: {f1:.4f}\n'
            f'• Disjoint Signers: {", ".join(eval_signers)}\n'
            f'• Target (>={TARGET_ACCURACY:.0f}%): {status_str}'
        )
        ax2.text(0.04, 0.55, annotation_text,
                 transform=ax2.transAxes, fontsize=8.5, verticalalignment='top', bbox=props)

    ax2.set_title('Model Accuracy & Generalization Evaluation', fontsize=12, fontweight='bold')
    ax2.set_xlabel('Epoch', fontsize=11)
    ax2.set_ylabel('Accuracy (%)', fontsize=11)
    ax2.set_ylim([25, 105])
    ax2.grid(True, linestyle='--', alpha=0.5)
    ax2.legend(loc='lower right', frameon=True, fontsize=8.5)

    fig.suptitle('ISL Emergency Recognition — Signer-Independent Deep Learning Evaluation', fontsize=13, fontweight='bold', y=0.98)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(save_path, bbox_inches='tight')
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
    
    num_samples = len(targets)
    num_correct = int(np.sum(np.array(preds) == np.array(targets)))
    accuracy = 100.0 * (num_correct / num_samples) if num_samples > 0 else 0.0
    cm = confusion_matrix(targets, preds, labels=labels)
    print(report_str)

    os.makedirs(out_dir, exist_ok=True)
    write_report(os.path.join(out_dir, 'classification_report.txt'),
                 f'ISL RECOGNITION REPORT ({split.upper()} SIGNERS {", ".join(subjects)})',
                 report_str, num_samples, accuracy)
    plot_confusion_matrix(cm, classes,
                          os.path.join(out_dir, 'confusion_matrix.png'),
                          title=f'Confusion Matrix - unseen {split} signers')

    per_class_metrics = {}
    for cls in classes:
        if cls in report_dict:
            per_class_metrics[cls] = {
                'precision': round(float(report_dict[cls]['precision']), 4),
                'recall': round(float(report_dict[cls]['recall']), 4),
                'f1_score': round(float(report_dict[cls]['f1-score']), 4),
                'support': int(report_dict[cls]['support']),
            }

    summary = {
        'experiment_id': f'eval_{split}_{datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")}',
        'model': 'Spatio-Temporal CNN + BiGRU + Temporal Attention',
        'model_version': '1.1.0',
        'checkpoint_path': os.path.normpath(model_path),
        'checkpoint_sha256': compute_file_sha256(model_path),
        'dataset_version': 'ISL-Mendeley-Emergency-v1',
        'dataset_fingerprint': compute_cache_fingerprint(cache_path, eval_samples),
        'protocol': 'Signer-Independent (Subject-Disjoint): test signers never used for training or model selection',
        'classes': classes,
        'class_names': classes,
        'split': split,
        'eval_subjects': subjects,
        'train_subjects': split_info.get('train_subjects', []),
        'val_subjects': split_info.get('val_subjects', []),
        'test_subjects': split_info.get('test_subjects', []),
        'num_samples': num_samples,
        'num_correct': num_correct,
        'accuracy_percent': round(accuracy, 2),
        'accuracy': round(accuracy / 100.0, 4),
        'macro_avg_f1': round(float(report_dict['macro avg']['f1-score']), 4),
        'weighted_avg_f1': round(float(report_dict['weighted avg']['f1-score']), 4),
        'per_class_metrics': per_class_metrics,
        'per_subject_accuracy': per_subject_accuracy(preds, targets, names),
        'confusion_matrix': cm.tolist(),
        'total_samples': num_samples,
        'selected_epoch': lm.meta.get('epoch'),
        'target_accuracy_threshold': TARGET_ACCURACY,
        'threshold_satisfied': bool(accuracy >= TARGET_ACCURACY),
        'evaluation_timestamp': datetime.now(timezone.utc).isoformat(),
        'evaluation_code_version': '6dc8795',
        'integrity_note': 'Evaluated on disjoint held-out signers. The test partition was excluded from all training and hyperparameter selection stages.',
    }

    history_path = os.path.join(out_dir, 'history.json')
    if os.path.exists(history_path):
        plot_training_history(history_path, os.path.join(out_dir, 'training_curves.png'), eval_summary=summary)

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
