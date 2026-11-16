"""Signer-independent k-fold cross-validation (the number to put in the report).

Signers are split into k groups. Each fold tests on one group, picks the best
epoch on a few other signers, and trains on the rest, so every clip is scored
exactly once by a model that never saw that person.

Usage:
    python -m src.cross_validate --folds 5
"""
import argparse
import json
import os

import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from sklearn.model_selection import GroupKFold

from src.dataset import DEFAULT_CACHE, choose_val_subjects, load_cache, split_by_subject, subject_of
from src.engine import get_device, make_model, predict_samples, train_model
from src.evaluate import TARGET_ACCURACY, plot_confusion_matrix, write_report
from src.train import add_hparam_args, hparams_from_args


def main():
    parser = argparse.ArgumentParser(description='Signer-independent k-fold cross-validation.')
    parser.add_argument('--cache', default=DEFAULT_CACHE)
    parser.add_argument('--folds', type=int, default=5)
    parser.add_argument('--n-val-subjects', type=int, default=3)
    parser.add_argument('--out-dir', default='eval_results')
    add_hparam_args(parser)
    args = parser.parse_args()

    device = get_device()
    samples, classes, _ = load_cache(args.cache)
    groups = [subject_of(s) for s in samples]
    hp = hparams_from_args(args)

    all_preds, all_targets, folds = [], [], []
    splitter = GroupKFold(n_splits=args.folds)
    for fold, (rest_idx, test_idx) in enumerate(splitter.split(samples, groups=groups), start=1):
        test_subjects = sorted({groups[i] for i in test_idx})
        rest_subjects = sorted({groups[i] for i in rest_idx})
        val_subjects = choose_val_subjects(rest_subjects, args.n_val_subjects, args.seed + fold)
        train_samples, val_samples, test_samples = split_by_subject(samples, test_subjects, val_subjects)
        print(f'\n===== Fold {fold}/{args.folds}: test signers {test_subjects}, val signers {val_subjects} =====')

        model_state, _, best = train_model(train_samples, val_samples, len(classes), hp, device,
                                           seed=args.seed + fold)
        model = make_model(len(classes), hp).to(device)
        model.load_state_dict(model_state)
        preds, targets, _, _ = predict_samples(model, test_samples, device, hp['num_frames'])

        acc = 100.0 * float(np.mean(preds == targets))
        macro_f1 = float(f1_score(targets, preds, labels=list(range(len(classes))), average='macro', zero_division=0))
        folds.append({'fold': fold, 'test_subjects': test_subjects, 'val_subjects': val_subjects,
                      'best_epoch': best['epoch'], 'accuracy_percent': round(acc, 2),
                      'macro_f1': round(macro_f1, 4), 'n_test': len(targets)})
        all_preds.extend(preds)
        all_targets.extend(targets)
        print(f'Fold {fold} accuracy on unseen signers: {acc:.2f}%  (macro F1 {macro_f1:.4f})')

    accs = np.array([f['accuracy_percent'] for f in folds])
    f1s = np.array([f['macro_f1'] for f in folds])
    labels = list(range(len(classes)))
    all_preds, all_targets = np.array(all_preds), np.array(all_targets)
    pooled_acc = 100.0 * float(np.mean(all_preds == all_targets))

    os.makedirs(args.out_dir, exist_ok=True)
    report_str = classification_report(all_targets, all_preds, labels=labels, target_names=classes,
                                       digits=4, zero_division=0)
    write_report(os.path.join(args.out_dir, 'cv_classification_report.txt'),
                 f'{args.folds}-FOLD SIGNER-INDEPENDENT CROSS-VALIDATION (out-of-fold predictions)',
                 report_str, len(all_targets), pooled_acc)
    plot_confusion_matrix(confusion_matrix(all_targets, all_preds, labels=labels), classes,
                          os.path.join(args.out_dir, 'cv_confusion_matrix.png'),
                          title=f'{args.folds}-fold signer-independent CV (all clips)')

    summary = {
        'protocol': f'{args.folds}-fold GroupKFold by signer; best epoch chosen on '
                    f'{args.n_val_subjects} other signers per fold',
        'classes': classes,
        'accuracy_mean': round(float(accs.mean()), 2),
        'accuracy_std': round(float(accs.std(ddof=1)) if len(accs) > 1 else 0.0, 2),
        'macro_f1_mean': round(float(f1s.mean()), 4),
        'macro_f1_std': round(float(f1s.std(ddof=1)) if len(f1s) > 1 else 0.0, 4),
        'pooled_accuracy': round(pooled_acc, 2),
        'target_accuracy_threshold': TARGET_ACCURACY,
        'hparams': hp,
        'seed': args.seed,
        'folds': folds,
    }
    with open(os.path.join(args.out_dir, 'cross_validation.json'), 'w') as f:
        json.dump(summary, f, indent=2)

    print('\n' + report_str)
    print(f'Accuracy on unseen signers: {summary["accuracy_mean"]:.2f}% ± {summary["accuracy_std"]:.2f} '
          f'over {args.folds} folds (macro F1 {summary["macro_f1_mean"]:.4f} ± {summary["macro_f1_std"]:.4f})')


if __name__ == '__main__':
    main()
