"""Train the model with a signer-independent split.

Signers are split into three disjoint groups:
  test  - never seen during training or model selection (default 022-026)
  val   - used only to pick the best epoch
  train - everything else

Usage:
    python -m src.train
    python -m src.train --test-subjects 001 002 003 004 005 --epochs 40
"""
import argparse
import json
import os

import torch

from src.dataset import DEFAULT_CACHE, all_subjects, choose_val_subjects, clip_seconds, load_cache, split_by_subject
from src.engine import DEFAULT_HPARAMS, get_device, train_model

DEFAULT_TEST_SUBJECTS = ['022', '023', '024', '025', '026']


def add_hparam_args(parser):
    parser.add_argument('--epochs', type=int, default=DEFAULT_HPARAMS['epochs'])
    parser.add_argument('--batch-size', type=int, default=DEFAULT_HPARAMS['batch_size'])
    parser.add_argument('--lr', type=float, default=DEFAULT_HPARAMS['lr'])
    parser.add_argument('--weight-decay', type=float, default=DEFAULT_HPARAMS['weight_decay'])
    parser.add_argument('--seed', type=int, default=42)


def hparams_from_args(args):
    hp = dict(DEFAULT_HPARAMS)
    hp.update(epochs=args.epochs, batch_size=args.batch_size, lr=args.lr, weight_decay=args.weight_decay)
    return hp


def save_checkpoint(path, state_dict, classes, hp, best, split, window_seconds, seed):
    os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
    torch.save({
        'model_state_dict': state_dict,
        'classes': list(classes),
        'num_frames': hp['num_frames'],
        'feature_dim': hp['feature_dim'],
        'hidden_dim': hp['hidden_dim'],
        'window_seconds': float(window_seconds),
        'epoch': best['epoch'],
        'val_acc': float(best['val_acc']),
        'val_loss': float(best['val_loss']),
        'selection_metric': 'val_loss',
        'split': {k: list(v) for k, v in split.items()},
        'hparams': hp,
        'seed': seed,
    }, path)


def main():
    parser = argparse.ArgumentParser(description='Train the ISL gesture recogniser.')
    parser.add_argument('--cache', default=DEFAULT_CACHE)
    parser.add_argument('--out', default=os.path.join('models', 'best_isl_model.pth'))
    parser.add_argument('--history', default=os.path.join('eval_results', 'history.json'))
    parser.add_argument('--test-subjects', nargs='+', default=DEFAULT_TEST_SUBJECTS,
                        help='Signers held out for the final test. Never used for training or model selection.')
    parser.add_argument('--n-val-subjects', type=int, default=4,
                        help='Number of remaining signers used to choose the best epoch.')
    add_hparam_args(parser)
    args = parser.parse_args()

    device = get_device()
    print(f'Training on device: {device}')

    samples, classes, _ = load_cache(args.cache)
    subjects = all_subjects(samples)
    remaining = [s for s in subjects if s not in set(args.test_subjects)]
    val_subjects = choose_val_subjects(remaining, args.n_val_subjects, args.seed)
    train_samples, val_samples, test_samples = split_by_subject(samples, args.test_subjects, val_subjects)
    train_subjects = sorted(set(remaining) - set(val_subjects))

    print(f'Classes: {classes}')
    print(f'Train: {len(train_samples)} clips / {len(train_subjects)} signers')
    print(f'Val:   {len(val_samples)} clips / signers {val_subjects}')
    print(f'Test:  {len(test_samples)} clips / signers {sorted(args.test_subjects)} (held out)')

    hp = hparams_from_args(args)
    best_state, history, best = train_model(train_samples, val_samples, len(classes), hp, device, seed=args.seed)

    split = {'train_subjects': train_subjects, 'val_subjects': val_subjects,
             'test_subjects': sorted(args.test_subjects)}
    save_checkpoint(args.out, best_state, classes, hp, best, split, clip_seconds(train_samples), args.seed)

    os.makedirs(os.path.dirname(args.history) or '.', exist_ok=True)
    with open(args.history, 'w') as f:
        json.dump(history, f, indent=2)

    print(f'Saved {args.out} (epoch {best["epoch"]}, val loss {best["val_loss"]:.4f}, val acc {best["val_acc"]:.1f}%)')
    print('Next: python -m src.evaluate   (scores the held-out test signers)')


if __name__ == '__main__':
    main()
