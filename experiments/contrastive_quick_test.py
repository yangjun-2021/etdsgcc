import argparse
import os
import sys

import numpy as np
from sklearn.model_selection import StratifiedKFold

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from config import OUTPUT_DIR, SEED
from src.data.preprocess_sgcc import preprocess_sgcc
from src.training.contrastive_trainer import ContrastiveTrainer
from src.utils.utils import seed_everything


def load_cache():
    data = np.load(os.path.join(OUTPUT_DIR, 'sgcc_preprocessed.npz'))
    return data['X_seq'], data['stat_features'], data['flags']


def sample_subset(X_seq, stat_features, flags, n_samples, n_theft):
    rng = np.random.RandomState(SEED)
    pos_idx = np.where(flags == 1)[0]
    neg_idx = np.where(flags == 0)[0]
    n_theft = min(n_theft, len(pos_idx), n_samples)
    n_normal = min(n_samples - n_theft, len(neg_idx))
    sel = np.concatenate([
        rng.choice(pos_idx, n_theft, replace=False),
        rng.choice(neg_idx, n_normal, replace=False),
    ])
    rng.shuffle(sel)
    return X_seq[sel], stat_features[sel], flags[sel]


def fold_assignments(y, n_folds):
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=SEED)
    folds = np.zeros(len(y), dtype=int)
    for fold, (_, val_idx) in enumerate(skf.split(np.zeros(len(y)), y)):
        folds[val_idx] = fold
    return folds


def main():
    parser = argparse.ArgumentParser(description='Contrastive encoder quick SGCC experiment')
    parser.add_argument('--n-samples', type=int, default=2000)
    parser.add_argument('--n-theft', type=int, default=200)
    parser.add_argument('--folds', type=int, default=3)
    parser.add_argument('--contrastive-epochs', type=int, default=5)
    parser.add_argument('--finetune-epochs', type=int, default=15)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--d-model', type=int, default=64)
    parser.add_argument('--lstm-hidden', type=int, default=32)
    parser.add_argument('--lstm-layers', type=int, default=2)
    parser.add_argument('--seq-target-len', type=int, default=128)
    parser.add_argument('--seq-stride', type=int, default=1)
    parser.add_argument('--load-cache', action='store_true')
    args = parser.parse_args()

    seed_everything(SEED)
    if args.load_cache and os.path.exists(os.path.join(OUTPUT_DIR, 'sgcc_preprocessed.npz')):
        X_seq, stat_features, flags = load_cache()
    else:
        X_seq, stat_features, flags, _ = preprocess_sgcc(use_advanced_features=False)
    X_seq, stat_features, flags = sample_subset(X_seq, stat_features, flags, args.n_samples, args.n_theft)
    if args.seq_stride > 1:
        X_seq = X_seq[:, :, ::args.seq_stride]
    folds = fold_assignments(flags, args.folds)
    trainer = ContrastiveTrainer(
        dataset='sgcc_quick',
        seq_target_len=args.seq_target_len,
        contrastive_epochs=args.contrastive_epochs,
        finetune_epochs=args.finetune_epochs,
        batch_size=args.batch_size,
        d_model=args.d_model,
        lstm_hidden=args.lstm_hidden,
        lstm_layers=args.lstm_layers,
        patience=max(3, args.finetune_epochs // 2),
    )
    trainer.train(X_seq, flags, stat_features=stat_features, fold_assignments=folds)


if __name__ == '__main__':
    main()
