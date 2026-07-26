"""
Expert C: SupCon + Co-teaching for noise-robust electricity theft detection.

This module implements the key innovations for breaking the F1=0.85 ceiling:

1. Supervised Contrastive Learning (SupCon):
   - Shapes embedding space to separate "mimicking" theft users from normals
   - Traditional CE only learns decision boundaries; SupCon learns representation geometry
   - Based on Liu et al. (2023) — first SupCon for ETD on SGCC

2. Symmetric Cross Entropy (SCE):
   - Noise-robust loss that handles ~3% label noise without cleanlab
   - Based on Wang et al. (2019, ICCV)

3. Co-teaching:
   - Two networks filter noisy labels for each other
   - Based on Han et al. (2018, NeurIPS)

4. GBDT OOF as prior input (V224 mode):
   - Expert A's prediction is fed as auxiliary feature to the classifier

Training: Joint optimization
  L = L_SCE(cls) + lambda * L_SupCon(proj)
  with co-teaching sample selection
"""
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import numpy as np
import torch
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score, recall_score, precision_score, roc_auc_score

from config import SEED, N_FOLDS, OUTPUT_DIR, DEVICE
from utils import seed_everything, best_f1_score
from coteaching import train_coteaching, predict_coteaching


def train_supcon_expert_sgcc(X_seq, flags, oof_proba_a=None,
                              forget_rate=0.15, supcon_weight=0.3,
                              epochs=50, batch_size=64, verbose=True):
    """Train SupCon + Co-teaching Expert C for SGCC.

    Args:
        X_seq: [N, C, T] multi-channel time series
        flags: [N] binary labels
        oof_proba_a: [N] GBDT OOF probabilities (Expert A prior)
        forget_rate: estimated label noise rate
        supcon_weight: weight for SupCon loss
        epochs: max training epochs
        batch_size: training batch size
        verbose: print progress

    Returns:
        oof_proba_c: [N] OOF probabilities from Expert C
    """
    print("=" * 60)
    print("Expert C: SupCon + Co-teaching (SGCC)")
    print("=" * 60)

    seed_everything(SEED)
    n = len(flags)

    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    oof_proba_c = np.zeros(n)

    for fold_idx, (train_idx, val_idx) in enumerate(skf.split(X_seq, flags)):
        print(f"\n  Fold {fold_idx + 1}/{N_FOLDS}")

        torch.cuda.empty_cache()

        X_train = X_seq[train_idx].astype(np.float32)
        y_train = flags[train_idx].astype(np.float32)
        X_val = X_seq[val_idx].astype(np.float32)
        y_val = flags[val_idx]

        model1, model2 = train_coteaching(
            X_train, y_train,
            oof_prior=oof_proba_a[train_idx] if oof_proba_a is not None else None,
            tcn_channels=[32, 32, 32, 16],
            kernel_size=5,
            dropout=0.3,
            proj_dim=64,
            epochs=epochs,
            batch_size=batch_size,
            lr=3e-4,
            supcon_weight=supcon_weight,
            supcon_temp=0.07,
            sce_alpha=1.0,
            sce_beta=0.5,
            forget_rate=forget_rate,
            warmup_epochs=10,
            device=DEVICE,
            seed=SEED + fold_idx,
            verbose=verbose,
        )

        val_probs = predict_coteaching(model1, model2, X_val, device=DEVICE)
        oof_proba_c[val_idx] = val_probs

        f1, th, rec, prec = best_f1_score(y_val, val_probs)
        auc = roc_auc_score(y_val, val_probs)
        print(f"  Fold {fold_idx+1}: F1={f1:.4f} AUC={auc:.4f} "
              f"Rec={rec:.4f} Prec={prec:.4f} th={th:.3f}")

        del model1, model2
        torch.cuda.empty_cache()

    overall_f1, best_th, overall_rec, overall_prec = best_f1_score(flags, oof_proba_c)
    overall_auc = roc_auc_score(flags, oof_proba_c)
    print(f"\n[Expert C SGCC] Overall: F1={overall_f1:.4f} AUC={overall_auc:.4f} "
          f"Rec={overall_rec:.4f} Prec={overall_prec:.4f} th={best_th:.3f}")

    np.savez_compressed(
        os.path.join(OUTPUT_DIR, 'sgcc_expert_c.npz'),
        oof_proba=oof_proba_c,
        flags=flags,
    )

    return oof_proba_c


if __name__ == '__main__':
    print("Run through pipeline.py with --supcon flag")
