"""Try a neural-network meta-learner on top of existing OOFs.

NN can learn non-linear interactions and sample-dependent gating.
"""
import os, sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score, recall_score, precision_score, roc_auc_score, confusion_matrix

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from config import OUTPUT_DIR, SEED, N_FOLDS
from src.utils.utils import seed_everything

seed_everything(SEED)


def load_top_oofs(y, n_top=15):
    """Load top OOFs by individual F1."""
    from src.training.meta_learner import _load_internal_oofs, _load_external_oofs
    all_oofs = {}
    all_oofs.update(_load_internal_oofs(y))
    all_oofs.update(_load_external_oofs(y))

    scores = {}
    for name, oof in all_oofs.items():
        best = 0
        for th in np.arange(0.05, 0.95, 0.01):
            pred = (oof > th).astype(int)
            if pred.sum() == 0: continue
            f = f1_score(y, pred, zero_division=0)
            if f > best: best = f
        scores[name] = best

    top = sorted(scores, key=scores.get, reverse=True)[:n_top]
    print('Top OOFs selected:')
    for name in top:
        print(f'  {name:40s} F1={scores[name]:.4f}')
    P = np.column_stack([all_oofs[n] for n in top])
    return P, top


class NNMeta(nn.Module):
    def __init__(self, n_features):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_features, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


class F1Loss(nn.Module):
    """Soft F1 loss."""
    def __init__(self, epsilon=1e-7):
        super().__init__()
        self.epsilon = epsilon

    def forward(self, logits, targets):
        probs = torch.sigmoid(logits)
        tp = (probs * targets).sum()
        fp = (probs * (1 - targets)).sum()
        fn = ((1 - probs) * targets).sum()
        f1 = (2 * tp + self.epsilon) / (2 * tp + fp + fn + self.epsilon)
        return 1 - f1


def train_nn_meta(P, y, n_folds=5, epochs=100, lr=1e-3, device='cuda'):
    n, d = P.shape
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=SEED)
    oof = np.zeros(n)

    for fi, (ti, vi) in enumerate(skf.split(P, y)):
        print(f'\nFold {fi+1}')
        X_t = torch.FloatTensor(P[ti]).to(device)
        y_t = torch.FloatTensor(y[ti]).to(device)
        X_v = torch.FloatTensor(P[vi]).to(device)
        y_v = torch.FloatTensor(y[vi]).to(device)

        model = NNMeta(d).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
        criterion = F1Loss()
        pos_weight = (y[ti] == 0).sum() / max((y[ti] == 1).sum(), 1)
        bce = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight], device=device))

        best_f1 = 0
        best_state = None
        patience = 0

        for epoch in range(epochs):
            model.train()
            optimizer.zero_grad()
            logits = model(X_t)
            loss = 0.5 * criterion(logits, y_t) + 0.5 * bce(logits, y_t)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            model.eval()
            with torch.no_grad():
                v_logits = model(X_v)
                v_probs = torch.sigmoid(v_logits).cpu().numpy()
            best_f = 0
            for th in np.arange(0.1, 0.9, 0.01):
                pred = (v_probs > th).astype(int)
                if pred.sum() == 0: continue
                f = f1_score(y[vi], pred, zero_division=0)
                if f > best_f: best_f = f

            if best_f > best_f1:
                best_f1 = best_f
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                patience = 0
            else:
                patience += 1

            if epoch % 10 == 0:
                print(f'  Epoch {epoch+1}: val_best_F1={best_f1:.4f}')
            if patience >= 20:
                break

        model.load_state_dict(best_state)
        model.eval()
        with torch.no_grad():
            oof[vi] = torch.sigmoid(model(X_v)).cpu().numpy()

    return oof


def main():
    y = np.load(os.path.join(OUTPUT_DIR, 'sgcc_preprocessed.npz'))['flags']
    P, names = load_top_oofs(y, n_top=15)
    print(f'\nOOF matrix: {P.shape}')

    oof = train_nn_meta(P, y, n_folds=N_FOLDS, epochs=100, lr=1e-3, device='cuda')

    best_f1, best_th = 0, 0.5
    for th in np.arange(0.05, 0.95, 0.005):
        pred = (oof > th).astype(int)
        if pred.sum() == 0: continue
        f = f1_score(y, pred, zero_division=0)
        if f > best_f1: best_f1, best_th = f, th
    pred = (oof > best_th).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred).ravel()

    print(f'\nNN Meta Result: F1={best_f1:.4f}, Rec={recall_score(y,pred):.4f}, '
          f'Prec={precision_score(y,pred,zero_division=0):.4f}, AUC={roc_auc_score(y,oof):.4f}, th={best_th:.3f}')
    print(f'TP={tp} FP={fp} FN={fn}')

    np.savez_compressed(os.path.join(OUTPUT_DIR, 'nn_meta_oof.npz'),
                        oof_nn_meta=oof, flags=y, names=np.array(names))


if __name__ == '__main__':
    main()
