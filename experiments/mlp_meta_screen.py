"""Screen a small MLP meta-learner on the OOF matrix."""
import os
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score, recall_score, precision_score, roc_auc_score

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

from config import OUTPUT_DIR, SEED, N_FOLDS
from src.utils.utils import seed_everything
from src.training.meta_learner import MetaLearner

seed_everything(SEED)

# Load OOF matrix produced by MegaMetaLearner
pre = np.load(os.path.join(OUTPUT_DIR, 'sgcc_preprocessed.npz'))
flags = pre['flags']
impute_mask = pre['impute_mask']

a = np.load(os.path.join(OUTPUT_DIR, 'sgcc_expert_a.npz'))
b = np.load(os.path.join(OUTPUT_DIR, 'sgcc_expert_b.npz'))

learner = MetaLearner(dataset='sgcc')
# monkey-patch to capture P after building
P_global = {}
original_train = learner.train

def _train_capture(*args, **kwargs):
    raise RuntimeError('capture only')

# Build OOF matrix same way as meta_learner
def build_P():
    from src.training.meta_learner import _load_internal_oofs, _load_external_oofs
    all_oofs = {}
    for name, oof in _load_internal_oofs(flags).items():
        all_oofs[name] = oof
    for name, oof in _load_external_oofs(flags).items():
        all_oofs[name] = oof
    all_oofs['Expert-A(GBDT)'] = a['oof_proba']
    all_oofs['Expert-B(TCN)'] = b['oof_proba']
    # correlation pruning
    names = sorted(all_oofs.keys())
    P_tmp = np.column_stack([all_oofs[nm] for nm in names])
    P_tmp = np.nan_to_num(P_tmp, nan=0.5, posinf=1.0, neginf=0.0)
    corrs = np.corrcoef(P_tmp.T)
    kept = []
    for i, nm in enumerate(names):
        drop = False
        for j in kept:
            if abs(corrs[i, names.index(j)]) > 0.999:
                drop = True; break
        if not drop:
            kept.append(nm)
    P = np.column_stack([all_oofs[nm] for nm in kept])
    P = np.nan_to_num(P, nan=0.5, posinf=1.0, neginf=0.0)
    miss = impute_mask.mean(axis=1).reshape(-1, 1)
    P = np.column_stack([P, miss])
    return P, kept + ['miss_ratio']

P, names = build_P()
print(f'OOF matrix: {P.shape}')

class MLP(nn.Module):
    def __init__(self, in_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.LayerNorm(64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 1),
        )
    def forward(self, x):
        return self.net(x).squeeze(-1)

skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
oof = np.zeros(len(flags))

device = 'cuda' if torch.cuda.is_available() else 'cpu'
for fi, (ti, vi) in enumerate(skf.split(P, flags)):
    print(f'\nFold {fi+1}/{N_FOLDS}')
    X_t = torch.FloatTensor(P[ti])
    y_t = torch.FloatTensor(flags[ti])
    X_v = torch.FloatTensor(P[vi])
    y_v = torch.FloatTensor(flags[vi])
    train_loader = DataLoader(TensorDataset(X_t, y_t), batch_size=512, shuffle=True)
    val_loader = DataLoader(TensorDataset(X_v, y_v), batch_size=1024)

    model = MLP(P.shape[1]).to(device)
    pos_weight = (flags[ti]==0).sum() / max((flags[ti]==1).sum(),1)
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight], device=device))
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50, eta_min=1e-6)

    best_f1 = 0
    best_state = None
    for epoch in range(50):
        model.train()
        for bx, by in train_loader:
            bx, by = bx.to(device), by.to(device)
            optimizer.zero_grad()
            logits = model(bx)
            loss = criterion(logits, by)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        scheduler.step()

        model.eval()
        with torch.no_grad():
            probs = torch.sigmoid(model(X_v.to(device))).cpu().numpy()
        bf1 = 0
        for th in np.arange(0.1,0.9,0.01):
            pred = (probs>th).astype(int)
            if pred.sum()==0: continue
            f = f1_score(flags[vi], pred, zero_division=0)
            if f>bf1: bf1=f
        if bf1 > best_f1:
            best_f1 = bf1
            best_state = {k:v.cpu().clone() for k,v in model.state_dict().items()}
        if (epoch+1)%10==0:
            print(f'  Epoch {epoch+1}: val best F1={best_f1:.4f}')
    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        oof[vi] = torch.sigmoid(model(X_v.to(device))).cpu().numpy()

best_f1, best_th = 0, 0.5
for th in np.arange(0.05,0.95,0.005):
    pred = (oof>th).astype(int)
    if pred.sum()==0: continue
    f = f1_score(flags, pred, zero_division=0)
    if f>best_f1: best_f1,best_th=f,th
pred = (oof>best_th).astype(int)
print(f'\nOverall MLP meta: F1={best_f1:.4f}, Rec={recall_score(flags,pred):.4f}, '
      f'Prec={precision_score(flags,pred,zero_division=0):.4f}, AUC={roc_auc_score(flags,oof):.4f}, th={best_th:.3f}')
