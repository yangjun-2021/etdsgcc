import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
import sys
sys.path.insert(0, r'D:\Project\ThiefElectricity')

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score, recall_score, precision_score, roc_auc_score
import warnings
warnings.filterwarnings('ignore')

SEED = 42
N_FOLDS = 5
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), 'output')

from dl_data import load_raw_data, prepare_sequences, prepare_aggregates, find_oof, best_f1, evaluate

print(f"Device: {DEVICE}")
print("Loading raw data...")
X_raw, y = load_raw_data()
print(f"  X_raw: {X_raw.shape}, y: {y.shape}")

print("Loading V225 OOF (Expert A, F1=0.8457)...")
oof_v225 = find_oof('v225_results_20')
evaluate(y, oof_v225, prefix='  V225: ')

print("Preparing 3-channel sequences (log1p, missing_mask, zero_mask)...")
X_seq = prepare_sequences(X_raw)
print(f"  X_seq: {X_seq.shape}")

print("Preparing aggregate features...")
X_agg = prepare_aggregates(X_raw, oof_v225)
print(f"  X_agg: {X_agg.shape}")

class SimpleTCN(nn.Module):
    def __init__(self, in_channels=3, hidden_dim=32, dropout=0.3, prior_dim=1):
        super().__init__()
        self.tcn = nn.Sequential(
            nn.Conv1d(in_channels, hidden_dim, 7, padding=3), nn.BatchNorm1d(hidden_dim), nn.GELU(),
            nn.Conv1d(hidden_dim, hidden_dim, 7, padding=6, dilation=2), nn.BatchNorm1d(hidden_dim), nn.GELU(),
            nn.Conv1d(hidden_dim, hidden_dim, 7, padding=12, dilation=4), nn.BatchNorm1d(hidden_dim), nn.GELU(),
            nn.Conv1d(hidden_dim, hidden_dim, 7, padding=24, dilation=8), nn.BatchNorm1d(hidden_dim), nn.GELU(),
            nn.Conv1d(hidden_dim, hidden_dim, 7, padding=48, dilation=16), nn.BatchNorm1d(hidden_dim), nn.GELU(),
        )
        self.head = nn.Sequential(
            nn.Linear(hidden_dim + prior_dim, 32), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(32, 1),
        )
    
    def forward(self, x, prior=None):
        x = self.tcn(x)
        x = torch.mean(x, dim=2)
        if prior is not None:
            x = torch.cat([x, prior.reshape(-1, 1)], dim=1)
        return self.head(x).squeeze(-1)


class FocalLoss(nn.Module):
    def __init__(self, alpha=0.75, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
    
    def forward(self, logits, targets):
        probs = torch.sigmoid(logits).clamp(1e-6, 1-1e-6)
        loss_pos = -self.alpha * (1-probs)**self.gamma * torch.log(probs) * targets
        loss_neg = -(1-self.alpha) * probs**self.gamma * torch.log(1-probs) * (1-targets)
        return (loss_pos + loss_neg).mean()


def predict_batched(model, X_np, prior_np, bs=512):
    model.eval()
    probs = []
    with torch.no_grad():
        for i in range(0, len(X_np), bs):
            xb = torch.FloatTensor(X_np[i:i+bs]).to(DEVICE)
            pb = torch.FloatTensor(prior_np[i:i+bs]).to(DEVICE) if prior_np is not None else None
            out = model(xb, pb)
            probs.append(torch.sigmoid(out).cpu().numpy())
    return np.concatenate(probs)


print("\n" + "="*60)
print("Expert B: TCN Training (5-fold CV)")
print("="*60)

skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
oof_b = np.zeros(len(y))
fold_metrics = []

for fold_idx, (ti, vi) in enumerate(skf.split(np.zeros(len(y)), y)):
    print(f"\n  Fold {fold_idx+1}/{N_FOLDS} ({len(ti)} train, {len(vi)} val)")
    
    Xt, Xv = X_seq[ti], X_seq[vi]
    pt, pv = oof_v225[ti], oof_v225[vi]
    yt, yv = y[ti], y[vi]
    
    model = SimpleTCN(in_channels=3, hidden_dim=32, dropout=0.3).to(DEVICE)
    criterion = FocalLoss(alpha=0.75, gamma=2.0)
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=30, eta_min=1e-6)
    
    ds = TensorDataset(torch.FloatTensor(Xt), torch.FloatTensor(pt), torch.FloatTensor(yt))
    dl = DataLoader(ds, batch_size=64, shuffle=True, drop_last=True)
    
    best_f1_fold = 0
    best_state = None
    patience = 0
    
    for epoch in range(30):
        model.train()
        for bx, bp, by in dl:
            bx, bp, by = bx.to(DEVICE), bp.to(DEVICE), by.to(DEVICE)
            optimizer.zero_grad()
            loss = criterion(model(bx, bp), by)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        scheduler.step()
        
        vp = predict_batched(model, Xv, pv)
        
        best_f1_val = 0
        for th in np.arange(0.2, 0.8, 0.01):
            pred = (vp > th).astype(int)
            if pred.sum() == 0: continue
            f1 = f1_score(yv, pred)
            if f1 > best_f1_val: best_f1_val = f1
        
        if best_f1_val > best_f1_fold:
            best_f1_fold = best_f1_val
            best_state = model.state_dict().copy()
            patience = 0
        else:
            patience += 1
        
        if epoch % 5 == 0 or epoch == 29:
            print(f"    Epoch {epoch+1}: best_f1={best_f1_val:.4f}")
        
        if patience >= 6:
            print(f"    Early stop at epoch {epoch+1}")
            break
    
    model.load_state_dict(best_state)
    oof_b[vi] = predict_batched(model, Xv, pv)
    fold_metrics.append(best_f1_fold)

print(f"\n  TCN OOF complete. Best val F1 per fold: {[f'{m:.4f}' for m in fold_metrics]}")

print("\n" + "="*60)
print("Expert A (V225 GBDT) Evaluation")
print("="*60)
res_a = evaluate(y, oof_v225, prefix='  Expert A: ')

print("\n" + "="*60)
print("Expert B (TCN+Prior) Evaluation")
print("="*60)
res_b = evaluate(y, oof_b, prefix='  Expert B: ')

print("\n" + "="*60)
print("Simple Ensemble (Avg A+B)")
print("="*60)
oof_avg = (oof_v225 + oof_b) / 2
res_avg = evaluate(y, oof_avg, prefix='  AVG(A+B): ')

print("\n" + "="*60)
print("Meta-Learner: XGBoost Stacking")
print("="*60)

X_meta = np.column_stack([X_agg, oof_v225.reshape(-1,1), oof_b.reshape(-1,1)])
oof_meta = np.zeros(len(y))

import xgboost as xgb
for fold_idx, (ti, vi) in enumerate(skf.split(np.zeros(len(y)), y)):
    Xt, Xv = X_meta[ti], X_meta[vi]
    yt, yv = y[ti], y[vi]
    
    pw = (yt==0).sum() / max((yt==1).sum(),1)
    model = xgb.XGBClassifier(
        n_estimators=200, max_depth=3, learning_rate=0.05,
        scale_pos_weight=pw, tree_method='hist', random_state=SEED, verbosity=0
    )
    model.fit(Xt, yt)
    oof_meta[vi] = model.predict_proba(Xv)[:,1]

res_meta = evaluate(y, oof_meta, prefix='  Meta: ')

print("\n" + "="*60)
print("Component Comparison")
print("="*60)
print(f"{'Component':<20s} {'F1':>8s} {'Recall':>8s} {'Precision':>8s} {'AUC':>8s}")
print("-" * 55)
for name, (f1, rec, pre, auc) in [
    ('Expert A (V225)', (res_a['f1'], res_a['rec'], res_a['pre'], res_a['auc'])),
    ('Expert B (TCN)', (res_b['f1'], res_b['rec'], res_b['pre'], res_b['auc'])),
    ('Average A+B', (res_avg['f1'], res_avg['rec'], res_avg['pre'], res_avg['auc'])),
    ('Meta Stacking', (res_meta['f1'], res_meta['rec'], res_meta['pre'], res_meta['auc'])),
]:
    print(f"{name:<20s} {f1:8.4f} {rec:8.4f} {pre:8.4f} {auc:8.4f}")

np.savez_compressed(os.path.join(OUTPUT_DIR, 'sgcc_final_oof.npz'),
                    oof_v225=oof_v225, oof_b=oof_b, oof_meta=oof_meta, y=y)
print(f"\nResults saved to {OUTPUT_DIR}/sgcc_final_oof.npz")