"""Screen PatchTransformer on raw 3ch with recall-oriented focal loss."""
import os
import sys
import time

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

from config import OUTPUT_DIR, SEED
from src.utils.utils import seed_everything
from src.models.patch_transformer import PatchTransformerClassifier
from src.models.models import RecallOrientedFocalLoss
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, recall_score, precision_score, roc_auc_score
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

seed_everything(SEED)

pre = np.load(os.path.join(OUTPUT_DIR, 'sgcc_preprocessed_raw_3ch.npz'))
X_seq = pre['X_seq']
flags = pre['flags']

prior_data = np.load(os.path.join(OUTPUT_DIR, 'strong_gbdt_prior.npz'))
oof_prior = prior_data['prior']

train_idx, val_idx = train_test_split(
    np.arange(len(flags)), test_size=0.2, random_state=SEED, stratify=flags)

X_t = torch.FloatTensor(X_seq[train_idx])
y_t = torch.FloatTensor(flags[train_idx])
p_t = torch.FloatTensor(oof_prior[train_idx])
train_ds = TensorDataset(X_t, y_t, p_t)
train_loader = DataLoader(train_ds, batch_size=128, shuffle=True, drop_last=False, num_workers=0)

X_v = torch.FloatTensor(X_seq[val_idx])
y_v = torch.FloatTensor(flags[val_idx])
p_v = torch.FloatTensor(oof_prior[val_idx])
val_ds = TensorDataset(X_v, y_v, p_v)
val_loader = DataLoader(val_ds, batch_size=256, shuffle=False, num_workers=0)

model = PatchTransformerClassifier(
    in_channels=X_seq.shape[1], seq_len=X_seq.shape[2],
    patch_len=30, stride=15, d_model=64, n_layers=2, n_heads=4,
    dropout=0.2, use_prior=True,
).to('cuda')
print(f'Model params: {sum(p.numel() for p in model.parameters()):,}', flush=True)

pos_weight = (flags[train_idx]==0).sum() / max((flags[train_idx]==1).sum(), 1)
criterion = RecallOrientedFocalLoss(alpha=0.75, gamma=2.0, recall_weight=5.0).to('cuda')
optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=20, eta_min=1e-6)
use_amp = True
scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

best_f1 = 0
best_state = None
for epoch in range(20):
    model.train()
    for bx, by, bp in train_loader:
        bx, by, bp = bx.to('cuda'), by.to('cuda'), bp.to('cuda')
        optimizer.zero_grad()
        with torch.cuda.amp.autocast(enabled=use_amp):
            logits = model(bx, bp)
            loss = criterion(logits, by, pos_weight=torch.tensor([pos_weight], device='cuda'))
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()
    scheduler.step()

    model.eval()
    probs = []
    with torch.no_grad():
        for bx, by, bp in val_loader:
            bx, bp = bx.to('cuda'), bp.to('cuda')
            with torch.cuda.amp.autocast(enabled=use_amp):
                logits = model(bx, bp)
            probs.append(torch.sigmoid(logits).cpu().numpy())
    probs = np.concatenate(probs)
    bf1, bth = 0, 0.5
    for th in np.arange(0.05, 0.95, 0.005):
        pred = (probs > th).astype(int)
        if pred.sum() == 0: continue
        f = f1_score(flags[val_idx], pred, zero_division=0)
        if f > bf1: bf1, bth = f, th
    if bf1 > best_f1:
        best_f1 = bf1
        best_state = {k:v.cpu().clone() for k,v in model.state_dict().items()}
    if (epoch+1) % 5 == 0:
        print(f'  Epoch {epoch+1}: val F1={bf1:.4f} (best={best_f1:.4f})', flush=True)

model.load_state_dict(best_state)
model.eval()
probs = []
with torch.no_grad():
    for bx, by, bp in val_loader:
        bx, bp = bx.to('cuda'), bp.to('cuda')
        with torch.cuda.amp.autocast(enabled=use_amp):
            logits = model(bx, bp)
        probs.append(torch.sigmoid(logits).cpu().numpy())
probs = np.concatenate(probs)
best_f1, best_th = 0, 0.5
for th in np.arange(0.05, 0.95, 0.005):
    pred = (probs > th).astype(int)
    if pred.sum() == 0: continue
    f = f1_score(flags[val_idx], pred, zero_division=0)
    if f > best_f1: best_f1, best_th = f, th
pred = (probs > best_th).astype(int)
print(f'Val: F1={f1_score(flags[val_idx], pred):.4f}, Rec={recall_score(flags[val_idx], pred):.4f}, '
      f'Prec={precision_score(flags[val_idx], pred, zero_division=0):.4f}, '
      f'AUC={roc_auc_score(flags[val_idx], probs):.4f}, th={best_th:.3f}', flush=True)
