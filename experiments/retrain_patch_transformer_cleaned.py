"""Train recall-oriented PatchTransformer on cleaned labels with 5-fold CV."""
import os
import sys
import time

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

from config import OUTPUT_DIR, SEED, N_FOLDS
from src.utils.utils import seed_everything
from src.models.patch_transformer import PatchTransformerClassifier
from src.models.models import RecallOrientedFocalLoss
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score, recall_score, precision_score, roc_auc_score
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

seed_everything(SEED)

LOG_PATH = os.path.join(OUTPUT_DIR, 'patch_transformer_cleaned.log')
_log_fh = open(LOG_PATH, 'w', buffering=1, encoding='utf-8')
sys.stdout = _log_fh
sys.stderr = _log_fh
print(f'Logging to {LOG_PATH}')

pre = np.load(os.path.join(OUTPUT_DIR, 'sgcc_preprocessed_raw_3ch.npz'))
X_seq = pre['X_seq']
flags = np.load(os.path.join(OUTPUT_DIR, 'cleaned_labels_v1.npz'))['y_clean'].astype(int)
y_orig = np.load(os.path.join(OUTPUT_DIR, 'cleaned_labels_v1.npz'))['y_orig'].astype(int)
print(f'X_seq shape: {X_seq.shape}, clean theft rate: {flags.mean()*100:.2f}%, orig theft rate: {y_orig.mean()*100:.2f}%')

prior_data = np.load(os.path.join(OUTPUT_DIR, 'strong_gbdt_prior.npz'))
oof_prior = prior_data['prior']

EPOCHS = 20
BATCH_SIZE = 128
LR = 3e-4
skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
splits = list(skf.split(X_seq, flags))

oof = np.zeros(len(flags), dtype=np.float32)
completed_folds = []
for fi in range(N_FOLDS):
    fpath = os.path.join(OUTPUT_DIR, f'patch_transformer_cleaned_fold{fi}.npz')
    if os.path.exists(fpath):
        d = np.load(fpath)
        oof[d['vi']] = d['oof']
        completed_folds.append(fi)
        print(f'Resumed fold {fi+1} from {fpath}')

for fi, (ti, vi) in enumerate(splits):
    if fi in completed_folds:
        print(f'Fold {fi+1}/{N_FOLDS} already done, skipping')
        continue

    print(f'\n=== Fold {fi+1}/{N_FOLDS} ===')
    t0 = time.time()
    X_t = torch.FloatTensor(X_seq[ti])
    y_t = torch.FloatTensor(flags[ti])
    p_t = torch.FloatTensor(oof_prior[ti])
    train_loader = DataLoader(TensorDataset(X_t, y_t, p_t), batch_size=BATCH_SIZE, shuffle=True, drop_last=False, num_workers=0)

    X_v = torch.FloatTensor(X_seq[vi])
    y_v = torch.FloatTensor(flags[vi])
    p_v = torch.FloatTensor(oof_prior[vi])
    val_loader = DataLoader(TensorDataset(X_v, y_v, p_v), batch_size=256, shuffle=False, num_workers=0)

    model = PatchTransformerClassifier(
        in_channels=X_seq.shape[1], seq_len=X_seq.shape[2],
        patch_len=30, stride=15, d_model=64, n_layers=2, n_heads=4,
        dropout=0.2, use_prior=True,
    ).to('cuda')
    n_params = sum(p.numel() for p in model.parameters())
    print(f'  Model params: {n_params:,}')

    pos_weight = (flags[ti]==0).sum() / max((flags[ti]==1).sum(), 1)
    criterion = RecallOrientedFocalLoss(alpha=0.75, gamma=2.0, recall_weight=5.0).to('cuda')
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-6)
    use_amp = True
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    best_f1 = 0
    best_state = None
    for epoch in range(EPOCHS):
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
            f = f1_score(flags[vi], pred, zero_division=0)
            if f > bf1: bf1, bth = f, th
        if bf1 > best_f1:
            best_f1 = bf1
            best_state = {k:v.cpu().clone() for k,v in model.state_dict().items()}
        if (epoch+1) % 5 == 0 or epoch == EPOCHS-1:
            print(f'  Epoch {epoch+1}: val F1={bf1:.4f} (best={best_f1:.4f})')

    model.load_state_dict(best_state)
    model.eval()
    probs = []
    with torch.no_grad():
        for bx, by, bp in val_loader:
            bx, bp = bx.to('cuda'), bp.to('cuda')
            with torch.cuda.amp.autocast(enabled=use_amp):
                logits = model(bx, bp)
            probs.append(torch.sigmoid(logits).cpu().numpy())
    oof[vi] = np.concatenate(probs)

    np.savez_compressed(
        os.path.join(OUTPUT_DIR, f'patch_transformer_cleaned_fold{fi}.npz'),
        oof=oof[vi],
        vi=vi,
        ti=ti,
    )
    torch.save(model.state_dict(), os.path.join(OUTPUT_DIR, f'patch_transformer_cleaned_fold{fi}.pt'))

    best_f1, best_th = 0, 0.5
    for th in np.arange(0.05, 0.95, 0.005):
        pred = (oof[vi] > th).astype(int)
        if pred.sum() == 0: continue
        f = f1_score(flags[vi], pred, zero_division=0)
        if f > best_f1: best_f1, best_th = f, th
    pred = (oof[vi] > best_th).astype(int)
    print(f'  Fold {fi+1}: F1={f1_score(flags[vi], pred):.4f}, '
          f'Rec={recall_score(flags[vi], pred):.4f}, '
          f'Prec={precision_score(flags[vi], pred, zero_division=0):.4f}, '
          f'AUC={roc_auc_score(flags[vi], oof[vi]):.4f}, th={best_th:.3f}, time={(time.time()-t0)/60:.1f}min')
    del model
    torch.cuda.empty_cache()

overall_f1, overall_th = 0, 0.5
for th in np.arange(0.05, 0.95, 0.005):
    pred = (oof > th).astype(int)
    if pred.sum() == 0: continue
    f = f1_score(flags, pred, zero_division=0)
    if f > overall_f1: overall_f1, overall_th = f, th
pred = (oof > overall_th).astype(int)
print(f'\n=== Overall PatchTransformer Cleaned (cleaned labels) ===')
print(f'F1={f1_score(flags, pred):.4f}, Rec={recall_score(flags, pred):.4f}, '
      f'Prec={precision_score(flags, pred, zero_division=0):.4f}, '
      f'AUC={roc_auc_score(flags, oof):.4f}, th={overall_th:.3f}')

overall_f1, overall_th = 0, 0.5
for th in np.arange(0.05, 0.95, 0.005):
    pred = (oof > th).astype(int)
    if pred.sum() == 0: continue
    f = f1_score(y_orig, pred, zero_division=0)
    if f > overall_f1: overall_f1, overall_th = f, th
pred = (oof > overall_th).astype(int)
print(f'\n=== Overall PatchTransformer Cleaned (original labels) ===')
print(f'F1={f1_score(y_orig, pred):.4f}, Rec={recall_score(y_orig, pred):.4f}, '
      f'Prec={precision_score(y_orig, pred, zero_division=0):.4f}, '
      f'AUC={roc_auc_score(y_orig, oof):.4f}, th={overall_th:.3f}')

np.savez_compressed(
    os.path.join(OUTPUT_DIR, 'patch_transformer_cleaned_oof.npz'),
    oof_patch_transformer_cleaned=oof,
    y_clean=flags,
    y_orig=y_orig,
)
print(f'Saved to {os.path.join(OUTPUT_DIR, "patch_transformer_cleaned_oof.npz")}')
