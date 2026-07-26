"""Diagnose AMST training on cleaned labels - 1 fold, few epochs, verbose."""
import os
import sys
import time

sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

from config import OUTPUT_DIR, SEED, N_FOLDS
from src.utils.utils import seed_everything
from src.training.amst_trainer import AMSTTrainer
from src.models.amst_net import AMSTNet
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score, recall_score, precision_score, roc_auc_score
import numpy as np
import torch

seed_everything(SEED)

print(f'PyTorch: {torch.__version__}, CUDA: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'GPU: {torch.cuda.get_device_name(0)}')

pre = np.load(os.path.join(OUTPUT_DIR, 'sgcc_preprocessed_3ch.npz'))
X_seq = pre['X_seq']
flags = np.load(os.path.join(OUTPUT_DIR, 'cleaned_labels_v1.npz'))['y_clean'].astype(int)
prior_data = np.load(os.path.join(OUTPUT_DIR, 'strong_gbdt_prior.npz'))
oof_prior = prior_data['prior']

trainer_kwargs = dict(
    dataset='sgcc', use_diffaug=False, use_supcon=False, use_coteaching=False, use_prior=True,
    d_mamba=64, d_trans=128, d_freq=64, proj_dim=128, n_mamba_layers=2, n_trans_layers=2,
    n_heads=4, dropout=0.2, epochs=10, batch_size=64, lr=1e-4, patience=15, recall_weight=10.0,
)

skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
splits = list(skf.split(X_seq, flags))
fi, (ti, vi) = 0, splits[0]

print(f'Fold {fi+1}: train={len(ti)}, val={len(vi)}')
t0 = time.time()
trainer = AMSTTrainer(device='cuda', **trainer_kwargs)
X_train, y_train = X_seq[ti], flags[ti]
X_val, y_val = X_seq[vi], flags[vi]
prior_train = oof_prior[ti]
prior_val = oof_prior[vi]

print('Augmenting...')
X_train_aug, y_train_aug = trainer._augment(X_train, y_train)
prior_train_aug = np.concatenate([
    prior_train,
    np.full(len(y_train_aug) - len(y_train), prior_train.mean(), dtype=np.float32)
])
print(f'Augmented: {X_train_aug.shape}')

print('Building loaders...')
train_loader = trainer._build_loaders(X_train_aug, y_train_aug, prior=prior_train_aug, shuffle=True)
val_loader = trainer._build_loaders(X_val, y_val, prior=prior_val, shuffle=False)

print('Creating model...')
model = AMSTNet(
    in_channels=X_seq.shape[1], seq_len=X_seq.shape[2],
    d_mamba=trainer.d_mamba, d_trans=trainer.d_trans, d_freq=trainer.d_freq, proj_dim=trainer.proj_dim,
    n_mamba_layers=trainer.n_mamba_layers, n_trans_layers=trainer.n_trans_layers, n_heads=trainer.n_heads,
    dropout=trainer.dropout, use_freq=True, use_supcon=trainer.use_supcon,
    prior_dim=1 if trainer.use_prior and oof_prior is not None else 0,
).to('cuda')
print(f'Model params: {sum(p.numel() for p in model.parameters()):,}')

print('Training 3 epochs...')
model = trainer._train_single_network(model, train_loader, val_loader, y_val,
                                      epochs=10, lr=trainer.lr, weight_decay=trainer.weight_decay,
                                      patience=trainer.patience, fold_idx=fi)
print(f'Training done in {(time.time()-t0)/60:.1f}min')

print('Predicting...')
val_proba = trainer._predict_proba(model, val_loader)
best_f1, best_th = 0, 0.5
for th in np.arange(0.05, 0.95, 0.005):
    pred = (val_proba > th).astype(int)
    if pred.sum() == 0: continue
    f = f1_score(y_val, pred, zero_division=0)
    if f > best_f1: best_f1, best_th = f, th
pred = (val_proba > best_th).astype(int)
print(f'Val: F1={f1_score(y_val, pred):.4f}, Rec={recall_score(y_val, pred):.4f}, '
      f'Prec={precision_score(y_val, pred, zero_division=0):.4f}, '
      f'AUC={roc_auc_score(y_val, val_proba):.4f}, th={best_th:.3f}')
print('All done')
