"""Train Patch Transformer raw 3ch with PeerJ synthetic anomalies + stronger GBDT prior v2."""
import os
import sys
import time

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

from config import OUTPUT_DIR, SEED, N_FOLDS
from src.utils.utils import seed_everything
from src.models.patch_transformer import train_patch_transformer, predict_patch_transformer
from src.data.synthetic_anomalies import SyntheticAnomalyAugmenter
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score, recall_score, precision_score, roc_auc_score
import numpy as np
import torch

seed_everything(SEED)

LOG_PATH = os.path.join(OUTPUT_DIR, 'patch_transformer_raw_3ch_synthetic.log')
_log_fh = open(LOG_PATH, 'w', buffering=1, encoding='utf-8')
sys.stdout = _log_fh
sys.stderr = _log_fh
print(f'Logging to {LOG_PATH}')

print(f'PyTorch: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'GPU: {torch.cuda.get_device_name(0)}')

pre = np.load(os.path.join(OUTPUT_DIR, 'sgcc_preprocessed_raw_3ch.npz'))
X_seq = pre['X_seq']
flags = pre['flags']
print(f'X_seq shape: {X_seq.shape}, theft rate: {flags.mean()*100:.2f}%')

prior_data = np.load(os.path.join(OUTPUT_DIR, 'stronger_gbdt_prior_v2.npz'))
oof_prior = prior_data['prior']

# Strong config from raw 3ch recall run
CONFIG = {
    'patch_len': 30,
    'stride': 15,
    'd_model': 64,
    'n_layers': 2,
    'n_heads': 4,
    'dropout': 0.2,
    'epochs': 20,
    'batch_size': 128,
    'lr': 3e-4,
}

SYN_KWARGS = dict(
    anomaly_types=['point', 'contextual', 'collective'],
    point_lambda=0.5,
    contextual_lambda=1.0,
    contextual_k=7,
    collective_lambda=0.5,
    n_synthetic=3 * int((flags == 1).sum() * 4 / 5),
)

skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
splits = list(skf.split(X_seq, flags))

oof = np.zeros(len(flags), dtype=np.float32)
completed_folds = []
for fi in range(N_FOLDS):
    fpath = os.path.join(OUTPUT_DIR, f'patch_transformer_raw_3ch_synthetic_sp_fold{fi}.npz')
    if os.path.exists(fpath):
        d = np.load(fpath)
        oof[d['vi']] = d['oof']
        completed_folds.append(fi)
        print(f'Resumed fold {fi+1} from {fpath}')

t0 = time.time()
for fi, (ti, vi) in enumerate(splits):
    if fi in completed_folds:
        print(f'Fold {fi+1}/{N_FOLDS} already done, skipping')
        continue

    print(f'\n=== Fold {fi+1}/{N_FOLDS} ===')
    ft0 = time.time()

    X_train, y_train = X_seq[ti], flags[ti]
    prior_train = oof_prior[ti]

    aug = SyntheticAnomalyAugmenter(seed=SEED + fi, **SYN_KWARGS)
    X_syn, y_syn = aug.fit_transform(X_train, y_train)
    X_train_aug = np.concatenate([X_train, X_syn], axis=0)
    y_train_aug = np.concatenate([y_train, y_syn], axis=0)
    prior_train_aug = np.concatenate([
        prior_train,
        np.full(len(y_syn), prior_train.mean(), dtype=np.float32)
    ])
    print(f'  After synthetic aug: {X_train_aug.shape}, theft rate={y_train_aug.mean()*100:.2f}%')

    model = train_patch_transformer(
        X_train_aug, y_train_aug,
        oof_prior=prior_train_aug,
        device='cuda', seed=SEED + fi, verbose=True,
        **CONFIG
    )
    probs = predict_patch_transformer(model, X_seq[vi], oof_prior[vi], device='cuda')
    oof[vi] = np.nan_to_num(probs, nan=0.5)

    np.savez_compressed(
        os.path.join(OUTPUT_DIR, f'patch_transformer_raw_3ch_synthetic_sp_fold{fi}.npz'),
        oof=oof[vi],
        vi=vi,
        ti=ti,
    )
    torch.save(model.state_dict(), os.path.join(OUTPUT_DIR, f'patch_transformer_raw_3ch_synthetic_sp_fold{fi}.pt'))

    best_f1, best_th = 0, 0.5
    for th in np.arange(0.05, 0.95, 0.005):
        pred = (oof[vi] > th).astype(int)
        if pred.sum() == 0:
            continue
        f = f1_score(flags[vi], pred, zero_division=0)
        if f > best_f1:
            best_f1, best_th = f, th
    pred = (oof[vi] > best_th).astype(int)
    print(f'  Fold {fi+1}: F1={f1_score(flags[vi], pred):.4f}, '
          f'Rec={recall_score(flags[vi], pred):.4f}, '
          f'Prec={precision_score(flags[vi], pred, zero_division=0):.4f}, '
          f'AUC={roc_auc_score(flags[vi], oof[vi]):.4f}, th={best_th:.3f}, '
          f'time={(time.time()-ft0)/60:.1f}min')

    del model
    torch.cuda.empty_cache()

overall_f1, overall_th = 0, 0.5
for th in np.arange(0.05, 0.95, 0.005):
    pred = (oof > th).astype(int)
    if pred.sum() == 0:
        continue
    f = f1_score(flags, pred, zero_division=0)
    if f > overall_f1:
        overall_f1, overall_th = f, th
pred = (oof > overall_th).astype(int)
print(f'\n=== Overall Patch Transformer raw 3ch synthetic anomalies ===')
print(f'F1={f1_score(flags, pred):.4f}, Rec={recall_score(flags, pred):.4f}, '
      f'Prec={precision_score(flags, pred, zero_division=0):.4f}, '
      f'AUC={roc_auc_score(flags, oof):.4f}, th={overall_th:.3f}')

np.savez_compressed(
    os.path.join(OUTPUT_DIR, 'patch_transformer_raw_3ch_synthetic_sp_oof.npz'),
    oof_patch_transformer_raw_3ch_synthetic_sp=oof,
    flags=flags,
)
print(f'Saved to {os.path.join(OUTPUT_DIR, "patch_transformer_raw_3ch_synthetic_sp_oof.npz")}')
print(f'Total time: {(time.time()-t0)/60:.1f} minutes')
