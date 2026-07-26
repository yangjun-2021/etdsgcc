"""Fast AMST raw 3ch: mixed PeerJ + theft-specific synthetic anomalies + stronger prior v2."""
import os
import sys
import time

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

from config import OUTPUT_DIR, SEED, N_FOLDS
from src.utils.utils import seed_everything
from src.training.amst_trainer import AMSTTrainer
from src.models.amst_net import AMSTNet
from src.data.synthetic_anomalies import SyntheticAnomalyAugmenter
from src.data.synthetic_anomalies_theft import TheftSyntheticAnomalyAugmenter
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score, recall_score, precision_score, roc_auc_score
import numpy as np
import torch

seed_everything(SEED)

print(f'PyTorch: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'GPU: {torch.cuda.get_device_name(0)}')

pre = np.load(os.path.join(OUTPUT_DIR, 'sgcc_preprocessed_raw_3ch.npz'))
X_seq = pre['X_seq']
flags = pre['flags']

prior_data = np.load(os.path.join(OUTPUT_DIR, 'stronger_gbdt_prior_v2.npz'))
oof_prior = prior_data['prior']

# Mixed augmentation: PeerJ + theft-specific, total 2x training-fold positives
N_SYN = int((flags == 1).sum() * 4 / 5) // 2

skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
splits = list(skf.split(X_seq, flags))

oof = np.zeros(len(flags), dtype=np.float32)
completed_folds = []
for fi in range(N_FOLDS):
    fpath = os.path.join(OUTPUT_DIR, f'amst_3ch_synthetic_mixed_ls_fast_fold{fi}.npz')
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

    X_train, y_train = X_seq[ti], flags[ti]
    X_val, y_val = X_seq[vi], flags[vi]
    prior_train = oof_prior[ti]
    prior_val = oof_prior[vi]

    # PeerJ anomalies: 1x
    peerj = SyntheticAnomalyAugmenter(
        seed=SEED + fi, n_synthetic=N_SYN,
        anomaly_types=['point', 'contextual', 'collective'],
        point_lambda=0.5, contextual_lambda=1.0, contextual_k=7, collective_lambda=0.5)
    X_pj, y_pj = peerj.fit_transform(X_train, y_train)

    # Theft anomalies: 1x
    theft = TheftSyntheticAnomalyAugmenter(
        seed=SEED + fi + 100, n_synthetic=N_SYN,
        anomaly_types=['scale_global', 'zero_interval', 'scale_local',
                       'subtract_random', 'subtract_mean', 'reverse_daily'])
    X_th, y_th = theft.fit_transform(X_train, y_train)

    X_train_aug = np.concatenate([X_train, X_pj, X_th], axis=0)
    y_train_aug = np.concatenate([y_train, y_pj, y_th], axis=0)
    prior_train_aug = np.concatenate([
        prior_train,
        np.full(len(y_pj) + len(y_th), prior_train.mean(), dtype=np.float32)
    ])
    print(f'  After mixed aug: {X_train_aug.shape}, theft rate={y_train_aug.mean()*100:.2f}%')

    trainer = AMSTTrainer(
        dataset='sgcc', device='cuda',
        use_diffaug=False, use_supcon=False, use_coteaching=False, use_prior=True,
        d_mamba=64, d_trans=128, d_freq=64, proj_dim=128,
        n_mamba_layers=2, n_trans_layers=2, n_heads=4, dropout=0.2,
        epochs=15, batch_size=128, lr=1e-4, patience=7, recall_weight=10.0, use_amp=True, label_smoothing=0.1,
        use_synthetic_anomalies=False,  # already augmented above
    )

    train_loader = trainer._build_loaders(X_train_aug, y_train_aug, prior=prior_train_aug, shuffle=True)
    val_loader = trainer._build_loaders(X_val, y_val, prior=prior_val, shuffle=False)

    model = AMSTNet(
        in_channels=X_seq.shape[1],
        seq_len=X_seq.shape[2],
        d_mamba=trainer.d_mamba,
        d_trans=trainer.d_trans,
        d_freq=trainer.d_freq,
        proj_dim=trainer.proj_dim,
        n_mamba_layers=trainer.n_mamba_layers,
        n_trans_layers=trainer.n_trans_layers,
        n_heads=trainer.n_heads,
        dropout=trainer.dropout,
        use_freq=True,
        use_supcon=trainer.use_supcon,
        prior_dim=1,
    )
    n_params = sum(p.numel() for p in model.parameters())
    print(f'  Model params: {n_params:,}')

    model = trainer._train_single_network(
        model, train_loader, val_loader, y_val,
        epochs=trainer.epochs, lr=trainer.lr, weight_decay=trainer.weight_decay,
        patience=trainer.patience, fold_idx=fi
    )

    val_proba = trainer._predict_proba(model, val_loader)
    oof[vi] = val_proba

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
          f'AUC={roc_auc_score(flags[vi], oof[vi]):.4f}, th={best_th:.3f}')

    np.savez_compressed(
        os.path.join(OUTPUT_DIR, f'amst_3ch_synthetic_mixed_ls_fast_fold{fi}.npz'),
        oof=val_proba,
        vi=vi,
        ti=ti,
    )
    print(f'  Saved fold {fi+1} OOF, time={(time.time()-t0)/60:.1f}min')
    torch.save(model.state_dict(), os.path.join(OUTPUT_DIR, f'amst_3ch_synthetic_mixed_ls_fast_fold{fi}.pt'))
    del model
    torch.cuda.empty_cache()

best_f1, best_th = 0, 0.5
for th in np.arange(0.05, 0.95, 0.005):
    pred = (oof > th).astype(int)
    if pred.sum() == 0:
        continue
    f = f1_score(flags, pred, zero_division=0)
    if f > best_f1:
        best_f1, best_th = f, th
pred = (oof > best_th).astype(int)
print(f'\n=== Overall AMST 3ch mixed synthetic anomalies ===')
print(f'F1={f1_score(flags, pred):.4f}, Rec={recall_score(flags, pred):.4f}, '
      f'Prec={precision_score(flags, pred, zero_division=0):.4f}, '
      f'AUC={roc_auc_score(flags, oof):.4f}, th={best_th:.3f}')

np.savez_compressed(
    os.path.join(OUTPUT_DIR, 'amst_3ch_synthetic_mixed_ls_fast_oof.npz'),
    oof_amst_3ch_synthetic_mixed_ls_fast=oof,
    flags=flags,
)
print(f'Saved to {os.path.join(OUTPUT_DIR, "amst_3ch_synthetic_mixed_ls_fast_oof.npz")}')
