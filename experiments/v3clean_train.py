"""Phase 3: train final experts on v3-cleaned labels (untainted, NO prior feature).

Trains the v3clean_* model family on cleaned_labels_v3.npz['y_clean'] with the
same architectures/configs as the untainted v3 voters. All models are trained
without any cleaned-label-derived prior feature; evaluation is reported on BOTH
v3-cleaned labels (primary protocol) and original labels (secondary protocol).

Usage:
    conda run -n ml python experiments/v3clean_train.py --models gbdt          # CPU, fast
    conda run -n ml python experiments/v3clean_train.py --models patcht,tcn_ct # GPU
    conda run -n ml python experiments/v3clean_train.py --models amst_ct,amst_aug

Requires: GPU for the deep models; cleaned_labels_v3.npz must exist
(run experiments/v3_clean_labels.py first).
"""
import argparse
import os
import sys
import time

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score, recall_score, precision_score, roc_auc_score

from config import OUTPUT_DIR, SEED, N_FOLDS
from src.utils.utils import seed_everything

seed_everything(SEED)


# ---------------------------------------------------------------- helpers
def load_labels():
    v3 = np.load(os.path.join(OUTPUT_DIR, 'cleaned_labels_v3.npz'))
    return v3['y_clean'].astype(int), v3['y_orig'].astype(int)


def load_seq(kind='raw'):
    fname = 'sgcc_preprocessed_raw_3ch.npz' if kind == 'raw' else 'sgcc_preprocessed_3ch.npz'
    return np.load(os.path.join(OUTPUT_DIR, fname))['X_seq']


def eval_both(oof, y_clean, y_orig, tag):
    for y, name in [(y_clean, 'v3clean'), (y_orig, 'y_orig')]:
        best_f1, best_th = 0, 0.5
        for th in np.arange(0.05, 0.95, 0.005):
            pred = (oof > th).astype(int)
            if pred.sum() == 0:
                continue
            f = f1_score(y, pred, zero_division=0)
            if f > best_f1:
                best_f1, best_th = f, th
        pred = (oof > best_th).astype(int)
        print(f'  [{tag} @ {name}] F1={best_f1:.4f}, Rec={recall_score(y, pred):.4f}, '
              f'Prec={precision_score(y, pred, zero_division=0):.4f}, '
              f'AUC={roc_auc_score(y, oof):.4f}, th={best_th:.3f}')


def fold_status(out_prefix):
    oof = None
    done = []
    for fi in range(N_FOLDS):
        fpath = os.path.join(OUTPUT_DIR, f'{out_prefix}_fold{fi}.npz')
        if os.path.exists(fpath):
            d = np.load(fpath)
            if oof is None:
                oof = np.zeros(d['n_total'], dtype=np.float32)
            oof[d['vi']] = d['oof']
            done.append(fi)
            print(f'Resumed fold {fi + 1} from {fpath}')
    return oof, done


def save_oof(out_prefix, oof, y_clean, y_orig):
    path = os.path.join(OUTPUT_DIR, f'{out_prefix}_oof.npz')
    np.savez_compressed(path, **{f'oof_{out_prefix}': oof}, y_clean_v3=y_clean, y_orig=y_orig)
    print(f'Saved to {path}')


# ---------------------------------------------------------------- models
def run_tcn_ct(y, y_orig):
    from src.training.coteaching import train_coteaching_cv
    X = load_seq('raw').astype(np.float32)
    oof = train_coteaching_cv(
        X, y, leaf_indices=None, oof_prior=None,
        n_folds=N_FOLDS, seed=SEED, device='cuda',
        tcn_channels=[32, 32, 32, 16], kernel_size=5, dropout=0.3, proj_dim=64,
        epochs=50, batch_size=128, lr=3e-4,
        supcon_weight=0.3, supcon_temp=0.07, sce_alpha=1.0, sce_beta=0.5,
        forget_rate=0.15, warmup_epochs=10,
    )
    save_oof('v3clean_tcn_ct', oof, y, y_orig)
    eval_both(oof, y, y_orig, 'tcn_ct')


def _tcn_fast_train_fold(X_tr, y_tr, X_va, y_va, device, seed, epochs=20):
    """Single TCN (SupCon encoder) with AMP; ~2 min/epoch at bs256 on RTX 5060 Laptop."""
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset
    from src.models.supcon_model import SupConClassifier

    seed_everything(seed)
    model = SupConClassifier(X_tr.shape[1], [64, 64, 64, 32], kernel_size=5,
                             dropout=0.3, proj_dim=64, use_prior=False).to(device)
    pw = torch.tensor([(y_tr == 0).sum() / max((y_tr == 1).sum(), 1)],
                      dtype=torch.float32, device=device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pw)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    scaler = torch.cuda.amp.GradScaler(enabled=True)
    ds = TensorDataset(torch.FloatTensor(X_tr), torch.FloatTensor(y_tr.astype(np.float32)))
    loader = DataLoader(ds, batch_size=256, shuffle=True, drop_last=True, num_workers=0)
    X_va_t = torch.FloatTensor(X_va)

    def predict():
        model.eval()
        probs = []
        with torch.no_grad():
            for s in range(0, len(X_va_t), 1024):
                xb = X_va_t[s:s + 1024].to(device)
                with torch.cuda.amp.autocast(enabled=True):
                    probs.append(torch.sigmoid(model(xb).squeeze(-1)).float().cpu().numpy())
        return np.concatenate(probs)

    best_auc, best_state = 0.0, None
    for ep in range(epochs):
        model.train()
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=True):
                loss = criterion(model(xb).squeeze(-1), yb)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
        sched.step()
        auc = roc_auc_score(y_va, predict())
        if auc > best_auc:
            best_auc = auc
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        print(f'    ep{ep + 1}: val_AUC={auc:.4f} (best={best_auc:.4f})', flush=True)
    model.load_state_dict(best_state)
    return predict()


def _run_tcn_fast(y, y_orig, augment, out_prefix):
    import torch
    X = load_seq('raw').astype(np.float32)
    oof, done = fold_status(out_prefix)
    if oof is None:
        oof = np.zeros(len(y), dtype=np.float32)
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    t0 = time.time()
    for fi, (ti, vi) in enumerate(skf.split(X, y)):
        if fi in done:
            continue
        print(f'\n=== Fold {fi + 1}/{N_FOLDS} ({out_prefix}) ===', flush=True)
        X_tr, y_tr = X[ti], y[ti]
        if augment:
            from src.data.synthetic_anomalies import SyntheticAnomalyAugmenter
            from src.data.synthetic_anomalies_theft import TheftSyntheticAnomalyAugmenter
            n_syn = int(y_tr.sum())
            peerj = SyntheticAnomalyAugmenter(
                seed=SEED + fi, n_synthetic=n_syn,
                anomaly_types=['point', 'contextual', 'collective'],
                point_lambda=0.5, contextual_lambda=1.0, contextual_k=7, collective_lambda=0.5)
            X_pj, y_pj = peerj.fit_transform(X_tr, y_tr)
            theft = TheftSyntheticAnomalyAugmenter(
                seed=SEED + fi + 100, n_synthetic=n_syn,
                anomaly_types=['scale_global', 'zero_interval', 'scale_local',
                               'subtract_random', 'subtract_mean', 'reverse_daily'])
            X_th, y_th = theft.fit_transform(X_tr, y_tr)
            X_tr = np.concatenate([X_tr, X_pj, X_th], axis=0)
            y_tr = np.concatenate([y_tr, y_pj, y_th], axis=0)
            print(f'  After mixed aug: {X_tr.shape}, theft rate={y_tr.mean() * 100:.2f}%')
        oof[vi] = _tcn_fast_train_fold(X_tr, y_tr, X[vi], y[vi], 'cuda', SEED + fi)
        np.savez_compressed(os.path.join(OUTPUT_DIR, f'{out_prefix}_fold{fi}.npz'),
                            oof=oof[vi], vi=vi, ti=ti, n_total=len(y))
        print(f'  fold {fi + 1} done, elapsed {(time.time() - t0) / 60:.1f} min', flush=True)
        torch.cuda.empty_cache()
    save_oof(out_prefix, oof, y, y_orig)
    eval_both(oof, y, y_orig, out_prefix)


def run_patcht(y, y_orig):
    import torch
    from src.models.patch_transformer import train_patch_transformer, predict_patch_transformer
    X = load_seq('raw')
    out_prefix = 'v3clean_patcht'
    oof, done = fold_status(out_prefix)
    if oof is None:
        oof = np.zeros(len(y), dtype=np.float32)
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    cfg = dict(patch_len=30, stride=15, d_model=64, n_layers=2, n_heads=4,
               dropout=0.2, epochs=30, batch_size=128, lr=3e-4)
    for fi, (ti, vi) in enumerate(skf.split(X, y)):
        if fi in done:
            continue
        print(f'\n=== Fold {fi + 1}/{N_FOLDS} (v3clean_patcht) ===')
        model = train_patch_transformer(X[ti], y[ti], oof_prior=None,
                                        device='cuda', seed=SEED + fi, **cfg)
        oof[vi] = np.nan_to_num(predict_patch_transformer(model, X[vi], None, device='cuda'), nan=0.5)
        np.savez_compressed(os.path.join(OUTPUT_DIR, f'{out_prefix}_fold{fi}.npz'),
                            oof=oof[vi], vi=vi, ti=ti, n_total=len(y))
        torch.save(model.state_dict(), os.path.join(OUTPUT_DIR, f'{out_prefix}_fold{fi}.pt'))
        del model
        torch.cuda.empty_cache()
    save_oof(out_prefix, oof, y, y_orig)
    eval_both(oof, y, y_orig, 'patcht')


def _run_amst(y, y_orig, augment, out_prefix):
    import torch
    from src.training.amst_trainer import AMSTTrainer
    from src.models.amst_net import AMSTNet
    X = load_seq('log')
    oof, done = fold_status(out_prefix)
    if oof is None:
        oof = np.zeros(len(y), dtype=np.float32)
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)

    for fi, (ti, vi) in enumerate(skf.split(X, y)):
        if fi in done:
            continue
        print(f'\n=== Fold {fi + 1}/{N_FOLDS} ({out_prefix}) ===')
        t0 = time.time()
        X_train, y_train = X[ti], y[ti]
        X_val = X[vi]

        if augment:
            from src.data.synthetic_anomalies import SyntheticAnomalyAugmenter
            from src.data.synthetic_anomalies_theft import TheftSyntheticAnomalyAugmenter
            n_syn = int(y_train.sum())
            peerj = SyntheticAnomalyAugmenter(
                seed=SEED + fi, n_synthetic=n_syn,
                anomaly_types=['point', 'contextual', 'collective'],
                point_lambda=0.5, contextual_lambda=1.0, contextual_k=7, collective_lambda=0.5)
            X_pj, y_pj = peerj.fit_transform(X_train, y_train)
            theft = TheftSyntheticAnomalyAugmenter(
                seed=SEED + fi + 100, n_synthetic=n_syn,
                anomaly_types=['scale_global', 'zero_interval', 'scale_local',
                               'subtract_random', 'subtract_mean', 'reverse_daily'])
            X_th, y_th = theft.fit_transform(X_train, y_train)
            X_train = np.concatenate([X_train, X_pj, X_th], axis=0)
            y_train = np.concatenate([y_train, y_pj, y_th], axis=0)
            print(f'  After mixed aug: {X_train.shape}, theft rate={y_train.mean() * 100:.2f}%')
            trainer = AMSTTrainer(
                dataset='sgcc', device='cuda',
                use_diffaug=False, use_supcon=False, use_coteaching=False, use_prior=False,
                d_mamba=64, d_trans=128, d_freq=64, proj_dim=128,
                n_mamba_layers=2, n_trans_layers=2, n_heads=4, dropout=0.2,
                epochs=25, batch_size=128, lr=1e-4, patience=10, recall_weight=10.0,
                use_amp=True, label_smoothing=0.1, use_synthetic_anomalies=False)
        else:
            trainer = AMSTTrainer(
                dataset='sgcc', device='cuda',
                use_diffaug=False, use_supcon=False, use_coteaching=True,
                forget_rate=0.15, num_gradual=10, use_prior=False,
                d_mamba=64, d_trans=128, d_freq=64, proj_dim=128,
                n_mamba_layers=2, n_trans_layers=2, n_heads=4, dropout=0.2,
                epochs=50, batch_size=64, lr=1e-4, patience=15, recall_weight=5.0)

        train_loader = trainer._build_loaders(X_train, y_train, prior=None, shuffle=True)
        val_loader = trainer._build_loaders(X_val, y[vi], prior=None, shuffle=False)
        model = AMSTNet(
            in_channels=X.shape[1], seq_len=X.shape[2],
            d_mamba=trainer.d_mamba, d_trans=trainer.d_trans, d_freq=trainer.d_freq,
            proj_dim=trainer.proj_dim, n_mamba_layers=trainer.n_mamba_layers,
            n_trans_layers=trainer.n_trans_layers, n_heads=trainer.n_heads,
            dropout=trainer.dropout, use_freq=True, use_supcon=trainer.use_supcon,
            prior_dim=0)
        print(f'  Model params: {sum(p.numel() for p in model.parameters()):,}')
        model = trainer._train_single_network(
            model, train_loader, val_loader, y[vi],
            epochs=trainer.epochs, lr=trainer.lr, weight_decay=trainer.weight_decay,
            patience=trainer.patience, fold_idx=fi)
        oof[vi] = trainer._predict_proba(model, val_loader)
        np.savez_compressed(os.path.join(OUTPUT_DIR, f'{out_prefix}_fold{fi}.npz'),
                            oof=oof[vi], vi=vi, ti=ti, n_total=len(y))
        print(f'  Saved fold {fi + 1} OOF, time={(time.time() - t0) / 60:.1f}min')
        torch.save(model.state_dict(), os.path.join(OUTPUT_DIR, f'{out_prefix}_fold{fi}.pt'))
        del model
        torch.cuda.empty_cache()

    save_oof(out_prefix, oof, y, y_orig)
    eval_both(oof, y, y_orig, out_prefix)


def run_gbdt(y, y_orig):
    """Untainted GBDT (LGB+XGB+CatBoost) on handcrafted stat features, v3 labels."""
    import lightgbm as lgb
    import xgboost as xgb
    from catboost import CatBoostClassifier
    from config import SGCC_CONFIG

    pre = np.load(os.path.join(OUTPUT_DIR, 'sgcc_preprocessed.npz'))
    stat = np.nan_to_num(pre['stat_features'], nan=0.0, posinf=0.0, neginf=0.0)
    mask_frac = pre['impute_mask'].astype(float).mean(axis=1).reshape(-1, 1)
    X = np.hstack([stat, mask_frac]).astype(np.float32)

    n = len(y)
    oofs = {k: np.zeros(n) for k in ('lgb', 'xgb', 'cb')}
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    for fi, (ti, vi) in enumerate(skf.split(X, y)):
        pw = (y[ti] == 0).sum() / max((y[ti] == 1).sum(), 1)
        cfg = SGCC_CONFIG['gbdt_params']['lgb'].copy()
        cfg.update(scale_pos_weight=pw, random_state=SEED + fi, verbose=-1)
        m = lgb.LGBMClassifier(**cfg)
        m.fit(X[ti], y[ti], eval_set=[(X[vi], y[vi])],
              callbacks=[lgb.early_stopping(80, verbose=False), lgb.log_evaluation(0)])
        oofs['lgb'][vi] = m.predict_proba(X[vi])[:, 1]

        cfg = SGCC_CONFIG['gbdt_params']['xgb'].copy()
        cfg.update(scale_pos_weight=pw, random_state=SEED + fi, verbosity=0, tree_method='hist')
        m = xgb.XGBClassifier(**cfg)
        m.fit(X[ti], y[ti])
        oofs['xgb'][vi] = m.predict_proba(X[vi])[:, 1]

        cfg = SGCC_CONFIG['gbdt_params']['catboost'].copy()
        cfg.update(random_seed=SEED + fi)
        m = CatBoostClassifier(**cfg)
        m.fit(X[ti], y[ti], eval_set=(X[vi], y[vi]), early_stopping_rounds=80, verbose=False)
        oofs['cb'][vi] = m.predict_proba(X[vi])[:, 1]

    best_f1, best_w = 0, (0.4, 0.3, 0.3)
    for wl in np.arange(0.0, 1.01, 0.1):
        for wx in np.arange(0.0, 1.0 - wl + 0.001, 0.1):
            wc = 1.0 - wl - wx
            ens = wl * oofs['lgb'] + wx * oofs['xgb'] + wc * oofs['cb']
            for th in np.arange(0.05, 0.95, 0.01):
                f = f1_score(y, (ens > th).astype(int), zero_division=0)
                if f > best_f1:
                    best_f1, best_w = f, (wl, wx, wc)
    ens = best_w[0] * oofs['lgb'] + best_w[1] * oofs['xgb'] + best_w[2] * oofs['cb']
    print(f'GBDT blend weights: LGB={best_w[0]:.2f} XGB={best_w[1]:.2f} Cat={best_w[2]:.2f}')
    save_oof('v3clean_gbdt', ens, y, y_orig)
    for k, o in oofs.items():
        save_oof(f'v3clean_gbdt_{k}', o, y, y_orig)
    eval_both(ens, y, y_orig, 'gbdt')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--models', default='gbdt,tcn_fast,tcn_aug,patcht',
                        help='comma list: gbdt,tcn_fast,tcn_aug,patcht,tcn_ct,amst_ct,amst_aug '
                             '(amst_* are fp32-slow on laptop GPUs)')
    args = parser.parse_args()

    y, y_orig = load_labels()
    print(f'V3 labels: positives {y.sum()} ({y.mean() * 100:.2f}%), '
          f'diff vs original: {(y != y_orig).sum()}')

    for name in args.models.split(','):
        name = name.strip()
        t0 = time.time()
        if name == 'gbdt':
            run_gbdt(y, y_orig)
        elif name == 'tcn_fast':
            _run_tcn_fast(y, y_orig, augment=False, out_prefix='v3clean_tcn_fast')
        elif name == 'tcn_aug':
            _run_tcn_fast(y, y_orig, augment=True, out_prefix='v3clean_tcn_aug')
        elif name == 'tcn_ct':
            run_tcn_ct(y, y_orig)
        elif name == 'patcht':
            run_patcht(y, y_orig)
        elif name == 'amst_ct':
            _run_amst(y, y_orig, augment=False, out_prefix='v3clean_amst_ct')
        elif name == 'amst_aug':
            _run_amst(y, y_orig, augment=True, out_prefix='v3clean_amst_aug')
        else:
            print(f'Unknown model: {name}')
            continue
        print(f'--- {name} done in {(time.time() - t0) / 60:.1f} min ---')


if __name__ == '__main__':
    main()
