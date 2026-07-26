"""
Full OOF stacking pipeline: GBDT + TCN(V108-style) + Meta-learner.

V108-style preprocessing: global StandardScaler (not per-user winsorize).
This preserves absolute consumption level information.
"""
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import time
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.metrics import f1_score, recall_score, precision_score, roc_auc_score
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostClassifier

from config import SEED, N_FOLDS, OUTPUT_DIR
from utils import seed_everything, best_f1_score, best_f1_recall_constrained
from models import TCNWithLeafEmbedding, RecallOrientedFocalLoss


def v108_preprocess(X_raw):
    """V108-style: log1p + global StandardScaler + 3-channel."""
    nmk = np.isnan(X_raw)
    X_filled = np.nan_to_num(X_raw, nan=0.0)
    X_log = np.log1p(np.maximum(X_filled, 0))
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_log).astype(np.float32)
    X_scaled = np.clip(X_scaled, -5, 5)
    X_missing = nmk.astype(np.float32)
    X_zero = (X_filled == 0).astype(np.float32)
    X_seq = np.stack([X_scaled, X_missing, X_zero], axis=1)
    return X_seq


def train_gbdt_oof(stat_features, y, impute_mask, skf):
    """GBDT ensemble 5-fold OOF."""
    print('\n--- GBDT Ensemble (Expert A) ---')
    n = len(y)
    mr = impute_mask.mean(axis=1).reshape(-1, 1)
    X = np.hstack([stat_features, mr]).astype(np.float32)
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

    oof = np.zeros(n)
    leaf_lgb = np.zeros((n, 100), dtype=np.int32)
    leaf_xgb = np.zeros((n, 100), dtype=np.int32)

    for fi, (ti, vi) in enumerate(skf.split(X, y)):
        pw = (y[ti] == 0).sum() / max((y[ti] == 1).sum(), 1)
        lgb_m = lgb.LGBMClassifier(n_estimators=1000, max_depth=7, learning_rate=0.05,
            num_leaves=63, subsample=0.8, colsample_bytree=0.8, scale_pos_weight=pw,
            random_state=SEED, verbose=-1)
        lgb_m.fit(X[ti], y[ti], eval_set=[(X[vi], y[vi])],
                  callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(0)])
        xgb_m = xgb.XGBClassifier(n_estimators=1000, max_depth=7, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8, scale_pos_weight=pw, tree_method='hist',
            random_state=SEED, verbosity=0)
        xgb_m.fit(X[ti], y[ti], eval_set=[(X[vi], y[vi])], verbose=False)
        cb_m = CatBoostClassifier(iterations=1000, depth=7, learning_rate=0.05,
            auto_class_weights='Balanced', random_seed=SEED, verbose=0)
        cb_m.fit(X[ti], y[ti], eval_set=(X[vi], y[vi]), early_stopping_rounds=50, verbose=0)
        oof[vi] = (0.4 * lgb_m.predict_proba(X[vi])[:, 1] +
                   0.3 * xgb_m.predict_proba(X[vi])[:, 1] +
                   0.3 * cb_m.predict_proba(X[vi])[:, 1])

        ll = lgb_m.predict(X[vi], pred_leaf=True)
        leaf_lgb[vi] = ll[:, :100] if ll.shape[1] >= 100 else np.pad(ll, ((0, 0), (0, 100 - ll.shape[1])))
        xl = xgb_m.get_booster().predict(xgb.DMatrix(X[vi]), pred_leaf=True)
        leaf_xgb[vi] = xl[:, :100] if xl.shape[1] >= 100 else np.pad(xl, ((0, 0), (0, 100 - xl.shape[1])))

    leaf_indices = np.concatenate([leaf_lgb, leaf_xgb], axis=1)
    f1, _, rec, prec = best_f1_score(y, oof)
    auc = roc_auc_score(y, oof)
    print(f'  GBDT: AUC={auc:.4f} F1={f1:.4f} Rec={rec:.4f} Prec={prec:.4f}')
    return oof, leaf_indices


def train_tcn_v108_oof(X_seq, y, leaf_indices, oof_prior, skf, device='cuda'):
    """TCN with V108-style preprocessing, 5-fold OOF."""
    print('\n--- TCN V108 (Expert B) ---')
    n = len(y)
    oof = np.zeros(n)

    for fi, (ti, vi) in enumerate(skf.split(X_seq, y)):
        print(f'  Fold {fi+1}/{N_FOLDS}...', end=' ', flush=True)
        torch.cuda.empty_cache()

        model = TCNWithLeafEmbedding(
            in_channels=3, tcn_channels=[32, 32, 32, 16], kernel_size=5, dropout=0.3,
            n_trees=200, num_leaves=31, leaf_embed_dim=4, leaf_output_dim=32,
            use_prior=True,
        ).to(device)

        criterion = RecallOrientedFocalLoss(alpha=0.75, gamma=2.0, recall_weight=3.0)
        optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50, eta_min=1e-6)

        prior_tr = oof_prior[ti].astype(np.float32)
        prior_va = oof_prior[vi].astype(np.float32)
        ds = TensorDataset(
            torch.FloatTensor(X_seq[ti]),
            torch.LongTensor(leaf_indices[ti]),
            torch.FloatTensor(y[ti]),
            torch.FloatTensor(prior_tr),
        )
        dl = DataLoader(ds, batch_size=64, shuffle=True, drop_last=True)

        best_f1 = 0
        best_state = None
        pat = 0
        for ep in range(50):
            model.train()
            for bx, bl, by, bp in dl:
                bx, bl, by, bp = bx.to(device), bl.to(device), by.to(device), bp.to(device)
                optimizer.zero_grad()
                loss = criterion(model(bx, bl, bp), by)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
            scheduler.step()

            model.eval()
            with torch.no_grad():
                xv = torch.FloatTensor(X_seq[vi]).to(device)
                lv = torch.LongTensor(leaf_indices[vi]).to(device)
                pv = torch.FloatTensor(prior_va).to(device)
                probs = torch.sigmoid(model(xv, lv, pv)).cpu().numpy()

            bf = max((f1_score(y[vi], (probs > t).astype(int), zero_division=0)
                      for t in np.arange(0.2, 0.8, 0.01)), default=0)
            if bf > best_f1:
                best_f1 = bf
                best_state = model.state_dict().copy()
                pat = 0
            else:
                pat += 1
            if pat >= 7:
                break

        model.load_state_dict(best_state)
        model.eval()
        with torch.no_grad():
            xv = torch.FloatTensor(X_seq[vi]).to(device)
            lv = torch.LongTensor(leaf_indices[vi]).to(device)
            pv = torch.FloatTensor(prior_va).to(device)
            oof[vi] = torch.sigmoid(model(xv, lv, pv)).cpu().numpy()
        print(f'F1={best_f1:.4f} AUC={roc_auc_score(y[vi], oof[vi]):.4f}')

        del model
        torch.cuda.empty_cache()

    f1, _, rec, prec = best_f1_score(y, oof)
    auc = roc_auc_score(y, oof)
    print(f'  TCN: AUC={auc:.4f} F1={f1:.4f} Rec={rec:.4f} Prec={prec:.4f}')
    return oof


def meta_stack(oof_a, oof_b, stat_features, impute_mask, y, skf):
    """XGBoost meta-learner stacking."""
    print('\n--- Meta-Learner ---')
    n = len(y)
    mr = impute_mask.mean(axis=1).reshape(-1, 1)
    X = np.column_stack([
        stat_features, mr,
        oof_a.reshape(-1, 1), oof_b.reshape(-1, 1),
        np.abs(oof_a - oof_b).reshape(-1, 1),
        ((oof_a + oof_b) / 2).reshape(-1, 1),
    ]).astype(np.float32)
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

    oof = np.zeros(n)
    pos_weight = (y == 0).sum() / max((y == 1).sum(), 1)

    for fi, (ti, vi) in enumerate(skf.split(X, y)):
        m = xgb.XGBClassifier(
            n_estimators=500, max_depth=4, learning_rate=0.03,
            subsample=0.8, colsample_bytree=0.8, reg_alpha=0.1, reg_lambda=0.1,
            scale_pos_weight=pos_weight, tree_method='hist',
            random_state=SEED, verbosity=0,
        )
        m.fit(X[ti], y[ti])
        oof[vi] = m.predict_proba(X[vi])[:, 1]

    f1, th, rec, prec = best_f1_score(y, oof)
    auc = roc_auc_score(y, oof)
    print(f'  Meta: AUC={auc:.4f} F1={f1:.4f} Rec={rec:.4f} Prec={prec:.4f} th={th:.3f}')

    f1_rc, th_rc, rec_rc, prec_rc = best_f1_recall_constrained(y, oof, min_recall=0.90)
    if rec_rc >= 0.90:
        print(f'  Meta (Rec>=0.90): F1={f1_rc:.4f} Rec={rec_rc:.4f} Prec={prec_rc:.4f} th={th_rc:.3f}')
    else:
        print(f'  Meta (Rec>=0.90): Not achievable')

    return oof


def run_full_pipeline():
    seed_everything(SEED)
    t0 = time.time()

    print('=' * 60)
    print('  Full OOF Stacking Pipeline (V108-style)')
    print('=' * 60)

    # Load raw data for V108 preprocessing
    print('\n[1] Loading raw data...')
    df = pd.read_csv('data/raw_data.csv')
    date_cols = [c for c in df.columns if '/' in str(c) and len(str(c)) <= 10]
    X_raw = df[date_cols].values.astype(np.float32)
    y = df['FLAG'].values.astype(np.int32)
    del df
    print(f'  Raw: {X_raw.shape}, theft={y.sum()}/{len(y)}')

    # V108 preprocessing
    print('\n[2] V108 preprocessing (global StandardScaler)...')
    X_seq = v108_preprocess(X_raw)
    print(f'  X_seq: {X_seq.shape}')

    # Load stat features
    print('\n[3] Loading stat features...')
    d = np.load('output/sgcc_preprocessed.npz')
    stat_features = d['stat_features']
    impute_mask = d['impute_mask']
    print(f'  stat: {stat_features.shape}')

    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)

    # Stage 1: GBDT
    print('\n[4] GBDT training...')
    oof_a, leaf_indices = train_gbdt_oof(stat_features, y, impute_mask, skf)

    # Stage 2: TCN V108
    print('\n[5] TCN V108 training...')
    oof_b = train_tcn_v108_oof(X_seq, y, leaf_indices, oof_a, skf)

    # Stage 3: Meta stacking
    print('\n[6] Meta stacking...')
    oof_meta = meta_stack(oof_a, oof_b, stat_features, impute_mask, y, skf)

    # Summary
    print(f'\n{"=" * 60}')
    print(f'  FINAL RESULTS (0% review = full coverage)')
    print(f'{"=" * 60}')
    for name, oof in [('GBDT', oof_a), ('TCN V108', oof_b), ('Meta', oof_meta)]:
        f1, th, rec, prec = best_f1_score(y, oof)
        auc = roc_auc_score(y, oof)
        tp = ((oof > th) & (y == 1)).sum()
        fp = ((oof > th) & (y == 0)).sum()
        fn = ((oof <= th) & (y == 1)).sum()
        print(f'  {name:<10s}: AUC={auc:.4f} F1={f1:.4f} Rec={rec:.4f} Prec={prec:.4f} '
              f'TP={tp} FP={fp} FN={fn} th={th:.3f}')

    print(f'\n  V225 reference: AUC=0.9804 F1=0.8457 TP=2952 FP=414 FN=663')
    print(f'  Time: {(time.time()-t0)/60:.1f} min')

    # Save
    np.savez_compressed(
        os.path.join(OUTPUT_DIR, 'full_stacking_results.npz'),
        oof_a=oof_a, oof_b=oof_b, oof_meta=oof_meta, y=y,
    )
    print(f'  Saved to {OUTPUT_DIR}/full_stacking_results.npz')

    return oof_a, oof_b, oof_meta


if __name__ == '__main__':
    run_full_pipeline()
