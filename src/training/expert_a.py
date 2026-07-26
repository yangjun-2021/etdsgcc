"""
Expert A: GBDT Training (LightGBM + XGBoost + CatBoost ensemble).

Note: "Expert A" is a naming convention for a heterogeneous GBDT ensemble.
It is NOT a Mixture-of-Experts (MoE) architecture — there is no gating
network, expert routing, or load-balancing loss. The name "Expert" here
simply denotes an independent model component in the two-stage cascade
(Expert A GBDT → Expert B TCN → Meta Learner).

Unified trainer for SGCC and OEDI datasets, parameterized by dataset config.
"""
import os
import pickle
import numpy as np
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score, recall_score, roc_auc_score
import warnings

warnings.filterwarnings('ignore')

from config import SGCC_CONFIG, OEDI_CONFIG, SEED, N_FOLDS, OUTPUT_DIR


class ExpertATrainer:
    """Unified GBDT trainer for both SGCC and OEDI datasets.

    Parameters
    ----------
    dataset : str
        'sgcc' or 'oedi'
    """

    def __init__(self, dataset='sgcc'):
        self.dataset = dataset
        self.config = SGCC_CONFIG if dataset == 'sgcc' else OEDI_CONFIG
        self.dataset_name = self.config['name']
        self.n_leaf_trees = self.config['tcn_params']['n_trees']

    def _get_fold_splits(self, labels, fold_assignments=None):
        """Return list of (train_idx, val_idx) for each fold."""
        if self.dataset == 'sgcc':
            skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
            return list(skf.split(np.zeros(len(labels)), labels))
        else:
            unique_folds = np.unique(fold_assignments)
            splits = []
            for fold_idx in unique_folds:
                train_idx = np.where(fold_assignments != fold_idx)[0]
                val_idx = np.where(fold_assignments == fold_idx)[0]
                splits.append((train_idx, val_idx))
            return splits

    def _prepare_features(self, stat_features, impute_mask=None):
        """Prepare augmented features. For SGCC, add miss_ratio column."""
        stat_features = np.nan_to_num(stat_features, nan=0.0)
        if self.dataset == 'sgcc' and impute_mask is not None:
            miss_ratio = impute_mask.mean(axis=1).reshape(-1, 1)
            stat_features_aug = np.hstack([stat_features, miss_ratio]).astype(np.float32)
        else:
            stat_features_aug = stat_features.astype(np.float32)
        return stat_features_aug

    def _extract_leaf_indices(self, model, X_val, model_type):
        """Extract leaf indices from a trained model, padded/truncated to n_leaf_trees."""
        n_target = self.n_leaf_trees

        if model_type == 'lgb':
            leaf = model.predict(X_val, pred_leaf=True)
        elif model_type == 'xgb':
            booster = model.get_booster()
            dval = xgb.DMatrix(X_val)
            leaf = booster.predict(dval, pred_leaf=True)
        else:
            raise ValueError(f"Unknown model_type: {model_type}")

        # Degenerate models (0/1 trees) can return a 1-D array; normalize to 2-D
        if leaf.ndim == 1:
            leaf = leaf.reshape(-1, 1)

        n_trees = leaf.shape[1]
        if n_trees >= n_target:
            return leaf[:, :n_target]
        else:
            return np.pad(leaf, ((0, 0), (0, n_target - n_trees)), constant_values=0)

    def train(self, stat_features, labels, impute_mask=None, fold_assignments=None):
        """Train GBDT ensemble across folds.

        Parameters
        ----------
        stat_features : np.ndarray
            Statistical features matrix
        labels : np.ndarray
            Binary labels
        impute_mask : np.ndarray, optional
            Missing value mask (required for SGCC)
        fold_assignments : np.ndarray, optional
            Pre-defined fold assignments (required for OEDI)

        Returns
        -------
        oof_proba : np.ndarray
            Out-of-fold predicted probabilities
        combined_leaf : np.ndarray
            Combined leaf indices from LGB and XGB
        fold_models : list
            List of dicts with trained models per fold
        """
        print("=" * 60)
        print(f"Expert A: GBDT Training ({self.dataset_name.upper()})")
        print("=" * 60)

        n = len(labels)
        stat_features_aug = self._prepare_features(stat_features, impute_mask)
        print(f"  Features: {stat_features_aug.shape}, NaN after fill: {np.isnan(stat_features_aug).sum()}")

        splits = self._get_fold_splits(labels, fold_assignments)

        oof_proba = np.zeros(n)
        oof_leaf_indices = {
            'lgb': np.zeros((n, self.n_leaf_trees), dtype=np.int32),
            'xgb': np.zeros((n, self.n_leaf_trees), dtype=np.int32),
        }
        fold_models = []

        for fold_idx, (train_idx, val_idx) in enumerate(splits):
            print(f"\n  Fold {fold_idx + 1}/{N_FOLDS}")
            X_train, X_val = stat_features_aug[train_idx], stat_features_aug[val_idx]
            y_train, y_val = labels[train_idx], labels[val_idx]

            print(f"    Train: {len(train_idx)}, Val: {len(val_idx)}")

            # LightGBM
            lgb_params = self.config['gbdt_params']['lgb'].copy()
            lgb_model = lgb.LGBMClassifier(**lgb_params)
            lgb_model.fit(X_train, y_train, eval_set=[(X_val, y_val)],
                          callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(period=0)])

            # XGBoost
            xgb_params = self.config['gbdt_params']['xgb'].copy()
            xgb_params['eval_metric'] = 'auc'
            xgb_model = xgb.XGBClassifier(**xgb_params)
            # XGBoost 3.x requires early stopping via attributes, not fit kwargs
            xgb_model.early_stopping_rounds = 50
            xgb_model.callbacks = [xgb.callback.EarlyStopping(rounds=50, save_best=True)]
            xgb_model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)

            # CatBoost
            cb_params = self.config['gbdt_params']['catboost'].copy()
            cb_model = CatBoostClassifier(**cb_params)
            cb_model.fit(X_train, y_train, eval_set=(X_val, y_val),
                         early_stopping_rounds=50, verbose=0)

            # Ensemble with learned weights via validation-set grid search
            lgb_proba = lgb_model.predict_proba(X_val)[:, 1]
            xgb_proba = xgb_model.predict_proba(X_val)[:, 1]
            cb_proba = cb_model.predict_proba(X_val)[:, 1]

            val_probas = np.column_stack([lgb_proba, xgb_proba, cb_proba])
            best_w, best_f1 = None, -1.0
            for wl in np.arange(0.0, 1.01, 0.1):
                for wx in np.arange(0.0, 1.0 - wl + 0.001, 0.1):
                    wc = 1.0 - wl - wx
                    w = np.array([wl, wx, wc])
                    ens = val_probas.dot(w)
                    # threshold search for F1
                    ths = np.arange(0.1, 0.9, 0.02)
                    preds = (ens.reshape(-1, 1) > ths).astype(int)
                    f1s = np.array([f1_score(y_val, preds[:, i], zero_division=0) for i in range(len(ths))])
                    f = f1s.max()
                    if f > best_f1:
                        best_f1, best_w = f, w
            if best_w is None:
                best_w = np.array([0.4, 0.3, 0.3])
            ensemble_proba = val_probas.dot(best_w)
            oof_proba[val_idx] = ensemble_proba
            print(f"    Fold ensemble weights: LGB={best_w[0]:.2f}, XGB={best_w[1]:.2f}, Cat={best_w[2]:.2f}")

            # Leaf indices
            oof_leaf_indices['lgb'][val_idx] = self._extract_leaf_indices(lgb_model, X_val, 'lgb')
            oof_leaf_indices['xgb'][val_idx] = self._extract_leaf_indices(xgb_model, X_val, 'xgb')

            # Metrics
            val_pred = (ensemble_proba > 0.5).astype(int)
            f1 = f1_score(y_val, val_pred)
            recall = recall_score(y_val, val_pred)
            auc = roc_auc_score(y_val, ensemble_proba)
            print(f"    F1={f1:.4f}, Recall={recall:.4f}, AUC={auc:.4f}")

            fold_models.append({'lgb': lgb_model, 'xgb': xgb_model, 'cb': cb_model})

        # Overall metrics
        overall_pred = (oof_proba > 0.5).astype(int)
        overall_f1 = f1_score(labels, overall_pred)
        overall_recall = recall_score(labels, overall_pred)
        overall_auc = roc_auc_score(labels, oof_proba)

        # Find best threshold
        best_th = 0.5
        best_f1 = 0
        for th in np.arange(0.1, 0.9, 0.005):
            pred = (oof_proba > th).astype(int)
            f1 = f1_score(labels, pred)
            if f1 > best_f1:
                best_f1 = f1
                best_th = th

        print(f"\n[Expert A {self.dataset_name.upper()}] Overall: F1={overall_f1:.4f}, "
              f"Recall={overall_recall:.4f}, AUC={overall_auc:.4f}")
        print(f"  Best threshold: {best_th:.3f}, F1 at best: {best_f1:.4f}")

        combined_leaf = np.concatenate([oof_leaf_indices['lgb'], oof_leaf_indices['xgb']], axis=1)
        print(f"  Combined leaf indices shape: {combined_leaf.shape}")

        # Save results
        save_path = os.path.join(OUTPUT_DIR, f'{self.dataset_name}_expert_a.npz')
        label_key = 'flags' if self.dataset == 'sgcc' else 'y'
        np.savez_compressed(
            save_path,
            oof_proba=oof_proba,
            leaf_indices_lgb=oof_leaf_indices['lgb'],
            leaf_indices_xgb=oof_leaf_indices['xgb'],
            **{label_key: labels},
        )

        with open(os.path.join(OUTPUT_DIR, f'{self.dataset_name}_expert_a_models.pkl'), 'wb') as f:
            pickle.dump(fold_models, f)

        metadata = {
            'best_threshold': best_th,
            'overall_f1': overall_f1,
            'overall_auc': overall_auc,
            'n_trees_lgb': self.n_leaf_trees,
            'n_trees_xgb': self.n_leaf_trees,
        }
        with open(os.path.join(OUTPUT_DIR, f'{self.dataset_name}_expert_a_meta.pkl'), 'wb') as f:
            pickle.dump(metadata, f)

        return oof_proba, combined_leaf, fold_models


# Backward-compatible function wrappers
def train_gbdt_sgcc(stat_features, flags, impute_mask):
    """Backward-compatible wrapper for SGCC GBDT training."""
    trainer = ExpertATrainer(dataset='sgcc')
    return trainer.train(stat_features, flags, impute_mask=impute_mask)


def train_gbdt_oedi(stat_features, y, fold_assignments):
    """Backward-compatible wrapper for OEDI GBDT training."""
    trainer = ExpertATrainer(dataset='oedi')
    return trainer.train(stat_features, y, fold_assignments=fold_assignments)


if __name__ == '__main__':
    print("Run expert_a.py through the main pipeline (run_pipeline.py)")
