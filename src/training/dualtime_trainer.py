"""Trainer for DualTimeFusion model with 5-fold CV on SGCC.

Uses identical splits and evaluation protocol as Expert B for fair comparison.
"""
import os, sys, time, numpy as np, torch
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if PROJECT_ROOT not in sys.path: sys.path.insert(0, PROJECT_ROOT)

from config import OUTPUT_DIR, SEED, N_FOLDS, DEVICE
from src.models.dual_time_fusion import DualTimeFusionModel, AsymmetricFocalLoss
from src.utils.utils import seed_everything


class DualTimeTrainer:
    def __init__(self, dataset='sgcc', epochs=30, batch_size=32,
                 lr=3e-4, weight_decay=1e-4, patience=8,
                 d_model=256, dropout=0.2, seq_target_len=256,
                 use_amp=True, device=DEVICE):
        self.dataset = dataset; self.epochs = epochs; self.batch_size = batch_size
        self.lr = lr; self.weight_decay = weight_decay; self.patience = patience
        self.d_model = d_model; self.dropout = dropout
        self.seq_target_len = seq_target_len
        self.use_amp = use_amp and torch.device(device).type == 'cuda'
        self.device = torch.device(device)

    def _loader(self, X, y=None, stat=None, shuffle=True, weighted=False):
        X_t = torch.from_numpy(X.astype(np.float32))
        tensors = [X_t]
        if y is not None:
            y_t = torch.from_numpy(np.asarray(y, dtype=np.int64))
            tensors.append(y_t)
        if stat is not None:
            s_t = torch.from_numpy(stat.astype(np.float32))
            tensors.append(s_t)
        ds = TensorDataset(*tensors)
        if weighted and y is not None:
            y_arr = np.asarray(y)
            pos, neg = max((y_arr==1).sum(),1), max((y_arr==0).sum(),1)
            weights = np.where(y_arr==1, 1.0/pos, 1.0/neg)
            sampler = WeightedRandomSampler(weights, len(weights), replacement=True)
            return DataLoader(ds, batch_size=self.batch_size, sampler=sampler)
        return DataLoader(ds, batch_size=self.batch_size, shuffle=shuffle, drop_last=False)

    @staticmethod
    def _best_threshold(y_true, proba):
        idx = np.argsort(proba)
        ys = np.asarray(y_true)[idx]; n = len(y_true)
        tp = max(ys.sum(), 1.0); cp = np.cumsum(ys[::-1])
        k = np.arange(1, n+1, dtype=np.float64)
        p = cp/k; r = cp/tp; d = p+r; d[d==0]=1.0
        f1 = 2*p*r/d; best = int(np.argmax(f1))
        return float(f1[best]), float(proba[idx[-(best+1)]]), float(r[best]), float(p[best])

    def train(self, X_seq, flags, stat_features=None, fold_assignments=None):
        seed_everything(SEED)
        start = time.time()
        X_seq = np.asarray(X_seq, dtype=np.float32)
        flags = np.asarray(flags, dtype=np.int64)
        stat_features = np.asarray(stat_features, dtype=np.float32) if stat_features is not None else None

        # Downsample sequence for speed
        if self.seq_target_len and X_seq.shape[-1] > self.seq_target_len:
            X_t = torch.from_numpy(X_seq)
            X_seq = torch.nn.functional.adaptive_avg_pool1d(X_t, self.seq_target_len).numpy()

        if fold_assignments is not None:
            splits = [(np.where(fold_assignments!=f)[0], np.where(fold_assignments==f)[0])
                      for f in np.unique(fold_assignments)]
        else:
            skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
            splits = list(skf.split(np.zeros(len(flags)), flags))

        n = len(flags)
        oof = np.zeros(n, dtype=np.float32)
        n_in = X_seq.shape[1]
        stat_dim = stat_features.shape[1] if stat_features is not None else 0

        print("="*60)
        print(f"DualTimeFusion Training ({self.dataset.upper()})")
        print(f"  d_model={self.d_model} epochs={self.epochs} batch={self.batch_size}")
        print("="*60)

        for fold_idx, (train_idx, val_idx) in enumerate(splits):
            print(f"\nFold {fold_idx+1}/{len(splits)}")
            X_tr, X_va = X_seq[train_idx], X_seq[val_idx]
            y_tr, y_va = flags[train_idx], flags[val_idx]
            s_tr = stat_features[train_idx] if stat_features is not None else None
            s_va = stat_features[val_idx] if stat_features is not None else None

            model = DualTimeFusionModel(n_in, X_seq.shape[2], self.d_model, stat_dim, self.dropout).to(self.device)
            n_params = sum(p.numel() for p in model.parameters())
            print(f"  params={n_params:,} train={len(train_idx)} val={len(val_idx)}")

            opt = torch.optim.AdamW(model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
            sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=self.epochs, eta_min=self.lr*0.05)
            scaler = torch.cuda.amp.GradScaler() if self.use_amp else None

            pos, neg = max((y_tr==1).sum(),1), max((y_tr==0).sum(),1)
            focal = AsymmetricFocalLoss(0.75, 1.0, 3.0, pos_weight=neg/pos)
            train_loader = self._loader(X_tr, y_tr, s_tr, shuffle=True, weighted=True)
            val_loader = self._loader(X_va, y_va, s_va, shuffle=False)

            best_state, best_f1, patience_ctr = None, 0.0, 0

            for epoch in range(self.epochs):
                model.train(); losses = []
                for batch in train_loader:
                    x = batch[0].to(self.device)
                    y = batch[1].float().to(self.device)
                    sf = batch[2].to(self.device) if len(batch)>2 else None
                    opt.zero_grad()
                    if scaler is not None:
                        with torch.cuda.amp.autocast():
                            logit = model(x, sf); loss = focal(logit, y)
                        scaler.scale(loss).backward(); scaler.unscale_(opt)
                        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                        scaler.step(opt); scaler.update()
                    else:
                        logit = model(x, sf); loss = focal(logit, y)
                        loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                        opt.step()
                    losses.append(loss.item())
                sch.step()

                # Validate
                model.eval(); va_probs = []
                with torch.no_grad():
                    for batch in val_loader:
                        x = batch[0].to(self.device)
                        sf = batch[2].to(self.device) if len(batch)>2 else None
                        va_probs.append(torch.sigmoid(model(x, sf)).cpu().numpy())
                va_prob = np.concatenate(va_probs)
                val_f1, _, val_rec, val_prec = self._best_threshold(y_va, va_prob)

                if val_f1 > best_f1:
                    best_f1 = val_f1; patience_ctr = 0
                    best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                else:
                    patience_ctr += 1
                if epoch==0 or (epoch+1)%5==0:
                    print(f"  epoch {epoch+1}: loss={np.mean(losses):.4f} val_f1={val_f1:.4f} rec={val_rec:.4f} prec={val_prec:.4f}")
                if patience_ctr >= self.patience:
                    print(f"  early stop at epoch {epoch+1}"); break

            model.load_state_dict(best_state)
            model.eval()
            with torch.no_grad():
                va_all = []
                for batch in val_loader:
                    x = batch[0].to(self.device)
                    sf = batch[2].to(self.device) if len(batch)>2 else None
                    va_all.append(torch.sigmoid(model(x, sf)).cpu().numpy())
                oof[val_idx] = np.concatenate(va_all)

            f1, th, rec, prec = self._best_threshold(y_va, oof[val_idx])
            print(f"  fold result: F1={f1:.4f} th={th:.3f} rec={rec:.4f} prec={prec:.4f}")
            torch.save(model.state_dict(), os.path.join(OUTPUT_DIR, f'{self.dataset}_dualtime_fold{fold_idx}.pt'))
            del model
            if torch.cuda.is_available(): torch.cuda.empty_cache()

        f1, th, rec, prec = self._best_threshold(flags, oof)
        auc = roc_auc_score(flags, oof)
        print(f"\n[DualTimeFusion] F1={f1:.4f} AUC={auc:.4f} th={th:.3f} rec={rec:.4f} prec={prec:.4f}")
        print(f"  time={(time.time()-start)/60:.1f} min")

        np.savez_compressed(os.path.join(OUTPUT_DIR, f'{self.dataset}_dualtime_oof.npz'),
                           oof_proba=oof, flags=flags, f1=f1, auc=auc)
        return oof


if __name__ == '__main__':
    data = np.load(os.path.join(OUTPUT_DIR, 'sgcc_preprocessed.npz'))
    X_seq = data['X_seq']; flags = data['flags']
    stat = data.get('stat_features', np.zeros((len(flags),0)))

    trainer = DualTimeTrainer(epochs=30, batch_size=32, lr=3e-4)
    oof = trainer.train(X_seq, flags, stat_features=stat)
