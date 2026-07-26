"""Retrain Expert C (Informer) with a smaller/faster config suitable for CPU."""
import os
import sys
import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

from config import OUTPUT_DIR, SEED
from src.utils.utils import seed_everything
from src.training.expert_c import ExpertCTrainer


def main():
    seed_everything(SEED)

    print("Loading cached preprocessed data and Expert A OOF...")
    pre = np.load(os.path.join(OUTPUT_DIR, 'sgcc_preprocessed.npz'))
    X_seq = pre['X_seq']
    flags = pre['flags']

    a = np.load(os.path.join(OUTPUT_DIR, 'sgcc_expert_a.npz'))
    oof_proba_a = a['oof_proba']

    print(f"  X_seq={X_seq.shape}, flags={flags.shape}, prior={oof_proba_a.shape}")

    # Smaller config for faster CPU training while keeping reasonable capacity
    trainer = ExpertCTrainer(
        dataset='sgcc',
        d_model=96,
        n_heads=4,
        num_layers=2,
        dropout=0.3,
        epochs=30,
        batch_size=64,
        lr=3e-4,
    )
    oof_proba_c = trainer.train(X_seq, flags, oof_proba_a=oof_proba_a)

    save_path = os.path.join(OUTPUT_DIR, 'informer_oof_v2.npz')
    np.savez_compressed(save_path, oof_informer=oof_proba_c, flags=flags)
    print(f"\nSaved retrained Informer OOF to {save_path}")


if __name__ == '__main__':
    main()
