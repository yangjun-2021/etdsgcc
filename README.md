# ETD-SGCC: Electricity Theft Detection

Multi-expert stacking framework (GBDT + TCN + Informer) for electricity theft detection on smart-meter data. Supports two datasets: **SGCC** (~42k users × 1035 days) and **OEDI** (building-level hourly load).

## Project Structure

```
etdsgcc/
├── config.py                     # Central configuration (hyperparameters, paths, SEED=42)
├── requirements.txt              # Python dependencies
├── run_pipeline.py               # Unified entry point (--dataset sgcc|oedi|both)
├── run_meta_v2.py                # Meta-learner v2 from cached OOFs
├── src/
│   ├── data/                     # Preprocessing (SGCC, OEDI, fold-aware)
│   ├── features/                 # Statistical, FFT/EMD, autoencoder, GAN, YoY, TopConf
│   ├── models/                   # PyTorch: TCN, Informer, AMST-Net, ADL-Net, Patch Transformer, etc.
│   ├── training/                 # Trainers: ExpertA/B/C, MetaLearner, ImprovedMetaLearner v2
│   ├── evaluation/               # Metrics, conformal calibration, ensemble diversity
│   └── utils/                    # seed_everything(), label_noise, threshold helpers
├── experiments/                  # ~160 experiment scripts (train/screen/retrain/validate)
├── figures/                      # Architecture diagrams (SVG + PNG + drawio)
├── supplementary/                # Supplementary data for paper reproducibility
│   ├── oof_pool/                 # 46-channel OOF prediction probability pool
│   ├── ablation/                 # Ablation metrics, per-fold results, significance tests
│   ├── labels/                   # Original + cleaned labels (v1, v3)
│   └── tables/                   # Main results, threshold trade-off, subgroup analysis
├── docs/                         # Pipeline architecture diagram
└── AGENTS.md                     # Developer guide for agents
```

## Installation

```bash
pip install -r requirements.txt
```

## Usage

```bash
# Full pipeline (SGCC)
python run_pipeline.py --dataset sgcc

# Fast iteration (cached preprocessing + cached OOFs)
python run_pipeline.py --dataset sgcc --skip-preprocess

# Meta-learner v2 only, from cached OOFs
python run_meta_v2.py --dataset sgcc --label-source original

# OEDI dataset
python run_pipeline.py --dataset oedi
```

Additional entry points: `run_pipeline.py --amst|--foundation|--causal|--contrastive|--multiscale-cnn|--meta-v2`

## Architecture

Two-stage cascade with stacking meta-learner using 5-fold cross-validation with out-of-fold (OOF) predictions:

1. **Expert A (GBDT)**: LightGBM + XGBoost + CatBoost ensemble on hand-crafted statistical features
2. **Expert B (TCN+Leaf)**: Temporal Convolutional Network on raw consumption sequences, fused with GBDT leaf embeddings
3. **Expert C (optional)**: Informer or MultiScaleCNN1D
4. **Meta-Learner v2**: Greedy forward ensemble selection + NNLS weighting over a 46-channel OOF pool, with BDR/FPR-constrained threshold optimization

Trained experts cache OOF probabilities as `.npz` files in `output/`; the meta-learner loads these caches instead of retraining.

## Supplementary Data

Open-access materials for paper reproducibility are in [`supplementary/`](supplementary/):

| Directory | Contents | Key Files |
|-----------|----------|-----------|
| `oof_pool/` | 46-channel OOF signal pool | `sgcc_expert_a.npz`, `sgcc_expert_b.npz`, `final_blend_best_oof.npz` |
| `ablation/` | 6-stage ablation with fold-aware metrics | `ablation_summary.csv`, `per_fold_metrics.csv`, `significance_tests_corrected.csv` |
| `labels/` | Original & consensus-cleaned labels | `cleaned_labels_v1.npz` (744 flips), `cleaned_labels_v3.npz` (673 flips) |
| `tables/` | Main results, threshold trade-off, subgroup analysis | `table_main_results.csv`, `recall_f1_frontier_fast.csv` |

See [`supplementary/README.md`](supplementary/README.md) for detailed documentation and loading instructions.

## Baseline Results

| Label Source | F1 | AUC | Recall | Precision |
|-------------|-----|------|--------|-----------|
| Original (honest) | 0.8657 | 0.9874 | 0.8382 | 0.8951 |
| Cleaned v3 (committee) | 0.8561 | — | — | — |

The inflated F1≈0.95 previously reported was caused by circular label cleaning using self-consensus labels. All supplementary data includes original labels for honest evaluation.
