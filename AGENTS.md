# AGENTS.md — ETD-SGCC

## Project Overview

**ETD-SGCC** is a research codebase for **electricity theft detection** (non-technical loss detection) on smart-meter data. It implements a multi-expert stacking framework and supports two datasets:

- **SGCC** (State Grid Corporation of China): ~42k users × 1035 days of daily consumption, 8.53% theft rate, 25.6% missing values (`data/raw_data.csv`).
- **OEDI**: building-level hourly load data with 6 simulated theft types (`data/df.csv`).

The repository has three intertwined parts:

1. **`src/` — the core ML library**: data preprocessing, feature engineering, models, trainers, evaluation.
2. **Experiment / runner scripts** (`run_pipeline.py`, `run_meta_v2.py`, `experiments/`, and many root-level `run_*.py` / `retrain_*.py` scripts): drive training and ensemble experiments.
3. **Paper production toolchain** (`paper/`, `scripts/`, `figures/`, root-level `audit_*.py` / `fix_*.py` / `build_*.py` scripts): generates, audits, and fixes a Chinese journal manuscript (`BSC_ETD论文_v*` .docx, targeted at 《电网技术》) using `python-docx` + `lxml`.

### Architecture (core pipeline)

Two-stage cascade plus a stacking meta-learner, all using 5-fold cross-validation with out-of-fold (OOF) predictions:

1. **Expert A (GBDT)**: LightGBM + XGBoost + CatBoost ensemble on hand-crafted statistical features (`src/training/expert_a.py`). "Expert" is just a naming convention — this is **not** a Mixture-of-Experts model.
2. **Expert B (TCN+Leaf)**: Temporal Convolutional Network on the raw consumption sequence, fused with GBDT leaf embeddings (`src/training/expert_b.py`, `src/models/models.py`). Alternative Expert B implementations exist: AMST-Net, ADL-Net, etc.
3. **Expert C (optional)**: Informer (`expert_c.py`) or MultiScaleCNN1D (`expert_c_multiscale.py`).
4. **Meta-Learner**: cost-sensitive XGBoost stacking (`src/training/meta_learner.py`), or **ImprovedMetaLearner v2** (`src/training/meta_learner_v2.py`) which does greedy forward ensemble selection + NNLS weighting over a pool of cached OOF predictions, with threshold optimization (best-F1 / BDR / FPR-constrained).

Heavy experts are trained once and their OOF probabilities are **cached as `.npz` files in `output/`** (e.g. `sgcc_expert_a.npz`); downstream meta-learner work loads these caches instead of retraining.

## Build and Run Commands

There is no package build (`pyproject.toml`/`setup.py` do not exist); everything runs as plain scripts from the repo root.

```bash
pip install -r requirements.txt        # numpy, pandas, scipy, scikit-learn,
                                       # lightgbm, xgboost, catboost, torch, matplotlib
```

- Run the full pipeline: `python run_pipeline.py --dataset sgcc` (also `oedi` / `both`)
- Fast iteration (uses cached preprocessing + cached expert OOFs): `python run_pipeline.py --dataset sgcc --skip-preprocess`
- Meta-learner v2 only, from cached OOFs: `python run_meta_v2.py --dataset sgcc --label-source original` (use `--pool v3voter,v3clean` to restrict to the untainted v3 family)
- Autoresearch benchmark loop: `./autoresearch.sh` (prints `METRIC f1=<number>` lines; see `autoresearch.md`)
- Other entry points: `run_pipeline.py --amst | --foundation | --causal | --contrastive | --multiscale-cnn | --meta-v2`; assorted root scripts like `run_expert_c_multiscale.py`, `retrain_expert_c.py`.

Additional tooling not listed in `requirements.txt` but required by the paper/audit scripts: `python-docx`, `lxml` (used via `from docx import ...`).

Python 3.13 is used locally (older caches show 3.11/3.12 also worked). Default shell on Windows here is **PowerShell (pwsh)**, but `autoresearch.sh` is a bash script — run it under Git Bash / WSL, or invoke `python run_meta_v2.py --dataset sgcc --label-source original` directly and parse the `METRIC` lines yourself. All scripts set `KMP_DUPLICATE_LIB_OK=TRUE` to work around OpenMP duplication; in pwsh use `$env:KMP_DUPLICATE_LIB_OK='TRUE'` before running.

## Code Organization

- `config.py` — **central configuration**: dataset paths, per-dataset hyperparameters (`SGCC_CONFIG`, `OEDI_CONFIG`), `SEED=42`, `DEVICE` (cuda if available), `N_FOLDS=5`. New experiments should read hyperparameters from here, not hardcode them.
- `src/data/` — preprocessing (`preprocess_sgcc.py`, `preprocess_oedi.py`), synthetic anomaly generation (`synthetic_anomalies*.py`, `ts_augment.py`, `diffusion_augment.py`), and fold-aware preprocessing.
- `src/features/` — feature engineering: advanced stats, FFT/EMD, autoencoder, GAN, YoY, deep features, TopConf.
- `src/models/` — PyTorch models: TCN, Informer, TFT, AMST-Net, ADL-Net, foundation/contrastive/causal encoders, etc.
- `src/training/` — one trainer class per model (`ExpertATrainer`, `ExpertBTrainer`, `MetaLearner`, `ImprovedMetaLearner`, …), unified across datasets via the config dict (`trainer = ExpertATrainer(dataset='sgcc'|'oedi')`).
- `src/evaluation/evaluate.py` — `evaluate_dataset(name, results, output_dir)` reports F1/recall/precision/AUC and writes figures to `output/`.
- `src/utils/utils.py` — `seed_everything()`, `best_f1_score()` threshold search helpers.
- `experiments/` — ~160 one-off experiment scripts; `experiments/archive/` holds retired ones.
- `output/` — all run artifacts: cached preprocessed data, OOF `.npz` caches, trained models, logs, figures. **Git-ignored and treated as read-only inputs by the autoresearch loop.**
- `data/` — raw CSVs (git-ignored).
- `docs/` — pipeline architecture diagram (draw.io).
- `paper/`, `scripts/`, `scripts_dev/`, `figures/`, root `audit_*.py` / `fix_*.py` — manuscript generation/audit/fix pipeline (Chinese journal paper, versioned `v21`…`v31` docx files). `scripts/` holds paper build/fix scripts; `scripts_dev/` holds diagram generation utilities.

## Conventions

- **Reproducibility**: always call `seed_everything(SEED)` at entry; all trainers take the dataset config dict, so SGCC/OEDI share one code path.
- **OOF caching contract**: experts save `{dataset}_expert_{a,b,c}.npz` with at least an `oof_proba` array (expert A also stores `flags`); meta-learners auto-discover OOF files in `output/`. Do not modify or delete cached OOF files — many scripts depend on them.
- **Runner scripts insert the repo root into `sys.path`** and import via `from config import ...` / `from src.xxx import ...`; run them from the repo root.
- Experiment tracking: results are logged in `autoresearch.jsonl`; `autoresearch.md` documents the objective, constraints, and what has been tried. Follow that format when running optimization experiments.
- Code comments and docstrings in `src/` are primarily **English** (with occasional Chinese comments in feature-engineering code); the manuscript under `paper/` is **Chinese**.
- One-off scripts are expected and numerous; keep new experiment scripts self-contained, named after what they test, and place substantial reusable logic in `src/` instead.

## Testing

There is **no pytest/unittest suite**. Verification is done by:

- Smoke / quick-test scripts in `experiments/` (e.g. `amst_smoke_test.py`, `adl_quick_test.py`, `causal_quick_test.py`) — run the specific one relevant to your change.
- End-to-end pipeline runs on cached data (`python run_pipeline.py --dataset sgcc --skip-preprocess`) checking reported F1/AUC.
- The autoresearch benchmark (`./autoresearch.sh`) for meta-learner changes, which must finish in <10 minutes per run.
- For paper scripts: the root `audit_*.py` validators, which emit `*_output.md` reports.

When changing model/training code, run the corresponding quick test or a cached-OOF pipeline run and confirm metrics did not regress before considering the task done.

## Known Pitfalls / Security & Data Notes

- **Label leakage history**: the F1=0.9557 result was inflated — evaluation used consensus-cleaned labels (`cleaned_labels_v1.npz['y_clean']`, 744 flipped labels) derived from the same model family. The honest baseline on **original** labels is **F1≈0.8657** (see `experiments/cleaned_vs_original_label_report.md`). Always state which label source (`--label-source original|cleaned|cleaned_v3`) a metric uses.
- **V3 campaign (2026-07-18)**: `output/cleaned_labels_v3.npz` holds labels refined by an *independent, untainted* committee (4 original-label voters, no stacked-OOF prior features; 673 flips). On these labels the full stack scores F1=0.8561 (vs 0.8657 original, vs ~0.95 self-refined v1) — i.e. ~9–10pp of the v1 number is self-agreement. Untainted-only pool: 0.65 (original) / 0.73 (v3). Estimated original-label F1 ceiling ≈0.90 due to label noise. Note: `strong_gbdt_prior*.npz` targets were original labels (taint is feature-side via stacked OOFs), and `sgcc_expert_a.npz['flags']` on disk now equals `y_orig` after cache regeneration.
- **Compute**: mostly CPU-only; deep expert training (TCN/AMST/Informer/Patch Transformer) is prohibitively slow on CPU — prefer cached OOFs. `config.py` auto-selects CUDA if present.
- **Do not edit** `data/` raw CSVs or cached `output/*.npz` files; they are shared inputs to many scripts.
- Some paper scripts reference absolute paths outside the repo (e.g. `C:\Users\yangj\Desktop\论文`); they will only run on the author's machine.
- The repo is a personal research workspace: large binary artifacts, generated docx versions, and `__pycache__` are present; data/output/catboost_info are git-ignored. No CI, no secrets, no deployment process — results are "deployed" into the manuscript.
