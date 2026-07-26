# ADL-Net Design: Anomaly Dictionary Learning Network for SGCC ETD

## Motivation
Existing AMST experiments keep hitting a wall (~0.33 F1 on 1k subsets) because:
1. Diffusion augmentation fails when real theft samples are too scarce.
2. Treating AMST as a pure supervised classifier ignores the natural anomaly-detection structure of the task.
3. Using Expert-A prior leaks information from another model (GBDT) and is not a valid pure end-to-end solution.

ADL-Net reframes the problem as **unsupervised normal-pattern dictionary learning + supervised contrastive classification**.

## Core Idea
- Build a **learnable dictionary of normal electricity-consumption patterns** from all users (mostly normal).
- For each user, reconstruct its load profile as a sparse combination of dictionary atoms.
- Anomalous / theft users will have **higher reconstruction residual** and **different reconstruction-coefficient distribution** than normal users.
- A small CNN+Transformer encoder maps the original series, residual, and difference series into an embedding space.
- A **contrastive module** (MoCo-style) pulls embeddings of augmented views of the same user together, and pushes away anomalous users.
- A final classifier combines: encoder embedding + dictionary residual + sparse-code statistics.

## Architecture

### Input
- `x`: [B, C, T]  SGCC multi-channel series (value, impute_mask, dow, etc.)
- `x_res`: [B, C, T] reconstruction residual from dictionary
- `x_diff`: [B, C, T] first-order temporal difference

### Encoder: `ADLEncoder`
- 1D CNN front-end (channel expansion + downsampling)
- Positional encoding + Transformer block
- Global average pooling + MLP projection

### Dictionary: `SparseDictionary`
- `D`: [K, C*T] learnable dictionary atoms (K atoms)
- Given `x`, reshape to [B, C*T], solve for sparse code via differentiable soft-thresholding (ISTA-style, 5 iterations)
- Reconstruct: `x_hat = code @ D`
- Residual: `x - x_hat`

### Contrastive Head: `MoCoHead`
- Query encoder = main encoder
- Key encoder = EMA of query encoder
- Queue of negative samples from previous batches
- NT-Xent loss

### Classifier Head
- Input: [encoder_embedding, residual_stats, code_stats]
- MLP -> binary logit

## Losses
1. `L_recon`: MSE on normal users only (ignore labels to avoid leakage). Reconstruction should be good for normal users.
2. `L_cls`: Binary Focal loss with label smoothing.
3. `L_contrast`: NT-Xent / MoCo loss between augmented views of the same user. Positive pairs: same user different augmentations. Negatives: queue of other users.

## Why this can surpass 0.90
- **Explicitly models normality**: the dictionary learns what normal consumption looks like. Theft is, by definition, deviation from normal.
- **Sparse code as interpretable anomaly feature**: the coefficient distribution of anomalous users differs from normal users.
- **Contrastive learning uses unlabeled data**: even with 8.5% positives, the other 91.5% provide strong self-supervision.
- **No synthetic data generation needed**: avoids diffusion collapse.
- **No GBDT prior**: truly end-to-end.

## Implementation Plan
1. `src/models/adl_net.py` — model definition
2. `src/training/adl_trainer.py` — training loop with 5-fold OOF
3. `experiments/adl_quick_test.py` — quick subset verification
4. Run 1000-sample GPU experiment with ml environment
5. Scale to full SGCC if promising

## Hyperparameters (initial)
- K (dictionary atoms): 256
- sparse_lambda: 0.1
- encoder d_model: 256
- encoder n_layers: 4
- n_heads: 8
- contrast temperature: 0.07
- queue size: 4096
- lambda_recon: 1.0
- lambda_cls: 1.0
- lambda_contrast: 0.5
- Focal: alpha=0.75, gamma=2.0

## Expected Behavior on Small Subset
- If dictionary is meaningful, normal users should reconstruct with low error.
- Residual signal should be a strong auxiliary feature for the classifier.
- Even with 100 thefts / 1000 samples, the contrastive signal from 900 normal users should help.

