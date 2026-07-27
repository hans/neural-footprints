# Spec: structural input-representation fix for the InverseModel

## Context

The `InverseModel` maps three rendered frames to inferred physics. Its activations + predictions get projected into the simulated neural population, so every "abstract physics in neural activity" claim downstream is bounded by how good this model is.

`specs/pixel_pca_dim_sweep.md` ran the cheap diagnostic (sweep `pixel_pca_dim ∈ {50, 200, 500}`, fixed everything else) and **falsified the linear-PCA-capacity hypothesis**. Result:

| pixel_pca_dim | mean R² | pos_x  | pos_y  | pos_z  | linvel_x | x_accel | val MSE | early stop |
|---------------|---------|--------|--------|--------|----------|---------|---------|------------|
| 50            | +0.499  | +0.902 | +0.736 | +0.282 | +0.678   | −0.104  | 0.497   | epoch 56   |
| 200           | +0.361  | +0.802 | +0.651 | +0.190 | +0.362   | −0.202  | 0.649   | epoch 55   |
| 500           | +0.184  | +0.489 | +0.433 | +0.045 | +0.023   | −0.069  | 0.802   | epoch 53   |

(See `outputs/pca_dim_sweep.json` and `scripts/sweep_pixel_pca_dim.py`.)

The signal is unambiguous: more PCA capacity *actively hurts*. With ~1700 train scenes and a 256-wide first MLP layer, widening the input space lets the model overfit train noise faster — val MSE climbs monotonically (0.50 → 0.65 → 0.80) and best val loss plateaus by epoch ~5 in every setting. The two dims the previous spec wanted to lift (`pos_z`, `x_accel`) get *worse*, not better.

**Conclusion: the bottleneck isn't basis width; it's basis content.** Top-PC linear projections of three concatenated frames are dominated by lighting/shape variance and don't expose the temporal-difference signal that vertical-centroid drift and second-difference acceleration actually live in. To unblock the InverseModel, we need a structurally different input — not a wider one.

## Hypothesis

Two structural alternatives, **sequenced cheap-first**:

1. **Frame-difference features.** Replace the three concatenated whitened PCAs of raw frames with `(pca(t0), pca(t_mid) − pca(t0), pca(t_late) − pca(t_mid))` at the same per-frame PCA dim (50). The MLP no longer has to learn "subtract these for me from a high-variance basis"; velocity-like and accel-like signals are first-class input dimensions. Same input width, same MLP, same training schedule — the *only* change is the linear pre-mix.

2. **CNN on the raw 3-frame stack.** A small convolutional perceiver on the `(3, 64, 64, 4)` raw render stack. Skips PCA entirely, so the basis is no longer dominated by lighting/shape variance. Heavier change: new module, new training loop integration.

The frame-difference experiment is one PR's worth of work and either fixes the problem or sharpens the case for the CNN. The CNN only happens if (1) doesn't reach the success criterion.

## Procedure

### Phase 1 — Frame-difference features

- **What changes.** A new feature builder, e.g. `build_pp_diff_features(scenes, pixel_pca_dim=50)` in `analyses/predictive_processing.py`, returning `concat(pca(t0), pca(t_mid)−pca(t0), pca(t_late)−pca(t_mid))`. PCA is fit on `t0` only and applied to all three frames so the difference is in a shared basis. Whitening preserved.
- **Held fixed.** Same `data/scenes.npz`, same `InverseMLPNet` (256→256→128→5), same training schedule (300 epochs, lr 1e-3, dropout 0.05, patience 50), same train/val split (`random_state=42`, val_frac=0.15), same `pixel_pca_dim=50`.
- **Per-run record.** Per-dim val R² (all 5 observable dims), mean R², `r2_inferred` (encoding), `r2_resid_inferred` (residual), early-stop epoch, val MSE, wall-clock.

### Phase 2 — CNN perceiver (only if Phase 1 doesn't pass)

- **What changes.** A new `InverseCNN` in `analyses/predictive_processing.py`: small conv stack on `(B, 3, 4, 64, 64)` (3 frames as channels-of-channels, RGBA), feeding into the same head dims (256→256→128→5). Reasonable starting recipe: per-frame conv tower (3×3 convs, 32→64→128 channels with stride-2 downsamples), per-frame global-avg-pool, concat across the three frames, then the existing MLP head. Total params target: < 2M to stay sample-efficient at n=1700.
- **Held fixed.** Same scenes, same head architecture, same training schedule, same split, same eval metrics.
- **Per-run record.** Same metrics as Phase 1.

## Decision matrix

| outcome (Phase 1) | interpretation | next move |
|-------------------|----------------|-----------|
| mean R² ≥ 0.65 **and** every dim ≥ 0.4 (incl. x_accel ≥ 0.4) | frame-diff exposes the temporal signal cleanly; linear basis is sufficient when correctly oriented | wire `build_pp_diff_features` into `gen_features` / `config.yaml`; ship; **skip Phase 2** |
| mean R² climbs (≥ ~0.55) and pos_z lifts but x_accel still ≤ 0.2 | velocity-scale info is in the diff basis but second-difference still gets washed; PCA may be too coarse for accel | run Phase 2 — CNN keeps spatial detail PCA throws away |
| mean R² flat or worse than 0.499 baseline | the diff basis isn't carrying the right invariances either; linear pre-mix is fundamentally insufficient | run Phase 2 |

| outcome (Phase 2) | interpretation | next move |
|-------------------|----------------|-----------|
| mean R² ≥ 0.65, every dim ≥ 0.4 | CNN is the right perceiver | wire it in; ship |
| Mean R² climbs but x_accel still struggles | accel may genuinely require more frames, longer time window, or higher-fps sampling — this is a generative-process question, not a perceiver-architecture one | revisit 3-frame timing in `config.yaml` (out of scope here); document as a separate spec |
| Mean R² no better than Phase 1 | perceiver isn't the bottleneck; suspect data size / scene diversity / 3-frame stride | escalate to scene-generation review |

## Implementation

Two off-pipeline diagnostic scripts in `scripts/`, modeled on `scripts/sweep_pixel_pca_dim.py` (which is itself modeled on `scripts/eval_pp.py`):

- `scripts/eval_pp_framediff.py` — Phase 1. Reads `data/scenes.npz` if present, generates a fresh fixture otherwise (no Snakemake coupling). Builds frame-diff features, fits one `InverseModel`, records all metrics. Single JSON output `outputs/pp_framediff.json`. Prints summary table.
- `scripts/eval_pp_cnn.py` — Phase 2. Same shape: standalone, fixture-building, single JSON output `outputs/pp_cnn.json`, summary table. Only written if Phase 1 doesn't pass.

Each script is independently runnable on a clean checkout via `uv run`.

## Out of scope

- Changing `config.yaml`'s feature recipe — that's the *outcome* of the winning phase, not the experiment itself.
- Retuning MLP hidden_dim, dropout, learning rate, 3-frame timing, or scene generative ranges — confounds the input-representation question.
- Increasing `n_scenes` — orthogonal axis. If both phases plateau, that's the signal to address it as a separate spec.
- Running the full Snakemake pipeline at each candidate. Per-dim R² + inferred-physics encoding is sufficient for the decision.
- Comparing against the pixel_pca_dim sweep numbers as anything other than a sanity baseline at `pixel_pca_dim=50`.

## Success criterion

The combined experiment succeeds if it produces a structurally-different input representation that gets the InverseModel to **mean R² ≥ ~0.65 with no individual dim below ~0.4**, ready to wire into `config.yaml`. Phase 1 is the cheap shot; Phase 2 is the structural fallback. Either landing is a clear next PR; a flat result from both is the signal to escalate to the generative process or training set size.
