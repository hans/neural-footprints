# Spec: pixel_pca_dim sweep — improve overall InverseModel quality

## Context

The `InverseModel` is the cognitive perceiver in this paper's setup: it maps three rendered frames to inferred physics, and the projected neural activity is built from its hidden activations. **A behaviorally weak inverse model weakens every downstream claim** — the inferred-physics signal that lives in neural activity is only as faithful as the model that produced it.

Current per-dim R² on the val set (mean = **0.516**):

| dim | R² |
|------|------|
| pos_x | 0.935 |
| pos_y | 0.727 |
| pos_z | **0.298** |
| vel_x | 0.729 |
| x_accel | **−0.108** |

Two dims under-perform substantially: `pos_z` (vertical position) and `x_accel` (acceleration, newly added in PR #2). Both are likely starved of the signal they need from the current input representation. The shared diagnostic question is whether bumping input capacity fixes either or both — and if not, what does.

This is not a single-variable diagnostic; it is the first step in a sequence of input-representation experiments aimed at making the inverse model genuinely capable. The cheaper experiments come first.

## Hypothesis

The bottleneck is the *input representation*, not the MLP or the targets. The model sees `pca(frame_t0) ⊕ pca(frame_t5) ⊕ pca(frame_t15)` at `pixel_pca_dim = 50`. Top-50 PCs of three 64×64 RGBA frames are dominated by lighting / shape / color variance, which have far more pixel variance than the centroid-position drifts (vertical motion, second-difference motion) that pos_z and x_accel encode.

Two underperforming dims, two predictions:

- **pos_z** needs the basis to carry sub-PC vertical-centroid resolution. Vertical motion across the 0.4-unit z range maps to maybe ~10 pixels in a 64×64 image; that signal can be lost in the leading PCs.
- **x_accel** needs the basis to preserve frame-to-frame displacement detail finely enough that a second-difference is computable. Linear PCA on three concatenated frames with low rank may not.

If the hypothesis is right, both should improve together as `pixel_pca_dim` rises.

## Procedure

A clean sweep over `pixel_pca_dim`, holding everything else fixed.

- **Settings.** 50 (current), 200, 500. Add 1000 if mean R² is still climbing at 500.
- **Held fixed.** Same `data/scenes.npz`, same MLP architecture (`InverseMLPNet`, 256→256→128→5), same training schedule (300 epochs, lr 1e-3, dropout 0.05, patience 50), same train/val split (`random_state=42`, val_frac=0.15). Only `pixel_pca_dim` varies.

### Per setting, record

- Per-dim R² for all 5 observable dims (`pos_x, pos_y, pos_z, vel_x, x_accel`).
- Mean per-dim R² — the headline behavioral metric.
- Inferred-physics encoding R² (`r2_inferred` from `analyses/encoding.py`) — how much of the inferred signal lives in the projected neural activity.
- Residualized inferred-physics R² (from `analyses/residual.py`) — confirms the headline residual collapse still holds at the new operating point.
- Wall-clock training time.

## Decision matrix

| outcome | interpretation | next move |
|---------|----------------|-----------|
| Mean R² climbs meaningfully (≥ 0.65) and pos_z + x_accel both lift to ≥ 0.5 | linear basis was capacity-starved, MLP is fine | bump `pixel_pca_dim` in `config.yaml`, ship |
| Mean R² climbs but x_accel stays ≤ 0 | linear basis can't carry second-difference info regardless of dim; pos_z may benefit but x_accel won't | swap to CNN on raw 3-frame stack, OR try frame-difference features `(pca(t0), pca(t1)−pca(t0), pca(t2)−pca(t1))` as a structural intermediate |
| Per-dim R² is largely flat across all settings | basis isn't the bottleneck — the MLP itself or the time-window choice is | revisit MLP capacity / 3-frame timing / scene generative ranges |
| Some dims climb, some plateau | partial PCA-starvation; mixed verdict | use the per-dim slope to choose between bumping PCA dim further vs. structural change |

## Implementation

Off-pipeline diagnostic, not a Snakemake rule. Parameterized standalone script in `scripts/`, modeled on `scripts/eval_pp.py`:

- Reads `data/scenes.npz` once if available; otherwise generates a fresh small scenes fixture so the script is runnable on a clean checkout (must not couple to Snakemake-managed cache).
- For each `pixel_pca_dim` setting: rebuild three-frame whitened PCA via `analyses.predictive_processing.build_pp_features(scenes, pixel_pca_dim=k)`, fit a fresh `InverseModel`, evaluate per-dim R² and downstream encoding/residual numbers.
- Single JSON output (e.g. `outputs/pca_dim_sweep.json`) with one entry per setting.
- Print a summary table at the end.

## Out of scope

- Changing `config.yaml`'s `pixel_pca_dim` default — that's the *outcome* of this sweep, not the sweep itself.
- CNN swap, frame-difference features — flagged in the decision matrix as next-tier options, not implemented here.
- Retuning MLP hidden_dim, dropout, learning rate, or 3-frame timing — confounds the input-capacity question.
- Running the full Snakemake pipeline at each setting — unnecessary; per-dim R² and inferred-physics encoding numbers are sufficient signal.

## Success criterion

The sweep succeeds if it produces a clear answer to one question: *can the InverseModel become behaviorally good (mean R² ≥ ~0.65, no individual dim below ~0.4) on the existing linear-PCA input, or does it need a structurally different input representation?* Either answer is informative and unambiguously routes the next PR.
