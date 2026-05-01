# Scoping: scene-generation review for InverseModel x_accel and pos_z

## Why we're here

PR #4 (Phase 2 of `inverse_model_input_repr.md`, the CNN diagnostic) closed
with no perceiver swap beating the PCA50 baseline:

| run                | device | BN  | lr   | mean R² |
|--------------------|:------:|:---:|-----:|--------:|
| v1 (PCA50 reference)| —     | —   | —    | +0.499  |
| v1 CNN baseline    | CPU    | off | 1e-3 | +0.499  |
| v2 BN+lr=3e-3      | MPS    | on  | 3e-3 | +0.099  |
| v3 BN+lr=1e-3      | MPS    | on  | 1e-3 | +0.096  |
| v4 no-BN, MPS      | MPS    | off | 1e-3 | −0.001  |

Spec row-3 fired:

> Mean R² no better than Phase 1 → perceiver isn't the bottleneck; suspect data
> size / scene diversity / 3-frame stride → escalate to scene-generation review.

This doc is that escalation. Goal: pick the *one* scene-gen lever most likely
to lift the two failing dims to the spec target (≥ 0.4 each).

## What's actually gating each dim

Per-dim val R² from the v1 run (300 val scenes, 5 observable dims):

| dim       | R²     | What's needed for the perceiver to recover it |
|-----------|-------:|---|
| pos_x     | +0.963 | single-frame centroid readout |
| pos_y     | +0.750 | single-frame, partially gated by pillar occlusion |
| pos_z     | +0.297 | small vertical centroid drift across frames |
| linvel_x  | +0.497 | first-difference between frames |
| **x_accel** | **−0.013** | second-difference: 3-frame curvature visible in pixels |

`x_accel` (and to a smaller extent `pos_z`) is the binding constraint. Both
are inherently temporal, so the candidate levers are all about extracting more
temporal signal from the rendered scene.

## A first-principles look at why x_accel is unrecoverable now

Current generative parameters (from `config.yaml` and `scene_generator.py`):

- `linvel_x ∼ U(−8, +8)` m/s
- `x_accel ∼ U(−15, +15)` m/s²
- `n_timesteps = 30` (PyBullet default 240 Hz → simulation length **125 ms**)
- 3 frames at `t ∈ {0, 5, 15}` (i.e. 0 / 21 ms / 62.5 ms)
- Camera FOV 90°, distance 3 m, render 64×64 → ~6 m horizontal extent →
  **1 pixel ≈ 9.4 cm**

Position over time: `x(t) = v₀·t + 0.5·a·t²`.
At t = 62.5 ms with typical |a| = 7.5 m/s², the *acceleration-only*
displacement is:

    Δx_a = 0.5 · 7.5 · 0.0625² ≈ 0.0146 m ≈ **0.16 pixel**

Even at the extreme |a| = 15 m/s², that's 0.32 pixel — well below pixel
quantization. So x_accel's *visual signature* is sub-pixel at the current
3-frame stride.

The 3-frame setup is supposed to disentangle linvel from accel by linear
inversion. With frames at (t₁, t₂):

    [pos(t₁)]   [t₁    ½t₁²] [v₀]
    [pos(t₂)] = [t₂    ½t₂²] [a ]

The determinant is `½·t₁·t₂·(t₂ − t₁)`. For (t₁, t₂) = (5/240, 15/240) = (21 ms, 62.5 ms):

    det = ½ · 0.0208 · 0.0625 · 0.0417 ≈ 2.7 × 10⁻⁵

With pixel-quantization noise of σ_pos ≈ 0.094 m (one pixel), the inversion
amplifies it to **σ_a ≈ σ_pos / det ≈ 3470 m/s²** — two orders of magnitude
larger than the [−15, +15] generative range. **x_accel is fundamentally
unrecoverable at the current stride and resolution**, even with a perfect
perceiver.

## How each candidate lever moves the σ_a noise floor

Doubling frame times scales `det` by 8× (cubic in window length); doubling
resolution scales `σ_pos` by ½. Effects on σ_a:

| change                                              | (t₁, t₂)       | det       | σ_pos    | **σ_a**     | a-range covered? |
|-----------------------------------------------------|---------------|----------:|---------:|------------:|:----------------:|
| current                                             | 21, 62.5 ms   | 2.7e-5    | 0.094 m  | 3470 m/s²   | no               |
| even spacing within current sim                     | 62.5, 125 ms  | 2.4e-4    | 0.094 m  | 385 m/s²    | no               |
| extend `n_timesteps=60`, even spacing               | 125, 250 ms   | 2.0e-3    | 0.094 m  | 48 m/s²     | borderline       |
| extend `n_timesteps=120`, even spacing              | 250, 500 ms   | 1.6e-2    | 0.094 m  | 6 m/s²      | **yes**          |
| 128×128 render + extend to 60                       | 125, 250 ms   | 2.0e-3    | 0.047 m  | 24 m/s²     | borderline       |
| 128×128 render + extend to 120                      | 250, 500 ms   | 1.6e-2    | 0.047 m  | 3 m/s²      | **yes**          |

Headline: **window length is the cheapest lever**. Resolution helps but
matters less than time scale (linear vs. cubic in σ_a). x_accel becomes
cleanly recoverable once frames span ≥ 250 ms, i.e. 60+ PyBullet timesteps.

### Constraint introduced by extending the window

At `n_timesteps = 120` and `linvel_x = 8` m/s, the object travels
`8 × 0.5 = 4 m`, which exits the ~6 m frame extent. Two ways to keep the
object on-camera:

1. **Tighten initial-velocity range** to e.g. `linvel_x ∼ U(−3, +3)` m/s.
   Risks reducing `linvel_x` R² (it's currently 0.497 partly because v₀ has a
   wide range).
2. **Pull camera back** for a wider FOV / larger frame extent. Doubling the
   distance to 6 m doubles the extent to ~12 m at FOV 90, but halves angular
   resolution per object → loses the resolution gain we're trying to make.

Best balance is probably option (1) plus a modest `n_timesteps` bump
(e.g. 60–80).

## Recommendation: cheapest experiment first

Start with **A** (just stride changes). It's the only lever that doesn't
require re-baselining the rest of the pipeline.

### A. Extend simulation + even-spacing 3 frames (one PR)

- `config.yaml`: `n_timesteps: 60`, `pp_early_frame: 20`, `pp_late_frame: 40`.
- Re-run `scripts/eval_pp.py` (the v1 PCA-MLP harness) end-to-end with the
  fresh fixture. **No model changes.**
- **Pass criterion**: `x_accel` R² ≥ 0.2 *and* `pos_z` R² ≥ 0.4.
- **Cost**: ~30 min scene generation; one MLP train (≤ 5 min on the existing
  pipeline).

If A doesn't move x_accel, escalate to B.

### B. Tighten linvel_x + further stride extension

- `linvel_x ∼ U(−3, +3)` m/s, `n_timesteps: 120`, frames at `{0, 40, 80}`.
- Re-run eval_pp.
- **Pass criterion**: x_accel R² ≥ 0.4.
- **Risk**: linvel_x R² may regress because of the narrower range. Acceptable
  if the headline (x_accel) clears.

### C. Bump render resolution (only if A and B both fall short)

- `image_size: 128`. Touches `program_state` dimensionality, so the
  variance-fraction diagnostic in `neural_model.py` shifts. Needs review of
  the encoding-baseline numbers downstream (paper figures).
- Combine with A or B.

## Out of scope for this scoping doc

- Re-running the CNN. The perceiver isn't the lever (PR #4); the row-3 finding
  is independent of perceiver class.
- Architectural changes to InverseModel.
- More scenes (`n_scenes`). The CNN didn't show an overfitting gap, so n is
  not the binding constraint at this layer.
- Multi-object scenes. Touches `N_OBJECTS=1` assumption in many places; large
  refactor relative to the scene-gen-only changes above.

## Questions to resolve before A starts

- Does extending `n_timesteps` break any downstream assumption? Specifically:
  - Does `kinetic_energy` distribution shift in a way that breaks the median-
    split behavior label? (Probably yes — KE will be larger at the longer
    end-time.)
  - Are there code paths that assume frames at specific indices outside
    `pp_early_frame` / `pp_late_frame`?
- Do the existing paper figures depend on the current 30-timestep / 5-15 frame
  setup numerically, or only qualitatively?

If both questions are "yes, but tolerable", A is a one-PR change.
