# Per-block P contribution diagnostic

## Motivation

Under `zscore` block normalization, true physics `P` collapses to `R²_P = 0.0018`
(against an expected `> 0.10`) and adds nothing on top of X, even though P is
clearly linked to several of the inputs that drive neural activity. Under
`truncated_svd`, the same setup yields `R²_P = 0.037` with a meaningful `r²_P|X
= 0.073`. The norm choice is structurally changing where P-signal goes — but
the current pipeline does not expose this. The encoding/residualization outputs
only tell you what is observable from neural activity, not how each input block
carries P or how the chosen norm shapes its contribution.

This diagnostic decomposes neural variance into per-block contributions and
quantifies, per block, how much of that contribution is P-decodable. The
expected pattern under zscore: hidden_acts and inferred_physics have high
R²(P → block_signal) but tiny variance share; raw_frames and fwd_render
dominate variance but carry little P signal — so the product collapses
everywhere. Under truncated_svd, the variance shares equalize and the
P-carrying blocks survive into neural activity.

This is a one-off scientific diagnostic, not a pass/fail criterion. No changes
to `evaluation.py`.

## Inputs

Neural input concatenation is fixed by `scripts/gen_neural.py`:

```
neural_input = [raw_frames | fwd_render | hidden_acts | inferred_physics]
block_names  = [raw_frames, fwd_render, hidden_acts, inferred_physics]
```

True physics `P` is `scenes["physics_labels"]` (standardized with `StandardScaler`).
P is **not** a block — it enters neural activity only through correlations with
the four blocks.

## Analysis

For each block `B ∈ {raw_frames, fwd_render, hidden_acts, inferred_physics}`:

### 1. Feature-space P decodability (norm-agnostic)

Answers: "is P present in this block at all, before any normalization?"

- `r2_P_from_block_raw`: cross-validated ridge R² of P regressed on B.
  - For high-dim blocks (`raw_frames`, `fwd_render`), PCA-reduce B to
    `pixel_pca_dim` (from config) first; for low-dim blocks
    (`hidden_acts`, `inferred_physics`), use B directly.
  - Use `ridge_r2_per_neuron_fast` from `analyses.encoding`, scaling the per-
    target output up to the P dimensions (each physics label is a target).
  - Report per-physics-dimension R²s and their mean.

### 2. Block-wise neural contribution

Reconstructs each block's contribution to the noiseless neural signal and asks
how P-rich that contribution is and how much variance it carries.

Let `signal_b = normalize_b(centered_B) @ W_b.T` where `W_b` is the slice of `W`
along its post-normalization block boundary:
- Under `zscore`: `W_b` columns are `[cumsum(block_sizes)[b-1] : cumsum(block_sizes)[b])`.
- Under `truncated_svd`: `W_b` columns are
  `[cumsum(block_k_values)[b-1] : cumsum(block_k_values)[b])`.

Compute:
- `var_share_b`: block b's *attributed* share of total signal variance, defined
  to sum to 1 across blocks. For each neuron n,
  `var(total[:,n]) = Σ_b cov(signal_b[:,n], total[:,n])` (linearity of cov over
  decompositions). Summing over neurons:
  ```
  var_share_b = Σ_n cov(signal_b[:,n], total_signal[:,n])
              / Σ_n var(total_signal[:,n])
  ```
  Equivalently, `((signal_b - mean_b) * (total_signal - mean_total)).sum(axis=0).sum()
  / ((total_signal - mean_total) ** 2).sum(axis=0).sum() / (n_scenes - 1)`
  (the `/(n_scenes-1)` cancels in numerator and denominator).
  Blocks with strong cross-block correlation absorb some of each other's
  variance; this is correct behaviour and what makes the attribution sum to 1.
  *Also* report `var_share_independent_b = Σ_n var(signal_b[:,n]) / Σ_n var(total[:,n])`
  for the figure, since "this block's own variance" is also interpretable.
- `r2_P_from_block_signal_b`: cross-validated ridge R²(P → signal_b), averaged
  over physics dimensions.
- `effective_P_contribution_b = r2_P_from_block_signal_b × var_share_b`
  (using the attribution form so the contributions remain on the same scale as
  the total).

### 3. Sanity check

- `r2_P_from_total_signal`: R²(P → total_signal). Should match the encoding
  result `r2_P` within noise-induced shrinkage (encoding uses noisy
  `neural_activity`; this uses noiseless `total_signal`).

## Implementation

### `scripts/io_utils.py`

Extend `save_neural` / `load_neural` to persist:
- `block_sizes` (list[int]) — sizes of raw input blocks.
- `block_names` (list[str]) — human-readable labels.
- `block_k_values` (list[int] or None) — post-normalization sizes
  (`block_sizes` itself under zscore; `block_k_values` from neural_meta under
  truncated_svd).

Backward-compat: `load_neural` should default missing keys to `None`. Storing
empty arrays when the underlying value is `None` keeps `np.savez_compressed`
happy.

### `scripts/gen_neural.py`

Pass `block_names = ["raw_frames", "fwd_render", "hidden_acts", "inferred_physics"]`
into the metadata dict before calling `save_neural`.

### `analyses/p_block_contribution.py`

New module exposing one entry point:

```python
def run_p_block_contribution(
    *,
    neural_meta,            # from load_neural
    blocks_raw,             # dict[str, np.ndarray] keyed by block_name
    physics_labels,         # raw, will be standardized internally
    pixel_pca_dim,          # for high-dim block PCA reduction
    high_dim_blocks=("raw_frames", "fwd_render"),
    seed=42,
) -> dict
```

Returns a dict with these keys:
- `block_names: list[str]`
- `block_sizes: list[int]`
- `block_k_values: list[int]`
- `var_share: list[float]`
- `r2_P_from_block_raw: list[float]`  (mean over physics dims)
- `r2_P_from_block_signal: list[float]`
- `effective_P_contribution: list[float]`
- `r2_P_from_total_signal: float`
- `r2_P_per_physics_dim: dict[str, list[list[float]]]`  (per block, per phys-dim — for figure annotations / later inspection)
- `norm: str` (passed through from neural_meta)

Internals:
- Standardize physics with `StandardScaler` (matches encoding).
- Use `ridge_r2_per_neuron_fast(X, Y)` where `Y = physics_scaled`, `X = block features`.
  This treats each physics dimension as a target. Average across targets for
  the scalar reported value.
- Reconstruct `signal_b` by centering `blocks_raw[name]` with `neural_meta["means"]`
  sliced by `block_sizes`, normalizing per-block exactly as `neural_model.py`
  does, then multiplying by `W_b.T`.
  - **Preferred**: refactor `neural_model.py` to expose a helper that
    normalizes a single block under either norm (`_normalize_block_zscore` and
    `_normalize_block_truncated_svd`), and have `generate_neural_activity` call
    them. Reuse from this analysis.
  - **Acceptable fallback**: duplicate the per-block normalization logic inline
    in this analysis if refactoring risks regressing the existing pipeline.
    In that case, add a short comment in both places pointing to the other.
  - In either case, the existing snapshot tests / smoke run for both norms must
    still pass byte-for-byte.

### `scripts/run_p_block_contribution.py`

Snakemake script that:
1. Loads `scenes`, `neural_meta` (via `load_neural`), `pp_activations`, `forward_renders`.
2. Reconstructs the 4 raw input blocks identically to `gen_neural.py`
   (raw_frames concat, render slice via `render_indices`, hidden_acts,
   inferred_physics).
3. Calls `run_p_block_contribution`.
4. Saves JSON to `outputs/{norm}/p_block_contribution.json`.
5. Saves NPZ to `data/{norm}/p_block_plot_data.npz` with all numeric fields
   needed by the plotter.

### `scripts/plot_p_block_contribution.py`

Per-norm bar chart:
- X axis: 4 block names.
- 3 grouped bars per block: `var_share`, `r2_P_from_block_signal`,
  `effective_P_contribution`.
- Side annotation: `r2_P_from_total_signal` and the encoding-reported `r2_P`
  (read from the encoding plot_data NPZ to keep this rule's input contained,
  OR pass through results).
- Title: `f"P contribution by block — {norm} normalization"`.

### `scripts/plot_p_block_compare.py`

Cross-norm comparison figure (no `{norm}` wildcard). Reads both norms' plot
NPZs and emits `figures/p_block_contribution_compare.pdf`:
- 2 columns × 1 row, one panel per norm, sharing y-axis.
- Each panel: the per-norm bar chart described above.

### Snakefile

```python
rule p_block_contribution:
    input:
        scenes="data/scenes.npz",
        neural="data/{norm}/neural.npz",
        pp_activations="data/pp_activations.npz",
        forward_renders="data/forward_renders.npz",
    output:
        results="outputs/{norm}/p_block_contribution.json",
        plot_data="data/{norm}/p_block_plot_data.npz",
    script: "scripts/run_p_block_contribution.py"


rule plot_p_block_contribution:
    input:
        plot_data="data/{norm}/p_block_plot_data.npz",
    output:
        figure="figures/{norm}/p_block_contribution.pdf",
    script: "scripts/plot_p_block_contribution.py"


rule plot_p_block_compare:
    input:
        zscore="data/zscore/p_block_plot_data.npz",
        truncated_svd="data/truncated_svd/p_block_plot_data.npz",
    output:
        figure="figures/p_block_contribution_compare.pdf",
    script: "scripts/plot_p_block_compare.py"
```

Add `"figures/{norm}/p_block_contribution.pdf"` to `_ALL_FIGURES`. Add
`"figures/p_block_contribution_compare.pdf"` to `rule all`.

## Acceptance

After `uv run snakemake -j3`:

1. Both `outputs/{zscore,truncated_svd}/p_block_contribution.json` exist with
   the full schema above.
2. Both per-norm figures and the cross-norm compare figure render.
3. Sanity: `r2_P_from_total_signal` is within ~0.05 of the matching
   `outputs/{norm}/encoding_results.json["r2_P"]` mean (noise causes some
   shrinkage but the gap should be small).
4. Expected qualitative pattern under zscore:
   - `var_share[raw_frames] + var_share[fwd_render] > 0.95`.
   - `r2_P_from_block_signal[hidden_acts]` and `[inferred_physics]` are large
     (~0.5 or higher), but their `var_share` is ≪ 0.01, so
     `effective_P_contribution` is tiny.
5. Expected qualitative pattern under truncated_svd:
   - `var_share` is roughly balanced across blocks (each ≈ 0.2–0.4).
   - `effective_P_contribution[hidden_acts]` is the dominant contributor.

## Out of scope

- Partial-R² matrix across blocks.
- Pass/fail integration into `evaluation.py`.
- New unit tests beyond what exists for `neural_model.py`.
- Changes to encoding, RSA, residual, dissociation, dynamics analyses.
