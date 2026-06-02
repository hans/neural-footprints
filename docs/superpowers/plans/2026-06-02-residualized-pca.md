# Residualized PCA: motion decodability is a render confound

## Motivation

The plain PCA negative control (`analyses/pca_analysis.py`) decodes
`motion_dir = (vx > 0)` from the top-k neural PCs. Motion is only weakly
decodable, but it *is* "sort of" decodable — and the accuracy creeps up as more
(higher) PCs are added. We want to show this residual decodability is **driven
by the render confound**, not by a genuine abstract physics representation.

Causal chain: physics → forward renders → pixels. The 3-frame brain renders
carry the object's displacement, so any neural PC capturing render variance
inherits motion-correlated structure. Removing the pixel-explainable component
of neural activity should make motion decodability collapse to chance.

**Key conceptual point (state in the paper):** `vx` is *also* present directly
in the `program_state` physics block, so residualizing on pixels does **not**
remove the genuine causal motion footprint — only the render-mediated one. If
motion still collapses to chance, that is the *stronger* result: the true
physics footprint sits below the noise floor, and the apparent decodability rode
entirely on the render confound. The pixel-PC positive control proves the
renders genuinely carry motion.

## Method (decoding analog of `analyses/residual.py`)

Four conditions, each scored with the same top-k PCA + `LogisticRegressionCV`
sweep (`pca_analysis._decode_pc_sweep`):

- **raw** — decode targets from neural activity's own PCs (baseline).
- **resid_X** — regress `raw_pixel_pca` out of neural (cross-val ridge,
  `residual._ridge_cv_predict`), decode from the residual's PCs.
- **resid_XS** — residualize on `[X, S]` where S = forward-render predictions
  (`predicted_pixel_pca`); the stronger control.
- **pixel** — positive control: decode directly from `raw_pixel_pca`'s PCs.

## Files

- `analyses/pca_analysis.py` — extracted reusable `_decode_pc_sweep` /
  `_pc_counts_for`; `run_pca_analysis` now calls them.
- `analyses/residualized_pca.py` — `run_residualized_pca_analysis(...)`.
- `scripts/run_residualized_pca.py` — Snakemake script (mirrors
  `run_residual.py` to build X and S).
- `scripts/plot_residualized_pca.py` + `plot_figures.plot_residualized_pca`.
- `Snakefile` — `rule residualized_pca`, `rule plot_residualized_pca`, figure in
  `_ALL_FIGURES`, inputs added to `paper_macros` and `evaluate`.
- `scripts/gen_macros.py` — `\pcaMotion{Raw,ResidX,ResidXS,Pixel}AllPC`,
  `\pcaResidVarKeptXS`.
- `evaluation.py` — checks: motion decodable from raw; KEY collapse after X+S;
  pixel positive control above null; residual variance non-degenerate.
- `tests/test_residualized_pca.py` — synthetic structural tests (no engine).

## Expected result

- raw motion curve rises modestly with #PCs.
- resid_X and especially resid_XS motion curves drop to ~chance.
- pixel positive control stays clearly above chance.
- Pure-render controls (`pillar_gray`, `cam_height`) also collapse under
  residualization — expected (no non-pixel source); the distinctive point is
  motion collapsing *despite* having a genuine physics-block source.

## Verification

- `uv run --with pytest python -m pytest tests/test_residualized_pca.py -v`
- `uv run python -m py_compile` the new/edited modules and scripts.
- `uv run snakemake -n outputs/truncated_svd/residualized_pca_results.json`
  (dry run — render steps need GL, only available on prod).
- On prod / full run: inspect the JSON — confirm `resid_XS` all-PC motion ≈
  chance and `pixel` > chance; build `figures/<norm>/residualized_pca.pdf`.
