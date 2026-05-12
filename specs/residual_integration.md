# Residual Analysis Integration Spec

## Context

`analyses/residual.py` (added in commit `77e2033`) implements the two-stage
render-residualization procedure but is not wired into the pipeline. A prior
implementation on the `xaccel-physics-target` branch (`ba0145b`, `5619687`)
wired a PP/inferred-physics version. This branch stripped the PP dependency and
rewrote the analysis to use GT physics only — so the wiring must be rebuilt to
match the current `run_residual_analysis()` signature, **not** copied verbatim
from `xaccel-physics-target`.

## What `run_residual_analysis()` returns

```python
{
    'r2_raw_render':         np.ndarray,  # per-neuron R², raw neural ~ render PCA
    'r2_raw_physics_gt':     np.ndarray,  # per-neuron R², raw neural ~ GT physics
    'r2_resid_render':       np.ndarray,  # per-neuron R², residualized ~ render PCA
    'r2_resid_physics_gt':   np.ndarray,  # per-neuron R², residualized ~ GT physics
    'residual_variance_fraction': float,  # mean(var(y_resid)) / mean(var(y))
    'render_pca_dim': int,
    'n_splits': int,
    'random_state': int,
}
```

No `inferred` or `combined` keys — those are PP-era artifacts that do not exist
in this branch's implementation.

## Files to create

### `scripts/run_residual.py`

Follow the same boilerplate as `scripts/run_dynamics.py`:

```python
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
from load_config import load_config
from io_utils import load_scenes, load_neural, save_results
from analyses.residual import run_residual_analysis

cfg = load_config()
scenes = load_scenes(snakemake.input.scenes)
neural, neural_meta = load_neural(snakemake.input.neural)

results = run_residual_analysis(
    neural, scenes, neural_meta,
    render_pca_dim=cfg['render_pca_dim'],
)
save_results(results, snakemake.output.results)

np.savez_compressed(
    snakemake.output.plot_data,
    r2_raw_render=results['r2_raw_render'],
    r2_raw_physics_gt=results['r2_raw_physics_gt'],
    r2_resid_render=results['r2_resid_render'],
    r2_resid_physics_gt=results['r2_resid_physics_gt'],
    residual_variance_fraction=np.array(results['residual_variance_fraction']),
)
```

### `scripts/plot_residual.py`

Identical pattern to `scripts/plot_dynamics.py`:

```python
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
from analyses.plot_figures import plot_residual

fig_dir = os.path.dirname(snakemake.output.figure)
os.makedirs(fig_dir, exist_ok=True)

plot_data = dict(np.load(snakemake.input.plot_data, allow_pickle=False))
plot_residual(plot_data, fig_dir=fig_dir)
```

## Files to modify

### `analyses/plot_figures.py`

Append a `plot_residual()` function after the last existing function
(`plot_sample_scenes`, ending at line 472). Use the two-panel layout from the
PP-era version but with only the two GT predictor sets:

- **Panel A** — per-neuron scatter: x = `r2_raw_physics_gt`, y =
  `r2_resid_physics_gt`. Points below the diagonal show collapse. Add y=x
  dashed reference line and a y=0 dotted line. Label axes clearly.
- **Panel B** — grouped bars: 2 groups (`render`, `physics_gt`), each with
  two bars (raw vs residualized), ±SEM error bars, y=0 reference line.
  x-tick labels: `['Render', 'GT Physics']`.

Color convention (from `COLORS` dict, same as the rest of the file):
- Raw bars: `COLORS['pixels']`
- Residualized bars: `COLORS['physics']`

Title for Panel B should include `residual_variance_fraction` formatted to 2
decimal places.

### `Snakefile`

Add two rules and wire the figure into `rule all` / `rule figures`.

**`rule residual`** (after `rule dynamics`, before the plotting section):

```
rule residual:
    input:
        scenes="data/scenes.npz",
        neural="data/neural.npz",
    output:
        results="outputs/residual_results.json",
        plot_data="data/residual_plot_data.npz",
    script:
        "scripts/run_residual.py"
```

**`rule plot_residual`** (after `rule plot_dynamics`):

```
rule plot_residual:
    input:
        plot_data="data/residual_plot_data.npz",
    output:
        figure="figures/residual_analysis.pdf",
    script:
        "scripts/plot_residual.py"
```

Add `"figures/residual_analysis.pdf"` to the input lists of both `rule all`
and `rule figures`.

### `scripts/run_evaluate.py`

Add residual loading after the existing inputs, before the `evaluate()` call:

```python
residual_results = None
if hasattr(snakemake.input, 'residual'):
    residual_results = load_results(snakemake.input.residual)
    for key in ['r2_raw_render', 'r2_raw_physics_gt',
                'r2_resid_render', 'r2_resid_physics_gt']:
        if key in residual_results and residual_results[key] is not None:
            residual_results[key] = np.array(residual_results[key])
```

Pass `residual_results=residual_results` to `evaluate()`.

Also add `residual="outputs/residual_results.json"` to the `rule evaluate`
input block in the Snakefile.

### `evaluation.py`

Add a `residual_results=None` parameter to `evaluate()`. Add a new section
after the Dissociation block (and before Dynamics):

```
# --- Residual Encoding ---
if residual_results is not None:
    r2_resid_render  = float(residual_results['r2_resid_render'].mean())
    r2_raw_gt        = float(residual_results['r2_raw_physics_gt'].mean())
    r2_resid_gt      = float(residual_results['r2_resid_physics_gt'].mean())
    var_kept         = float(residual_results['residual_variance_fraction'])

    check("Stage-1 sanity: render does not predict its own residual",
          abs(r2_resid_render) < 0.05,
          f"R² = {r2_resid_render:.4f}",
          "expect |R²| < 0.05")
    check("Stage-1 leaves substantial residual variance",
          var_kept > 0.05,
          f"residual var fraction = {var_kept:.4f}",
          "expect > 0.05")
    check("Raw neural carries GT-physics signal (pre-residualization)",
          r2_raw_gt > 0.01,
          f"R² = {r2_raw_gt:.4f}",
          "expect > 0.01 — needed for the collapse to be meaningful")
    check("Residualization removes GT-physics signal (false negative)",
          r2_resid_gt < 0.01,
          f"R² = {r2_resid_gt:.4f}",
          "expect < 0.01")
```

Note: no `inverse_ok` gating — these checks are PP-independent.

## What NOT to port from `xaccel-physics-target`

- Do **not** add an `inferred` input to `rule residual`
- Do **not** reference `r2_raw_inferred`, `r2_resid_inferred`, or `r2_raw_combined` / `r2_resid_combined` anywhere
- Do **not** add `inferred_physics` parameter to `run_residual_analysis()`
- Do **not** copy the 4-bar panel B from the PP-era `plot_residual` — use 2 groups only

## Verification

```bash
uv run snakemake -n          # dry-run shows residual + plot_residual rules
uv run snakemake -j3 outputs/residual_results.json
uv run snakemake -j3 figures/residual_analysis.pdf
uv run snakemake -j1 outputs/evaluation.json
```

Expect:
- `outputs/residual_results.json` has keys `r2_raw_render`, `r2_raw_physics_gt`, `r2_resid_render`, `r2_resid_physics_gt`, `residual_variance_fraction`
- `figures/residual_analysis.pdf` renders without error
- Evaluation report contains a "Residual Encoding" section with 4 checks
- `r2_resid_render` mean should be near 0 (stage-1 sanity)
- `r2_resid_physics_gt` mean should be substantially lower than `r2_raw_physics_gt` (the false-negative result)
