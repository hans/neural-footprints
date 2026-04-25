# Neural Footprints Simulation — Project Spec

## Glossary

**Pixel state.**
The RGBA color buffer alone — what a camera would capture. A subset of render state, excluding depth and segmentation.

**Render state.**
All render buffers from PyBullet: RGBA color, depth map, and segmentation mask. A superset of pixel state. The high-dimensional sensory signal that dominates the program state.

**Physics labels.**
Per-object state extracted via the PyBullet API: position, orientation, linear velocity, angular velocity, mass, and friction. Concatenated into program state (and thus linearly present in neural activity), but occupying a tiny fraction of total dimensions. Also collected separately for use as analysis regressors.

**Scene config.**
Per-object shape and appearance parameters (shape type, dimensions, color, acceleration) encoded as a fixed-length float vector.

**Scene lighting.**
Per-scene rendering parameters not tied to any individual object: pillar gray level, light direction, light color, and light distance. Together with scene config and physics labels, these are sufficient to deterministically re-render the scene.

**Program state.**
The full state vector fed to the random projection. Concatenation of render state, physics labels, scene config, and scene lighting. Contains everything sufficient to resimulate the scene.

**Neural activity.**
Synthetic "brain data" produced by random linear projection of program state plus noise. Since program state contains both render and physics information, both are linearly decodable in principle.

**Behavior label.**
Binary label derived from final kinetic energy (median split). Recoverable from physics labels (which include velocity and mass) but not from pixels (which carry no velocity signal).

**Render model** / **Physics model.**
The two competing analysis-side models. The render model uses PCA-reduced render state as regressors; the physics model uses the low-dimensional physics labels. The core result is a double dissociation: the render model explains neural variance but not behavior; the physics model explains behavior but not neural variance.

**Information asymmetry.**
The structural reason variance-based methods fail. Render dimensions vastly outnumber physics dimensions in the program state, so encoding models, RSA, and PCA are dominated by render structure and systematically miss the physics signal — even though physics is linearly present and causally determines scene dynamics.

---

## Project Structure

```
neural_footprints/
├── SPEC.md                   # this file
├── requirements.txt          # pybullet, numpy, scipy, scikit-learn, matplotlib, snakemake
├── config.yaml               # all simulation parameters (YAML)
├── config.py                 # backward-compat shim (loads config.yaml)
├── Snakefile                 # pipeline DAG
├── scripts/
│   ├── load_config.py        # YAML config loader
│   ├── io_utils.py           # save/load intermediates (scenes, neural, results)
│   ├── gen_scenes.py         # rule: generate PyBullet scenes
│   ├── gen_neural.py         # rule: generate neural activity
│   ├── run_encoding.py       # rule: encoding model analysis
│   ├── run_rsa.py            # rule: RSA analysis
│   ├── run_dissociation.py   # rule: dissociation analysis
│   ├── run_pca.py            # rule: PCA analysis
│   ├── run_dynamics.py       # rule: dynamics analysis
│   ├── run_pp.py             # rule: predictive-processing analysis
│   ├── run_evaluate.py       # rule: evaluation pass/fail
│   ├── gen_macros.py         # rule: generate LaTeX macros for paper
│   ├── plot_encoding.py      # rule: encoding figure
│   ├── plot_rsa.py           # rule: RSA figure
│   ├── plot_dissociation.py  # rule: dissociation figure
│   ├── plot_pca.py           # rule: PCA figure
│   ├── plot_dynamics.py      # rule: dynamics figure
│   ├── plot_pp.py            # rule: PP figures (summary + frame grid)
│   └── plot_scenes.py        # rule: sample scene visualization
├── scene_generator.py        # PyBullet scene generation + raw state capture
├── neural_model.py           # random projection: program_state → neural activity
├── analyses/
│   ├── __init__.py
│   ├── encoding.py           # Simulation 1: encoding model false negatives
│   ├── rsa.py                # Simulation 2: RSA dominated by render
│   ├── dissociation.py       # Simulation 3: R² vs. behavioral sufficiency
│   ├── dynamics.py           # Simulation 4: future brain-state prediction
│   ├── predictive_processing.py  # Simulation 5: InverseModel (pixels→physics) +
│   │                              # PP chain (PyBullet forward); inferred-physics
│   │                              # array fed to encoding/rsa/dissociation
│   ├── pca_analysis.py       # Negative result: variance ≠ information
│   ├── plot_figures.py       # all figure rendering
│   └── plot_style.py         # shared plot styling utilities
├── evaluation.py             # pass/fail checks
├── data/                     # expensive intermediates (gitignored)
│   ├── scenes.npz
│   └── neural.npz
├── outputs/                  # cheap derived results (gitignored)
│   ├── encoding_results.json
│   ├── rsa_results.json
│   ├── dissociation_results.json
│   ├── pca_results.json
│   ├── dynamics_results.json
│   ├── pp_results.json
│   └── evaluation.json
└── figures/                  # all output figures saved here (PDF)
```

> **TODO (deferred from main):** `analyses/subtractive.py` (feature-removal
> importance analysis) was developed on the `main` branch but has not been
> ported to this scaffold. Resurrecting it would require a new Snakemake
> rule, plot script, and config knobs. Not blocked by anything in the PP
> port; revisit if the result is wanted in the paper.

---

## Data Flow

```
PyBullet scene (N_SCENES scenes)
    │
    ├──▶ render buffers: RGBA + depth + segmentation
    │      three image-sized buffers → tens of thousands of floats
    │
    ├──▶ physics labels (via API):
    │      per object: pos, orn, lin_vel, ang_vel, mass, friction
    │      low-dimensional vector (tens of floats)
    │
    ├──▶ scene config:
    │      per object: shape type, dimensions, color, acceleration
    │      low-dimensional vector
    │
    ├──▶ scene lighting:
    │      per scene: pillar gray, light direction, light color, light distance
    │      low-dimensional vector
    │
    └──▶ behavior label (derived from physics labels):
           binary kinetic-energy median split
           KE = Σ 0.5 × mass × |velocity|² at final timestep

                    │
                    ▼
    program_state = concat(render_bytes, physics_labels, scene_config, scene_lighting)
                  = [D] vector, dominated by render dimensions
                    │
                    ▼  z-score per dimension across scenes
                    │
                    ▼  W ~ N(0, 1/√D), shape [n_neurons × D]
                    │
    neural_activity = standardized @ W.T + noise
                    = [n_scenes × n_neurons]
```

Physics labels are part of the program state (and thus linearly present in
neural activity), but they are also extracted separately via the API for use
as analysis regressors — mirroring how neuroscientists construct feature sets
from external measurements rather than from the neural code itself.

### What the analyses receive

| Data | Source | Part of program state? |
|---|---|---|
| `neural_activity` | W @ program_state + noise | YES — this is the output |
| `program_states` | render + physics + config + lighting concat | YES — this is the input to W |
| `render_indices` | render portion of program_state | YES (as part of program_state) |
| `physics_labels` | PyBullet API calls | YES — concatenated into program_state |
| `behavior_labels` | KE median split from physics | NO — computed separately |

The critical asymmetry: render dimensions vastly outnumber physics dimensions
in the program state (hundreds-to-one ratio). Encoding models, RSA, and PCA
are dominated by render structure and systematically miss the physics signal —
even though physics is linearly present and causally determines scene dynamics.

---

## Figures

### Figure 1: Encoding Model False Negatives
**File:** `figures/encoding_analysis.png`
**Source:** `analyses/encoding.py`

Three panels:

1. **R² bar plot.** Two bars: "Pixels only" vs. "Pixels + Physics".
   Near-identical height, annotated with ΔR².
   Message: adding physics to the encoding model does essentially nothing.

2. **Subsampling curve.** X-axis: number of neurons sampled (10 → 500).
   Y-axis: mean ΔR². Shows that the tiny physics increment is stable but
   negligibly small regardless of how many neurons you record from.
   TODO: add a complementary curve for render detectability to show
   differential degradation rates.

3. **Control accuracy.** Single bar: physics → behavior prediction accuracy.
   Establishes that the physics labels are genuinely informative — the failure
   to detect them is a methodological failure, not a sign they're unimportant.

### Figure 2: RSA Dominated by Render Structure
**File:** `figures/rsa_analysis.png`
**Source:** `analyses/rsa.py`

Four panels:

1. **Neural RDM** heatmap (100×100 subsample)
2. **Render RDM** heatmap
3. **Physics RDM** heatmap
4. **Correlation bar plot.** Three bars:
   - Neural ↔ Render (dominant)
   - Neural ↔ Physics (small)
   - Neural ↔ Physics | Render (near zero after partialing)

   Message: neural similarity structure tracks render structure, not physics.
   What little physics signal exists is explained away by shared render variance.

### Figure 3: R² vs. Behavioral Sufficiency Dissociation — NEW
**File:** `figures/dissociation.png`
**Source:** `analyses/dissociation.py`

The key new figure. Directly visualizes the disconnect between two things
a neuroscientist might care about:

- **Neural variance explained (R²):** how well does this feature set predict
  neural activity?
- **Behavioral sufficiency:** how well does this feature set predict future
  environment states / behavioral outcomes?

Two "models" are compared:

| Model | What it uses | Neural R² | Behavior prediction |
|---|---|---|---|
| Render model | pixel PCA | HIGH | LOW (near chance) |
| Physics model | API physics labels | LOW | HIGH |

The render model accounts for most of the neural variance but is useless for
predicting what happens in the scene. The physics model is sufficient for
explaining environment dynamics but adds basically nothing to the encoding model.

**Panel layout — 2 panels side by side:**

1. **Left panel: Neural R² contribution.**
   Two bars: Render model R² (tall) vs. Physics model unique R² (short).
   Y-axis: "Neural variance explained (R²)"

2. **Right panel: Behavioral prediction accuracy.**
   Two bars: Render model → behavior (near chance) vs. Physics model → behavior (high).
   Y-axis: "Behavior prediction accuracy"
   Dashed line at 50% (chance).

The visual: the tall bar on the left becomes the short bar on the right, and vice versa.
The model that explains the brain data can't explain the world; the model that explains
the world can't be found in the brain data.

---

## Config Parameters (`config.yaml`)

All pipeline parameters live in `config.yaml`. Key settings:

- **Scene generation:** number of scenes, objects per scene, image resolution, physics timesteps
- **Neural model:** number of simulated neurons, noise level, random seed
- **Analysis:** RSA subsample size, PCA dimensions for render features, PCA dimensions for behavioral MLP
- **Behavioral objective:** `"next_frame_pixels"` (MLP R²) or `"kinetic_energy"` (logistic accuracy)

---

## Known Issues / Open Questions

1. **Subsampling curve is incomplete.** Currently only shows ΔR² (physics
   detectability). Spec asks for differential degradation: two curves showing
   render vs. physics detection rates diverging under subsampling.

2. **False positive simulation not built.** The Canada vs. Mexico pixel
   decoding demonstration is a separate experiment.

3. **Pixel/render slice consistency across analyses.** Previously, encoding
   used all render buffers while RSA and dissociation used RGBA only. This was
   fixed to use the full render slice everywhere. **Before finalizing results,
   verify all analyses use the same render slice** — this kind of silent
   mismatch can change numerical outcomes without any obvious error.
