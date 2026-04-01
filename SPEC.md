# Neural Footprints Simulation — Project Spec

## Glossary

**Pixel state.**
The RGBA color buffer alone — what a camera would capture. A subset of render state, excluding depth and segmentation.

**Render state.**
All render buffers from PyBullet: RGBA color, depth map, and segmentation mask. A superset of pixel state. The high-dimensional sensory signal that dominates the program state.

**Physics labels.**
Per-object state extracted via the PyBullet API: position, orientation, linear velocity, angular velocity, mass, and friction. Concatenated into program state (and thus linearly present in neural activity), but occupying a tiny fraction of total dimensions. Also collected separately for use as analysis regressors.

**Scene config.**
Per-object shape and launch parameters (shape type, dimensions, initial position, velocity, acceleration) encoded as a fixed-length float vector.

**Program state.**
The full state vector fed to the random projection. Concatenation of render state, physics labels, and scene config. Contains everything sufficient to resimulate the scene.

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
├── Snakefile                 # pipeline DAG (replaces run_all.py)
├── scripts/
│   ├── load_config.py        # YAML config loader
│   ├── io_utils.py           # save/load intermediates (scenes, neural, results)
│   ├── calibrate.py          # rule: calibrate bullet file size
│   ├── gen_scenes.py         # rule: generate PyBullet scenes
│   ├── gen_neural.py         # rule: generate neural activity
│   ├── run_encoding.py       # rule: encoding model analysis
│   ├── run_rsa.py            # rule: RSA analysis
│   ├── run_dissociation.py   # rule: dissociation analysis
│   └── run_evaluate.py       # rule: evaluation pass/fail
├── scene_generator.py        # PyBullet scene generation + raw state capture
├── neural_model.py           # random projection: program_state → neural activity
├── analyses/
│   ├── __init__.py
│   ├── encoding.py           # Simulation 1: encoding model false negatives
│   ├── rsa.py                # Simulation 2: RSA dominated by render
│   ├── dissociation.py       # Simulation 3: R² vs. behavioral sufficiency
│   └── dynamics.py           # STUB — Buster vs. Aaron temporal models
├── evaluation.py             # pass/fail checks
├── data/                     # expensive intermediates (gitignored)
│   ├── scenes.npz
│   └── neural.npz
├── outputs/                  # cheap derived results (gitignored)
│   ├── bullet_k.json
│   ├── encoding_results.json
│   ├── rsa_results.json
│   ├── dissociation_results.json
│   └── evaluation.json
└── figures/                  # all output figures saved here
    ├── encoding_analysis.png
    ├── rsa_analysis.png
    ├── dissociation.png
    └── predicted_frames.png
```

---

## Data Flow

```
PyBullet scene (N_SCENES = 2000, N_OBJECTS = 1)
    │
    ├──▶ render buffers: RGBA + depth + segmentation
    │      64×64×4 + 64×64×4 + 64×64×4 = 49,152 raw bytes → cast to float32
    │
    ├──▶ physics_labels: 15 × N_OBJECTS = 15 floats
    │      per object: pos(3), orn(4), lin_vel(3), ang_vel(3), mass(1), friction(1)
    │
    ├──▶ scene_config: 10 × N_OBJECTS = 10 floats
    │      per object: shape_is_box(1), radius(1), half_extents(3), color(4), x_accel(1)
    │
    └──▶ behavior_label: binary (median split on final kinetic energy)

                    │
                    ▼
    program_state = concat(render_bytes, physics_labels, scene_config)
                  = 49,152 + 15 + 10 = 49,177 floats
                    │
                    ▼  z-score per dimension across scenes
                    │
                    ▼  W ~ N(0, 1/√D), shape [500 × 49,177]
                    │
    neural_activity = standardized @ W.T + noise
                    = [n_scenes × 500]
```

Physics labels are part of the program state (and thus linearly present in
neural activity), but they are also extracted separately via the API for use
as analysis regressors — mirroring how neuroscientists construct feature sets
from external measurements rather than from the neural code itself.

### What the analyses receive

| Data | Source | Used in neural generation? |
|---|---|---|
| `neural_activity` [2000 × 500] | W @ program_state + noise | YES — this is the output |
| `program_states` [2000 × 49,177] | render + physics + config | YES — this is the input to W |
| `pixel_indices` slice(0, 16384) | RGBA portion of program_state | YES (as part of program_state) |
| `render_indices` slice(0, 49152) | RGBA + depth + seg portion | YES (as part of program_state) |
| `physics_labels` [2000 × 15] | pybullet API calls | YES (concatenated into program_state) and also collected separately as analysis regressors |
| `behavior_labels` [2000] | median split on final KE | NO — computed separately |

The critical asymmetry: render features used in analysis are a **direct linear readout**
of 49,152 bytes that entered the projection. Physics labels used as analysis regressors
are the same 15 floats that entered the projection, but they represent only 0.03% of the
total program state dimensions — too small a subspace to register in variance-based methods.

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

```yaml
n_scenes: 2000            # number of PyBullet scenes
n_objects: 1              # objects per scene
image_size: 64            # render resolution
n_neurons: 500            # simulated neurons
n_timesteps: 30           # physics steps per scene
noise_level: 0.3          # noise as fraction of signal std
random_seed: 42
rsa_subsample: 500        # scenes used for RDM computation
pixel_pca_dim: 500        # PCA components for pixel features in analyses
behavioral_pca_dim: 50    # PCA dims for behavioral task MLP
behavioral_objective: "next_frame_pixels"
```

---

## Known Issues / Open Questions

1. **Dimension asymmetry is extreme by design.** The program state is 49,177
   floats: 49,152 render bytes vs. 15 physics labels + 10 scene config. Physics
   is ~0.05% of the signal. This makes the false negative airtight — physics is
   literally in the neural code but occupies too small a subspace to register.

2. **Subsampling curve is incomplete.** Currently only shows ΔR² (physics
   detectability). Spec asks for differential degradation: two curves showing
   render vs. physics detection rates diverging under subsampling.

3. **Dynamics analysis is a stub.** The Buster vs. Aaron comparison (temporal
   prediction from high-level vs. low-level state) is the positive proposal
   and is not yet implemented.

4. **False positive simulation not built.** The Canada vs. Mexico pixel
   decoding demonstration (spec line 64-66) is a separate experiment.

5. **Pixel/render slice consistency across analyses.** Previously, encoding
   used `render_indices` (RGBA + depth + seg, 49,152 dims) while RSA and
   dissociation used `pixel_indices` (RGBA only, 16,384 dims). This was
   fixed to use `render_indices` everywhere. **Before finalizing results,
   verify all analyses use the same render slice** — this kind of silent
   mismatch can change numerical outcomes without any obvious error.
