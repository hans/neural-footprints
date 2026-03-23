# Neural Footprints Simulation — Project Spec

## Project Structure

```
neural_footprints/
├── SPEC.md                   # this file
├── CLAUDE.md                 # instructions for Claude Code
├── requirements.txt          # pybullet, numpy, scipy, scikit-learn, matplotlib, torch
├── config.py                 # all simulation parameters
├── scene_generator.py        # PyBullet scene generation + raw state capture
├── neural_model.py           # random projection: program_state → neural activity
├── analyses/
│   ├── __init__.py
│   ├── predictive_processing.py  # Simulation 4: PP InverseModel → inferred physics
│   ├── encoding.py               # Simulation 1: encoding model false negatives
│   ├── rsa.py                    # Simulation 2: RSA dominated by render
│   ├── dissociation.py           # Simulation 3: R² vs. behavioral sufficiency
│   └── dynamics.py               # STUB — temporal models
├── run_all.py                # orchestrator
└── figures/                  # all output figures saved here
    ├── encoding_analysis.png
    ├── rsa_analysis.png
    ├── dissociation.png
    ├── pp_prediction_bars.png
    ├── pp_inverse_model_dims.png
    └── pp_example_frames.png
```

---

## Pipeline Order

The pipeline runs in `run_all.py` as 7 steps:

1. **Calibrate** `.bullet` file size (pre-scan for consistent padding)
2. **Generate scenes** (2000 PyBullet simulations)
3. **Generate neural activity** (random projection of program states)
4. **Predictive processing** — trains InverseModel, produces `inferred_physics_all`
5. **Encoding analysis** — uses true physics AND inferred physics from step 4
6. **RSA analysis** — uses true physics AND inferred physics from step 4
7. **Dissociation analysis** — R² vs. behavioral sufficiency

PP runs before encoding/RSA so that inferred latents are available as an additional
feature set for the standard analyses. This makes the central dissociation visible
within each analysis rather than only in the PP-specific figures.

---

## Data Flow

```
PyBullet scene (N_SCENES = 2000, N_OBJECTS = 1)
    │
    ├──▶ render buffers: RGBA + depth + segmentation
    │      64×64×4 + 64×64×4 + 64×64×4 = 49,152 bytes
    │      captured at t=0 (initial), t=5 (early), t=30 (final)
    │
    ├──▶ saveBullet blob: ~20-24KB, padded to K bytes
    │
    └──▶ API labels (NEVER enter neural generation):
           physics_labels: [n_scenes × 15] floats
             per object: pos(3), orn(4), lin_vel(3), ang_vel(3), mass(1), friction(1)
             × 1 object = 15
           behavior_label: binary (median-split on kinetic energy)

                   │
                   ▼
    program_state = concat(render_bytes, bullet_bytes)
                  = ~78,024 bytes → cast to float32 → [78024] vector
                   │
                   ▼  z-score per dimension across scenes
                   │
                   ▼  W ~ N(0, 1/√D), shape [500 × 78024]
                   │
    neural_activity = standardized @ W.T + noise
                    = [n_scenes × 500]
```

### Scene Design

Object 0 (launcher, always visible), Object 1 (visible), Object 2 (occluded behind
wall at y=0). Occlusion makes pixel-only models fundamentally limited for behavior
prediction — velocity is invisible to single-frame pixels, and occluded objects
contribute to physics but not to renders.

### What the analyses receive

| Data | Source | Used in neural generation? |
|---|---|---|
| `neural_activity` [2000 × 500] | W @ program_state + noise | YES — this is the output |
| `program_states` [2000 × ~78024] | raw bytes | YES — this is the input to W |
| `pixel_indices` slice(0, 16384) | RGBA portion of program_state | YES (as part of program_state) |
| `initial_renders` [2000 × 16384] | t=0 RGBA pixels | NO — separate capture |
| `early_renders` [2000 × 16384] | t=5 RGBA pixels | NO — separate capture |
| `physics_labels` [2000 × 15] | pybullet API calls | NO — collected separately |
| `behavior_labels` [2000] | median split on KE | NO — computed separately |
| `inferred_physics_all` [2000 × 15] | InverseModel(t=0, t=5 → physics) | NO — derived from PP model |

The critical asymmetry: pixel features used in analysis are a **direct linear readout**
of bytes that entered the projection. Physics labels used in analysis are a **separate,
lossy, low-dimensional summary** of the engine state that entered the projection.
Inferred physics adds a third category: a **learned nonlinear transform** of pixel frames
that captures causal structure but is still methodologically invisible.

---

## Predictive Processing Model (Simulation 4)

### Architecture

```
Stage 1 (InverseModel): pixel_pca_t0 + pixel_pca_t5 → inferred_physics [n × 4]
  - Two-frame input: optical flow encodes velocity information
  - PyTorch MLP: 100 → 256 (ReLU, Dropout) → 256 (ReLU, Dropout) → observable dims
  - MC dropout active at inference for uncertainty estimation
  - Predicts only pixel-observable dims (position, variable velocity components)
  - Intrinsic properties (mass, friction) and constant dims (orientation,
    angular velocity) are not predicted — they cannot be recovered from pixels

Stage 2 (Forward): inferred_physics → PyBullet resimulation → final_pixels
  - Deterministic physics engine (not learned)
  - Unobservable intrinsic properties (mass, friction, orientation, angular
    velocity) are supplied from ground truth for resimulation
```

**Observable vs unobservable physics dims:** Not all physics labels are recoverable
from pixel observations. Position and (non-zero) velocity are visible in rendered
frames; mass, friction, orientation (constant at t=0), and angular velocity (zero
at t=0) are not. The InverseModel trains only on observable dims to avoid diluting
its R² with fundamentally unpredictable targets. For resimulation, ground-truth
values are used for unobservable dims since they are structural scene properties
(like shape and color, which are already supplied via `scene_configs`).

### What it tests

Does a behaviorally operative physics intermediate become detectable by standard
neural analyses? Expected answer: **no**. The `neural_r2_inferred_physics` should
remain as low as direct physics labels, even though:
- The inferred physics is behaviorally operative (PP chain R² > 0)
- It's extracted from visual frames (InverseModel R² > 0)
- It's causally needed for pixel prediction

### Integration with encoding and RSA

The `inferred_physics_all` array is passed to both downstream analyses:

- **Encoding (Sim 1):** fits `neural ~ inferred_physics` and `neural ~ pixels + inferred_physics`
  in addition to the original `neural ~ pixels` and `neural ~ pixels + true_physics`.
  Four-bar comparison shows all conditions produce similar R², confirming the inferred
  intermediate adds nothing to neural encoding.

- **RSA (Sim 2):** computes an Inferred Physics RDM and `neural ↔ inferred` Spearman
  correlation + partial correlation controlling for render. Shows the inferred physics
  RDM has the same low correlation with neural as true physics.

---

## Figures

### Figure 1: Encoding Model False Negatives
**File:** `figures/encoding_analysis.png`
**Source:** `analyses/encoding.py`

Three panels:

1. **R² bar plot.** Four bars when inferred physics is available:
   - "Pixels only" — baseline encoding R²
   - "Pixels + true physics" — negligible improvement (ΔR² ≈ 0)
   - "Inferred physics only" — low R² (physics alone doesn't explain neural)
   - "Pixels + inferred physics" — negligible improvement over pixels alone
   Message: neither true nor inferred physics adds to encoding, despite both being
   causally operative.

2. **Subsampling curve.** X-axis: number of neurons sampled (10 → 500).
   Y-axis: mean ΔR². Shows the tiny physics increment is stable but
   negligibly small regardless of how many neurons you record from.

3. **Control accuracy.** Single bar: physics → behavior prediction accuracy.
   Establishes that the physics labels are genuinely informative.

### Figure 2: RSA Dominated by Render Structure
**File:** `figures/rsa_analysis.png`
**Source:** `analyses/rsa.py`

Five panels when inferred physics is available:

1. **Neural RDM** heatmap (100×100 subsample)
2. **Render RDM** heatmap
3. **Physics RDM** heatmap
4. **Inferred Physics RDM** heatmap
5. **Correlation bar plot.** Five bars:
   - Neural ↔ Render (high — dominant)
   - Neural ↔ Physics (low)
   - Neural ↔ Physics | Render (near zero after partialing)
   - Neural ↔ Inferred Physics (low — same as true physics)
   - Neural ↔ Inferred | Render (near zero)

   Message: neural similarity structure tracks render structure, not physics — and
   this holds whether physics is measured directly or inferred from a trained model.

### Figure 3: R² vs. Behavioral Sufficiency Dissociation
**File:** `figures/dissociation.png`
**Source:** `analyses/dissociation.py`

Two panels side by side:

1. **Left panel: Neural R² contribution.**
   Render model R² (tall) vs. Physics model unique R² (short).

2. **Right panel: Behavioral prediction.**
   Render model → behavior (poor) vs. Physics model → behavior (high).

The visual: the tall bar on the left becomes the short bar on the right, and
vice versa. The model that explains the brain can't explain the world; the model
that explains the world can't be found in the brain.

### Figure 4: Predictive Processing Prediction Bars
**File:** `figures/pp_prediction_bars.png`
**Source:** `analyses/predictive_processing.py`

Bar chart comparing pixel prediction R² across four conditions:
- Prior MLP (architecture ceiling)
- Oracle (true physics → PyBullet, ~1.0)
- PP chain (inferred physics → PyBullet)
- Render-only (pixel MLP, no physics intermediate)

Plus neural R² bars: t=0 pixel PCA, two-frame PCA, inferred physics.

### Figure 5: InverseModel Per-Dimension R²
**File:** `figures/pp_inverse_model_dims.png`
**Source:** `analyses/predictive_processing.py`

Bar chart showing how well the InverseModel recovers each physics dimension
(position, orientation, velocity, angular velocity, mass, friction).

### Figure 6: PP Example Frames
**File:** `figures/pp_example_frames.png`
**Source:** `analyses/predictive_processing.py`

Side-by-side comparison of actual vs. predicted frames for sample scenes.

---

## Config Parameters

```python
N_SCENES = 2000           # number of PyBullet scenes
N_OBJECTS = 1             # objects per scene (physics labels = 15 dims each)
IMAGE_SIZE = 64           # render resolution
N_NEURONS = 500           # simulated neurons
N_TIMESTEPS = 30          # physics steps per scene
NOISE_LEVEL = 0.3         # noise as fraction of signal std
RANDOM_SEED = 42
RSA_SUBSAMPLE = 500       # scenes used for RDM computation
BULLET_BYTES_K = None     # set by calibration
PIXEL_PCA_DIM = 500       # PCA components for pixel features in encoding/RSA
BEHAVIORAL_PCA_DIM = 50   # PCA dims for next-frame behavioral task (MLP output)
BEHAVIORAL_OBJECTIVE = "next_frame_pixels"  # or "kinetic_energy"

# Predictive Processing model hyperparameters
PP_HIDDEN_DIM = 256       # InverseModel hidden layer width
PP_PIXEL_PCA_DIM = 50     # two-frame pixel PCA dimension
PP_EARLY_FRAME = 5        # timestep for second input frame
PP_DROPOUT_RATE = 0.1     # MC dropout probability
```

---

## Known Issues / Open Questions

1. **saveBullet blob opacity.** The physics bytes include serialization
   headers, broadphase trees, solver cache, padding — not just positions/masses.
   We can't decompose how much is "high-level physics" vs. engine bookkeeping.
   The variance diagnostic overstates the physics *content* contribution.

2. **Analysis-side asymmetry is doing work.** The false negative result depends
   partly on the fact that 15 API floats don't span the same subspace as ~29K raw
   bytes. The PP analysis partially addresses this: inferred physics is a learned
   transform of pixel data, yet it is still invisible to encoding/RSA.

3. **Subsampling curve is incomplete.** Currently only shows ΔR² (physics
   detectability). Could add differential degradation: two curves showing
   render vs. physics detection rates diverging under subsampling.

4. **Dynamics analysis is a stub.** The Buster vs. Aaron comparison (temporal
   prediction from high-level vs. low-level state) is the positive proposal
   and is not yet implemented.

5. **False positive simulation not built.** The Canada vs. Mexico pixel
   decoding demonstration is a separate experiment.
