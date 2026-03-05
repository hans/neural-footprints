# Neural Footprints Simulation — Project Spec

## Project Structure

```
neural_footprints/
├── SPEC.md                   # this file
├── requirements.txt          # pybullet, numpy, scipy, scikit-learn, matplotlib
├── config.py                 # all simulation parameters
├── scene_generator.py        # PyBullet scene generation + raw state capture
├── neural_model.py           # random projection: program_state → neural activity
├── analyses/
│   ├── __init__.py
│   ├── encoding.py           # Simulation 1: encoding model false negatives
│   ├── rsa.py                # Simulation 2: RSA dominated by render
│   ├── dissociation.py       # NEW — Simulation 3: R² vs. behavioral sufficiency
│   └── dynamics.py           # STUB — Buster vs. Aaron temporal models
├── run_all.py                # orchestrator
└── figures/                  # all output figures saved here
    ├── encoding_analysis.png
    ├── rsa_analysis.png
    └── dissociation.png      # NEW
```

---

## Data Flow

```
PyBullet scene (N_SCENES = 2000)
    │
    ├──▶ render buffers: RGBA + depth + segmentation
    │      64×64×4 + 64×64×4 + 64×64×4 = 49,152 bytes
    │
    ├──▶ saveBullet blob: ~20-24KB, padded to K = 28,872 bytes
    │
    └──▶ API labels (NEVER enter neural generation):
           physics_labels: [n_scenes × 75] floats
             per object: pos(3), orn(4), lin_vel(3), ang_vel(3), mass(1), friction(1)
             × 5 objects = 75
           behavior_label: binary
             did ≥2 non-launcher objects move >0.5 units?

                    │
                    ▼
    program_state = concat(render_bytes, bullet_bytes)
                  = 78,024 bytes → cast to float32 → [78024] vector
                    │
                    ▼  z-score per dimension across scenes
                    │
                    ▼  W ~ N(0, 1/√D), shape [500 × 78024]
                    │
    neural_activity = standardized @ W.T + noise
                    = [n_scenes × 500]
```

### What the analyses receive

| Data | Source | Used in neural generation? |
|---|---|---|
| `neural_activity` [2000 × 500] | W @ program_state + noise | YES — this is the output |
| `program_states` [2000 × 78024] | raw bytes | YES — this is the input to W |
| `pixel_indices` slice(0, 16384) | RGBA portion of program_state | YES (as part of program_state) |
| `physics_labels` [2000 × 75] | pybullet API calls | NO — collected separately |
| `behavior_labels` [2000] | computed from API positions | NO — computed separately |

The critical asymmetry: pixel features used in analysis are a **direct linear readout**
of bytes that entered the projection. Physics labels used in analysis are a **separate,
lossy, low-dimensional summary** of the engine state that entered the projection.

---

## Figures

### Figure 1: Encoding Model False Negatives
**File:** `figures/encoding_analysis.png`
**Source:** `analyses/encoding.py`

Three panels:

1. **R² bar plot.** Two bars: "Pixels only" (R² ≈ 0.436) vs. "Pixels + Physics"
   (R² ≈ 0.451). Near-identical height. Annotated with ΔR² = 0.015.
   Message: adding physics to the encoding model does essentially nothing.

2. **Subsampling curve.** X-axis: number of neurons sampled (10 → 500).
   Y-axis: mean ΔR². Shows that the tiny physics increment is stable but
   negligibly small regardless of how many neurons you record from.
   TODO: add a complementary curve for render detectability to show
   differential degradation rates.

3. **Control accuracy.** Single bar: physics → behavior prediction at 99.6%.
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
   - Neural ↔ Render: r = 0.238 (dominant)
   - Neural ↔ Physics: r = 0.040 (small)
   - Neural ↔ Physics | Render: r = 0.030 (near zero after partialing)

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
| Render model | pixel PCA (200 dims) | HIGH (~0.44) | LOW (near chance) |
| Physics model | API physics labels (75 dims) | LOW (ΔR² ~0.015) | HIGH (~99.6%) |

The render model accounts for most of the neural variance but is useless for
predicting what happens in the scene. The physics model is sufficient for
explaining environment dynamics but adds basically nothing to the encoding model.

**Panel layout — 2 panels side by side:**

1. **Left panel: Neural R² contribution.**
   Two bars: Render model R² (tall, ~0.44) vs. Physics model unique R² (short, ~0.015).
   Y-axis: "Neural variance explained (R²)"

2. **Right panel: Behavioral prediction accuracy.**
   Two bars: Render model → behavior (near chance) vs. Physics model → behavior (~99.6%).
   Y-axis: "Behavior prediction accuracy"
   Dashed line at 50% (chance).

The visual: the tall bar on the left becomes the short bar on the right, and vice versa.
The model that explains the brain data can't explain the world; the model that explains
the world can't be found in the brain data.

**What needs to be computed:**
- Render → behavior: logistic regression from pixel_PCA → behavior_label (cross-validated).
  Expected: poor, because pixel similarity doesn't predict which objects moved.
- Physics → behavior: already computed (99.6%).
- Render → neural R²: already computed (0.436).
- Physics unique → neural ΔR²: already computed (0.015).

---

## Config Parameters

```python
N_SCENES = 2000          # number of PyBullet scenes
N_OBJECTS = 5            # objects per scene
IMAGE_SIZE = 64          # render resolution
N_NEURONS = 500          # simulated neurons
N_TIMESTEPS = 30         # physics steps per scene
NOISE_LEVEL = 0.3        # noise as fraction of signal std
RANDOM_SEED = 42
RSA_SUBSAMPLE = 500      # scenes used for RDM computation
BULLET_BYTES_K = None    # set by calibration (~28K)
PIXEL_PCA_DIM = 200      # PCA components for pixel features in analyses
```

---

## Current Results (from last run)

```
Variance diagnostic:
  D_render = 49,152 dims, D_physics = 28,872 dims (ratio 1.7x)
  Render variance fraction: 59.4%
  Physics variance fraction: 40.6%

Encoding model:
  R² pixels only:        0.4356
  R² pixels + physics:   0.4505
  ΔR²:                   0.0149

Control:
  Physics → behavior:    99.60%

RSA:
  Neural ↔ Render:       r = 0.238
  Neural ↔ Physics:      r = 0.040
  Neural ↔ Phys|Render:  r = 0.030
```

---

## Known Issues / Open Questions

1. **saveBullet blob opacity.** The 28,872 physics bytes include serialization
   headers, broadphase trees, solver cache, padding — not just positions/masses.
   We can't decompose how much is "high-level physics" vs. engine bookkeeping.
   The variance diagnostic (40.6%) overstates the physics *content* contribution.

2. **Analysis-side asymmetry is doing work.** The false negative result depends
   partly on the fact that 75 API floats don't span the same subspace as 29K raw
   bytes. A fairer version would use an explicit two-component program state
   where physics_vector IS the 75 API floats, making the claim airtight:
   you know exactly what's in each component.

3. **Subsampling curve is incomplete.** Currently only shows ΔR² (physics
   detectability). Spec asks for differential degradation: two curves showing
   render vs. physics detection rates diverging under subsampling.

4. **Dynamics analysis is a stub.** The Buster vs. Aaron comparison (temporal
   prediction from high-level vs. low-level state) is the positive proposal
   and is not yet implemented.

5. **False positive simulation not built.** The Canada vs. Mexico pixel
   decoding demonstration (spec line 64-66) is a separate experiment.
