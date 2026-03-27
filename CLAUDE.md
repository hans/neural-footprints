# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Running the Project

```bash
# Run the full pipeline via Snakemake (tracks DAG, caches intermediates in data/ and outputs/)
uv run snakemake -j1

# Dry run (show what would execute without running)
uv run snakemake -n

# Run with parallelism (encoding, rsa, dissociation are independent)
uv run snakemake -j3
```

## Architecture

This is a simulation study showing that standard neuroscience methods (encoding models, RSA) systematically fail to detect causally operative physics variables. See `SPEC.md` for the full theoretical framing and expected numerical results.

**Data flow:**

```
PyBullet scene
  ├── render bytes (64×64 RGBA/depth/seg → 49,152 floats)
  └── physics blob (.bullet serialization → ~28,872 floats)
         ↓
  program_state = concat(render_bytes, physics_bytes)  [n_scenes × ~78,024]
         ↓ random projection W ~ N(0, 1/√D) + noise
  neural_activity  [n_scenes × 500]
```

The key asymmetry: pixels are a **direct linear readout** of bytes in the projection; physics labels are a **low-dimensional summary extracted via API**, so encoding models see pixels but not physics even though physics causally drives the scene.

**Scene design:** Object 0 (launcher, always visible), Object 1 (visible), Object 2 (occluded behind wall at y=0). Occlusion makes pixel-only models fundamentally limited for behavior prediction.

**`behavioral_objective`** in `config.yaml` switches between `"next_frame_pixels"` (MLP R²) and `"kinetic_energy"` (logistic accuracy) for the dissociation analysis. All pipeline parameters live in `config.yaml`; `config.py` is a backward-compat shim.
