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

**Data flow:** See `SPEC.md` for full details. In brief: PyBullet scenes produce a `program_state` vector (render bytes + physics labels + scene config + scene lighting), which is randomly projected into `neural_activity`. Render dimensions vastly outnumber non-render dimensions, creating the information asymmetry that drives all results.

**`behavioral_objective`** in `config.yaml` switches between `"next_frame_pixels"` (MLP R²) and `"kinetic_energy"` (logistic accuracy) for the dissociation analysis. All pipeline parameters live in `config.yaml`; `config.py` is a backward-compat shim.
