"""
Snakemake script: run per-block P contribution diagnostic.

Inputs:  scenes, neural, pp_activations, forward_renders
Outputs: outputs/{norm}/p_block_contribution.json
         data/{norm}/p_block_plot_data.npz
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import json
import numpy as np

from load_config import load_config
from io_utils import load_scenes, load_neural, save_results
from analyses.p_block_contribution import run_p_block_contribution

cfg = load_config()

# -----------------------------------------------------------------
# Load data
# -----------------------------------------------------------------
scenes = load_scenes(snakemake.input.scenes)
neural, neural_meta = load_neural(snakemake.input.neural)
pp = np.load(snakemake.input.pp_activations)
fwd = np.load(snakemake.input.forward_renders)

render_indices = scenes["metadata"]["render_indices"]

# Reconstruct the 4 raw input blocks exactly as gen_neural.py does.
raw_frames = np.concatenate(
    [scenes["initial_renders"], scenes["early_renders"], scenes["late_renders"]], axis=1
).astype(np.float32)
fwd_render = fwd["forward_program_states"][:, render_indices].astype(np.float32)
hidden_acts = pp["hidden_acts"].astype(np.float32)
inferred_physics = pp["inferred_physics"].astype(np.float32)

blocks_raw = {
    "raw_frames": raw_frames,
    "fwd_render": fwd_render,
    "hidden_acts": hidden_acts,
    "inferred_physics": inferred_physics,
}

physics_labels = scenes["physics_labels"]

# If block_names/block_sizes are missing (old NPZ), fall back to defaults.
if neural_meta.get("block_names") is None:
    neural_meta["block_names"] = [
        "raw_frames", "fwd_render", "hidden_acts", "inferred_physics"
    ]
if neural_meta.get("block_sizes") is None:
    neural_meta["block_sizes"] = [
        raw_frames.shape[1],
        fwd_render.shape[1],
        hidden_acts.shape[1],
        inferred_physics.shape[1],
    ]

# block_norm must be populated for signal reconstruction
if neural_meta.get("block_norm") is None:
    # Infer from the path (the {norm} wildcard)
    norm_from_path = os.path.basename(os.path.dirname(snakemake.input.neural))
    neural_meta["block_norm"] = norm_from_path

# -----------------------------------------------------------------
# Run diagnostic
# -----------------------------------------------------------------
results = run_p_block_contribution(
    neural_meta=neural_meta,
    blocks_raw=blocks_raw,
    physics_labels=physics_labels,
    pixel_pca_dim=cfg["pixel_pca_dim"],
    seed=cfg["random_seed"],
)

# -----------------------------------------------------------------
# Save outputs
# -----------------------------------------------------------------
save_results(results, snakemake.output.results)

# NPZ for plotting (all numeric fields)
np.savez_compressed(
    snakemake.output.plot_data,
    block_names_json=np.array(json.dumps(results["block_names"])),
    block_sizes=np.array(results["block_sizes"], dtype=np.int64),
    block_k_values=np.array(results["block_k_values"], dtype=np.int64),
    var_share=np.array(results["var_share"]),
    var_share_independent=np.array(results["var_share_independent"]),
    r2_P_from_block_raw=np.array(results["r2_P_from_block_raw"]),
    r2_P_from_block_signal=np.array(results["r2_P_from_block_signal"]),
    effective_P_contribution=np.array(results["effective_P_contribution"]),
    r2_P_from_total_signal=np.array(results["r2_P_from_total_signal"]),
    norm=np.array(results["norm"]),
)
