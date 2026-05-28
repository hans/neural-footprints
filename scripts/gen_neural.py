import sys, os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np

from load_config import load_config
from io_utils import load_scenes, save_neural
from neural_model import generate_neural_activity

cfg = load_config()

scenes = load_scenes(snakemake.input.scenes)
render_indices = scenes["metadata"]["render_indices"]
# Brain state block 1: raw observed frames (initial + early + late).
raw_frames = np.concatenate(
    [scenes["initial_renders"], scenes["early_renders"], scenes["late_renders"]], axis=1
).astype(np.float32)
del scenes

pp = np.load(snakemake.input.pp_activations)
fwd = np.load(snakemake.input.forward_renders)

# Brain state block 2: forward-model render (predicted S from inferred P).
render = fwd["forward_program_states"][:, render_indices].astype(np.float32)
hidden_acts = pp["hidden_acts"].astype(np.float32)
inferred_physics = pp["inferred_physics"].astype(np.float32)
del fwd, pp

n_scenes = raw_frames.shape[0]
D_raw = raw_frames.shape[1]
D_render = render.shape[1]
D_hidden = hidden_acts.shape[1]
D_inferred = inferred_physics.shape[1]
D_total = D_raw + D_render + D_hidden + D_inferred

neural_input = np.empty((n_scenes, D_total), dtype=np.float32)
neural_input[:, :D_raw] = raw_frames
neural_input[:, D_raw : D_raw + D_render] = render
neural_input[:, D_raw + D_render : D_raw + D_render + D_hidden] = hidden_acts
neural_input[:, D_raw + D_render + D_hidden :] = inferred_physics
del raw_frames, render, hidden_acts, inferred_physics

block_sizes = [D_raw, D_render, D_hidden, D_inferred]

block_norm = snakemake.params.get("block_norm", cfg.get("block_norm", "truncated_svd"))

neural, neural_meta = generate_neural_activity(
    neural_input,
    cfg["random_seed"],
    n_neurons=cfg["n_neurons"],
    noise_level=cfg["noise_level"],
    block_sizes=block_sizes,
    block_norm=block_norm,
)

block_names = ["raw_frames", "fwd_render", "hidden_acts", "inferred_physics"]
print(
    f"stable_rank_trunc: k={neural_meta['block_k_values'].tolist()} "
    f"(sr={[f'{r:.1f}' for r in neural_meta['block_stable_ranks'].tolist()]}) "
    f"D_proj={neural_meta['W'].shape[1]}"
)
print(f"  blocks: {dict(zip(block_names, block_sizes))}")

save_neural(neural, neural_meta, snakemake.output.neural)
