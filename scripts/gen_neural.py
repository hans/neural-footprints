import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np

from load_config import load_config
from io_utils import load_scenes, save_neural
from neural_model import generate_neural_activity, print_variance_diagnostic

cfg = load_config()

# Load only the slices we need and free large arrays immediately to stay within
# memory budget. scenes (2.75 GB) and fwd (1.18 GB) must not both be live when
# generate_neural_activity runs (it needs ~2.4 GB for centered + W internally).
scenes = load_scenes(snakemake.input.scenes)
render_indices = scenes['metadata']['render_indices']
del scenes

pp = np.load(snakemake.input.pp_activations)
fwd = np.load(snakemake.input.forward_renders)

# Use forward-model render (predicted S from inferred P) instead of actual camera bytes.
# Brain state is now [fwd_render | inv_acts | P_hat]: raw sensation enters only
# through the inverse model's input, not as a direct component of the brain state.
render = fwd['forward_program_states'][:, render_indices]
hidden_acts = pp['hidden_acts']
inferred_physics = pp['inferred_physics']
pp_layer = str(pp['layer'])
del fwd, pp

D_render = render.shape[1]
D_hidden = hidden_acts.shape[1]
D_inferred = inferred_physics.shape[1]

# Fill neural_input in-place to avoid a large temporary from np.concatenate.
n_scenes = render.shape[0]
neural_input = np.empty((n_scenes, D_render + D_hidden + D_inferred), dtype=np.float32)
neural_input[:, :D_render] = render
neural_input[:, D_render:D_render + D_hidden] = hidden_acts
neural_input[:, D_render + D_hidden:] = inferred_physics
del render, hidden_acts, inferred_physics

block_sizes = [D_render, D_hidden, D_inferred]
block_names = ['render', 'hidden_acts', 'inferred_physics']

neural_input_metadata = {
    'D_render': D_render,
    'D_hidden': D_hidden,
    'D_inferred': D_inferred,
    'D_total': D_render + D_hidden + D_inferred,
    'pp_layer': pp_layer,
    # Keys expected by print_variance_diagnostic:
    'D_render_bytes': D_render,
    'D_physics_labels': D_hidden + D_inferred,
    'D_scene_config': 0,
}

neural, neural_meta = generate_neural_activity(
    neural_input, cfg['random_seed'],
    n_neurons=cfg['n_neurons'], noise_level=cfg['noise_level'],
    block_sizes=block_sizes,
)
print_variance_diagnostic(neural_input_metadata, neural_meta, block_sizes, block_names)

save_neural(neural, neural_meta, snakemake.output.neural)
