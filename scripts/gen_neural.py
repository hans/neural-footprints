import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np

from load_config import load_config
from io_utils import load_scenes, save_neural
from neural_model import generate_neural_activity, print_variance_diagnostic

cfg = load_config()
scenes = load_scenes(snakemake.input.scenes)
pp = np.load(snakemake.input.pp_activations)

render_indices = scenes['metadata']['render_indices']
render = scenes['program_states'][:, render_indices]
hidden_acts = pp['hidden_acts']
inferred_physics = pp['inferred_physics']

D_render = render.shape[1]
D_hidden = hidden_acts.shape[1]
D_inferred = inferred_physics.shape[1]

neural_input = np.concatenate([render, hidden_acts, inferred_physics], axis=1).astype(np.float32)

neural_input_metadata = {
    'D_render': D_render,
    'D_hidden': D_hidden,
    'D_inferred': D_inferred,
    'D_total': D_render + D_hidden + D_inferred,
    'pp_layer': str(pp['layer']),
}

neural, neural_meta = generate_neural_activity(
    neural_input, cfg['random_seed'],
    n_neurons=cfg['n_neurons'], noise_level=cfg['noise_level'],
)
print_variance_diagnostic(neural_input_metadata, neural_meta)

save_neural(neural, neural_meta, snakemake.output.neural)
