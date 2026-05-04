"""Train the InverseModel that defines the cognitive layer of neural generation.

Outputs:
  data/inverse_model.pt   — checkpoint (state_dict + scalers + dim metadata)
  data/pp_activations.npz — per-scene hidden activations + inferred physics

The model is fit on ALL scenes (with an internal 15% held-out val split for early
stopping). Predictions for every scene are then dumped so gen_neural.py can stitch
them into the projection input. run_pp.py loads the same checkpoint to keep the
neural projection and the reported PP analysis numerically consistent.

Backbone is selected via ``cfg['pp_inverse_backbone']`` (defaults to 'mlp' if
the key is missing). Backbone-specific knobs:
  - 'mlp':         consumes ``cfg['pp_pixel_pca_dim']`` for the input PCA.
  - 'softmax_cnn': consumes the nested ``cfg['pp_softmax']`` block (n_filters,
                   learned_temp, hidden_dim, head_depth, training schedule).
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np

from load_config import load_config
from io_utils import load_scenes
from analyses.predictive_processing import make_inverse_model
from analyses.pp_io import save_inverse_model


cfg = load_config()
scenes = load_scenes(snakemake.input.scenes)

initial_physics = scenes['initial_physics_labels']
n = len(scenes['initial_renders'])

backbone = cfg.get('pp_inverse_backbone', 'mlp')
print(f"\nTraining InverseModel ({backbone}) for neural generation on {n} scenes")
print("=" * 60)

if backbone == 'mlp':
    inv = make_inverse_model('mlp', pixel_pca_dim=cfg['pp_pixel_pca_dim'])
    fit_kwargs = {}
elif backbone == 'softmax_cnn':
    sm_cfg = cfg.get('pp_softmax', {})
    inv = make_inverse_model(
        'softmax_cnn',
        n_filters=sm_cfg.get('n_filters', 128),
        learned_temp=sm_cfg.get('learned_temp', True),
        temp_per_channel=sm_cfg.get('temp_per_channel', True),
        include_variance=sm_cfg.get('include_variance', False),
        hidden_dim=sm_cfg.get('hidden_dim', 256),
        head_depth=sm_cfg.get('head_depth', 3),
        dropout_rate=sm_cfg.get('dropout_rate', 0.0),
    )
    fit_kwargs = {
        'n_epochs':   sm_cfg.get('n_epochs', 200),
        'patience':   sm_cfg.get('patience', 30),
        'min_epochs': sm_cfg.get('min_epochs', 60),
        'lr':         sm_cfg.get('lr', 1e-3),
        'batch_size': sm_cfg.get('batch_size', 64),
    }
else:
    raise ValueError(f"unknown pp_inverse_backbone: {backbone!r}")

inv_input = inv.prepare_input(scenes)
print(f"  input array: shape={inv_input.shape}  dtype={inv_input.dtype}")

inv.fit(inv_input, initial_physics, **fit_kwargs)

layer = cfg.get('pp_neural_layer', 'h2')
hidden_acts = inv.extract_activations(inv_input, layer=layer)
inferred_physics = inv.predict(inv_input)

print(f"\n  hidden_acts ({layer}): {hidden_acts.shape}")
print(f"  inferred_physics:       {inferred_physics.shape}")

save_inverse_model(inv, snakemake.output.model)
np.savez_compressed(
    snakemake.output.pp_acts,
    hidden_acts=hidden_acts,
    inferred_physics=inferred_physics,
    layer=np.array(layer),
)
print(f"\n  Saved checkpoint → {snakemake.output.model}")
print(f"  Saved activations → {snakemake.output.pp_acts}")
