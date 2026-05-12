"""Persistence + activation tap helpers for the PP InverseModel.

Reads activations via the named-block layout of InverseMLPNet (h1/h2/h3 +
forward_with_activations). Lives in a separate module so callers don't depend
on whichever methods happen to be declared on InverseModel itself — the helpers
here construct the same checkpoint format regardless.
"""

import torch

from analyses.predictive_processing import InverseModel, InverseMLPNet


_LAYERS = ('h1', 'h2', 'h3')


def extract_activations(inv_model: InverseModel, pixel_pca_two_frame, layer='h2'):
    """Deterministic post-ReLU activations of one hidden layer (dropout off)."""
    if layer not in _LAYERS:
        raise ValueError(f"layer must be one of {_LAYERS}; got {layer!r}")
    inv_model.net_.eval()
    X_scaled = inv_model.input_scaler_.transform(pixel_pca_two_frame)
    X_t = torch.tensor(X_scaled, dtype=torch.float32)
    with torch.no_grad():
        _, acts = inv_model.net_.forward_with_activations(X_t)
    return acts[layer].numpy()


def _net_dims(net: InverseMLPNet):
    h1_linear = net.h1[0]
    return {
        'input_dim':    h1_linear.in_features,
        'hidden_dim':   h1_linear.out_features,
        'output_dim':   net.out.out_features,
        'dropout_rate': net.d1.p,
    }


def save_inverse_model(inv_model: InverseModel, path):
    """Persist trained model + scalers + observable-dim metadata to a single .pt file."""
    torch.save({
        'state_dict':       inv_model.net_.state_dict(),
        'dims':             _net_dims(inv_model.net_),
        'input_scaler':     inv_model.input_scaler_,
        'phys_scaler':      inv_model.phys_scaler_,
        'per_dim_r2':       inv_model.per_dim_r2_,
        'valid_dims':       inv_model.valid_dims_,
        'full_physics_dim': inv_model.full_physics_dim_,
        'const_values':     inv_model.const_values_,
    }, path)


def load_inverse_model(path) -> InverseModel:
    """Reconstruct an InverseModel from a checkpoint produced by save_inverse_model."""
    ckpt = torch.load(path, weights_only=False)
    dims = ckpt['dims']
    m = InverseModel()
    m.net_ = InverseMLPNet(
        input_dim=dims['input_dim'],
        output_dim=dims['output_dim'],
        hidden_dim=dims['hidden_dim'],
        dropout_rate=dims['dropout_rate'],
    )
    m.net_.load_state_dict(ckpt['state_dict'])
    m.net_.eval()
    m.input_scaler_      = ckpt['input_scaler']
    m.phys_scaler_       = ckpt['phys_scaler']
    m.per_dim_r2_        = ckpt['per_dim_r2']
    m.valid_dims_        = ckpt['valid_dims']
    m.full_physics_dim_  = ckpt['full_physics_dim']
    m.const_values_      = ckpt['const_values']
    return m
