"""Persistence helpers for the CardinalityModel checkpoint.

Mirrors analyses/pp_io.py for the PP InverseModel. Stores the model state +
input/target scalers + the whitened-pixel PCA pipeline + per-dim val R² so
callers can reproduce the exact features the network was trained on.
"""

import torch

from analyses.cardinality import CardinalityMLPNet, CardinalityModel
from analyses.inferential_mlp import net_dims, extract_activations


__all__ = [
    'save_cardinality_model',
    'load_cardinality_model',
    'extract_activations',
]


def save_cardinality_model(model: CardinalityModel, pixel_scaler, pixel_pca, path):
    """Persist trained cardinality model + the pixel-PCA pipeline used to build features.

    The pixel-PCA pipeline (StandardScaler + PCA) is stored on the same
    checkpoint so downstream consumers (analysis, neural-projection) can
    apply the identical feature transform without recomputing PCA on a
    different sample.
    """
    torch.save({
        'state_dict':     model.net_.state_dict(),
        'dims':           net_dims(model.net_),
        'input_scaler':   model.input_scaler_,
        'target_scaler':  model.target_scaler_,
        'per_dim_r2':     model.per_dim_r2_,
        'hidden_dim':     model.hidden_dim,
        'dropout_rate':   model.dropout_rate,
        'pixel_scaler':   pixel_scaler,
        'pixel_pca':      pixel_pca,
    }, path)


def load_cardinality_model(path):
    """Reconstruct (CardinalityModel, pixel_scaler, pixel_pca) from a checkpoint."""
    ckpt = torch.load(path, weights_only=False)
    dims = ckpt['dims']
    m = CardinalityModel(
        hidden_dim=ckpt['hidden_dim'],
        dropout_rate=ckpt['dropout_rate'],
    )
    m.net_ = CardinalityMLPNet(
        input_dim=dims['input_dim'],
        output_dim=dims['output_dim'],
        hidden_dim=dims['hidden_dim'],
        dropout_rate=dims['dropout_rate'],
    )
    m.net_.load_state_dict(ckpt['state_dict'])
    m.net_.eval()
    m.input_scaler_  = ckpt['input_scaler']
    m.target_scaler_ = ckpt['target_scaler']
    m.per_dim_r2_    = ckpt['per_dim_r2']

    return m, ckpt['pixel_scaler'], ckpt['pixel_pca']
