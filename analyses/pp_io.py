"""Backbone-aware persistence for the PP InverseModel wrappers.

Checkpoints carry a ``backbone`` tag so a single loader can reconstruct any of
the supported wrappers (currently 'mlp' / 'softmax_cnn'). Missing tag is read
as 'mlp' for backward compatibility with checkpoints written before the
pluggable-backbone refactor.
"""

import torch

from analyses.predictive_processing import (
    InverseModel,
    InverseSoftmaxCNN,
    INVERSE_BACKBONES,
)
from models import InverseMLPNet, SpatialSoftmaxV2


def _arch_kwargs_mlp(inv: InverseModel):
    h1_linear = inv.net_.h1[0]
    return {
        "input_dim": h1_linear.in_features,
        "hidden_dim": h1_linear.out_features,
        "output_dim": inv.net_.out.out_features,
        "dropout_rate": inv.net_.d1.p,
    }


def _arch_kwargs_softmax_cnn(inv: InverseSoftmaxCNN):
    net = inv.net_
    return {
        "n_frames": net.n_frames,
        "n_channels": net.n_channels,
        "image_size": net.image_size,
        "output_dim": net.output_dim,
        "n_filters": net.n_filters,
        "learned_temp": net.learned_temp,
        "temp_per_channel": net.temp_per_channel,
        "include_variance": net.include_variance,
        "hidden_dim": net.hidden_dim,
        "head_depth": net.head_depth,
        "dropout_rate": net.dropout_rate,
    }


def _backbone_of(inv) -> str:
    if isinstance(inv, InverseModel):
        return "mlp"
    if isinstance(inv, InverseSoftmaxCNN):
        return "softmax_cnn"
    raise TypeError(
        f"unsupported wrapper type {type(inv).__name__}; "
        f"expected one of {INVERSE_BACKBONES}"
    )


def save_inverse_model(inv, path):
    """Persist trained model + scalers + observable-dim metadata to a single .pt file."""
    backbone = _backbone_of(inv)
    if backbone == "mlp":
        arch_kwargs = _arch_kwargs_mlp(inv)
    else:
        arch_kwargs = _arch_kwargs_softmax_cnn(inv)

    torch.save(
        {
            "backbone": backbone,
            "arch_kwargs": arch_kwargs,
            "state_dict": inv.net_.state_dict(),
            "input_scaler": inv.input_scaler_,  # None for raw-frame backbones
            "phys_scaler": inv.phys_scaler_,
            "per_dim_r2": inv.per_dim_r2_,
            "valid_dims": inv.valid_dims_,
            "full_physics_dim": inv.full_physics_dim_,
            "const_values": inv.const_values_,
        },
        path,
    )


def _load_mlp(ckpt) -> InverseModel:
    arch = ckpt["arch_kwargs"]
    m = InverseModel()
    m.net_ = InverseMLPNet(
        input_dim=arch["input_dim"],
        output_dim=arch["output_dim"],
        hidden_dim=arch["hidden_dim"],
        dropout_rate=arch["dropout_rate"],
    )
    m.net_.load_state_dict(ckpt["state_dict"])
    m.net_.eval()
    m.input_scaler_ = ckpt["input_scaler"]
    return m


def _load_softmax_cnn(ckpt) -> InverseSoftmaxCNN:
    arch = ckpt["arch_kwargs"]
    m = InverseSoftmaxCNN(
        n_filters=arch["n_filters"],
        learned_temp=arch["learned_temp"],
        temp_per_channel=arch["temp_per_channel"],
        include_variance=arch["include_variance"],
        hidden_dim=arch["hidden_dim"],
        head_depth=arch["head_depth"],
        dropout_rate=arch["dropout_rate"],
    )
    m.net_ = SpatialSoftmaxV2(
        n_frames=arch["n_frames"],
        n_channels=arch["n_channels"],
        image_size=arch["image_size"],
        output_dim=arch["output_dim"],
        n_filters=arch["n_filters"],
        learned_temp=arch["learned_temp"],
        temp_per_channel=arch["temp_per_channel"],
        include_variance=arch["include_variance"],
        hidden_dim=arch["hidden_dim"],
        head_depth=arch["head_depth"],
        dropout_rate=arch["dropout_rate"],
    )
    m.net_.load_state_dict(ckpt["state_dict"])
    m.net_.eval()
    m.input_scaler_ = ckpt["input_scaler"]  # None for this backbone
    return m


def load_inverse_model(path):
    """Reconstruct an inverse-model wrapper from a checkpoint.

    Dispatches on the checkpoint's ``backbone`` tag. Tag is missing on
    pre-refactor checkpoints; those are read as the MLP backbone.
    """
    ckpt = torch.load(path, weights_only=False)
    backbone = ckpt.get("backbone", "mlp")

    if backbone == "mlp":
        m = _load_mlp(ckpt)
    elif backbone == "softmax_cnn":
        m = _load_softmax_cnn(ckpt)
    else:
        raise ValueError(
            f"unknown backbone {backbone!r} in checkpoint; "
            f"expected one of {INVERSE_BACKBONES}"
        )

    m.phys_scaler_ = ckpt["phys_scaler"]
    m.per_dim_r2_ = ckpt["per_dim_r2"]
    m.valid_dims_ = ckpt["valid_dims"]
    m.full_physics_dim_ = ckpt["full_physics_dim"]
    m.const_values_ = ckpt["const_values"]
    return m
