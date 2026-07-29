"""Pure ``nn.Module`` architectures for the InverseModel pipeline.

Wrappers (fit/predict/scaler logic) live in ``analyses/predictive_processing.py``.
This package owns only the network definitions so they can be swapped in via
the ``pp_inverse_backbone`` config flag.
"""

from .inverse_mlp_net import InverseMLPNet
from .inverse_cnn_net import InverseCNNNet
from .spatial_softmax import (
    SpatialSoftmaxV2,
    SpatialSoftmaxTemporalDelta,
    SpatialSoftmaxDepthGated,
    SpatialSoftmaxDepthGatedTemporalDelta,
)
from .frame_stack import build_frame_stack, build_frame_stack_with_depth

__all__ = [
    "InverseMLPNet",
    "InverseCNNNet",
    "SpatialSoftmaxV2",
    "SpatialSoftmaxTemporalDelta",
    "SpatialSoftmaxDepthGated",
    "SpatialSoftmaxDepthGatedTemporalDelta",
    "build_frame_stack",
    "build_frame_stack_with_depth",
]
