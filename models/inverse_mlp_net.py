"""MLP backbone for the InverseModel: pixel-PCA features → physics."""

import torch.nn as nn

from config import (
    PP_HIDDEN_DIM as _CFG_PP_HIDDEN_DIM,
    PP_DROPOUT_RATE as _CFG_PP_DROPOUT_RATE,
)


class InverseMLPNet(nn.Module):
    """Three-hidden-layer MLP with dropout, kept active at inference for MC sampling.

    Layers are split into named blocks so post-ReLU activations of each hidden
    layer are individually addressable via ``forward_with_activations``. Dropout
    is applied AFTER the activation tap, so a tapped activation reflects the
    deterministic representation when the net is in eval mode (and the
    stochastic representation when in train mode for MC sampling).
    """
    def __init__(self, input_dim, output_dim,
                 hidden_dim=None, dropout_rate=None):
        super().__init__()
        if hidden_dim is None:
            hidden_dim = _CFG_PP_HIDDEN_DIM
        if dropout_rate is None:
            dropout_rate = _CFG_PP_DROPOUT_RATE
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.dropout_rate = dropout_rate

        self.h1 = nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.ReLU())
        self.d1 = nn.Dropout(dropout_rate)
        self.h2 = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.ReLU())
        self.d2 = nn.Dropout(dropout_rate)
        self.h3 = nn.Sequential(nn.Linear(hidden_dim, hidden_dim // 2), nn.ReLU())
        self.d3 = nn.Dropout(dropout_rate)
        self.out = nn.Linear(hidden_dim // 2, output_dim)

    def forward(self, x):
        h1 = self.h1(x)
        h2 = self.h2(self.d1(h1))
        h3 = self.h3(self.d2(h2))
        return self.out(self.d3(h3))

    def forward_with_activations(self, x):
        """Returns (output, {'h1': ..., 'h2': ..., 'h3': ...}). Post-ReLU, pre-dropout."""
        h1 = self.h1(x)
        h2 = self.h2(self.d1(h1))
        h3 = self.h3(self.d2(h2))
        out = self.out(self.d3(h3))
        return out, {'h1': h1, 'h2': h2, 'h3': h3}
