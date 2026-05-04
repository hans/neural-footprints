"""CNN backbone for the InverseModel: 3-frame stack → conv tower → MLP head."""

import torch.nn as nn

from config import (
    PP_HIDDEN_DIM as _CFG_PP_HIDDEN_DIM,
    PP_DROPOUT_RATE as _CFG_PP_DROPOUT_RATE,
)


class InverseCNNNet(nn.Module):
    """Per-frame conv tower → global avg pool → concat across frames → MLP head.

    Input shape: ``(B, F, C, H, W)`` where F=frames (default 3) and C=channels
    (RGBA, 4). The conv tower is shared across frames to stay sample-efficient
    at n≈1700. Layers are split so post-ReLU hidden activations are addressable
    via ``forward_with_activations``, mirroring InverseMLPNet.
    """
    def __init__(self, n_frames, n_channels, output_dim,
                 hidden_dim=None, dropout_rate=None, batch_norm=False):
        super().__init__()
        if hidden_dim is None:
            hidden_dim = _CFG_PP_HIDDEN_DIM
        if dropout_rate is None:
            dropout_rate = _CFG_PP_DROPOUT_RATE
        self.n_frames = n_frames
        self.n_channels = n_channels
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.dropout_rate = dropout_rate
        self.batch_norm = batch_norm
        self.conv_feat_dim = 128

        bn2 = (lambda c: nn.BatchNorm2d(c)) if batch_norm else (lambda c: nn.Identity())
        bn1 = (lambda c: nn.BatchNorm1d(c)) if batch_norm else (lambda c: nn.Identity())

        # 64 → 32 → 16 → 8 (stride-2 each), then GAP → 128-dim per frame.
        self.conv = nn.Sequential(
            nn.Conv2d(n_channels, 32, kernel_size=3, stride=2, padding=1),
            bn2(32), nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            bn2(64), nn.ReLU(inplace=True),
            nn.Conv2d(64, self.conv_feat_dim, kernel_size=3, stride=2, padding=1),
            bn2(self.conv_feat_dim), nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
        )
        feat_dim = self.conv_feat_dim * n_frames

        self.h1 = nn.Sequential(nn.Linear(feat_dim, hidden_dim), bn1(hidden_dim), nn.ReLU())
        self.d1 = nn.Dropout(dropout_rate)
        self.h2 = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), bn1(hidden_dim), nn.ReLU())
        self.d2 = nn.Dropout(dropout_rate)
        self.h3 = nn.Sequential(nn.Linear(hidden_dim, hidden_dim // 2), bn1(hidden_dim // 2), nn.ReLU())
        self.d3 = nn.Dropout(dropout_rate)
        self.out = nn.Linear(hidden_dim // 2, output_dim)

    def _frame_features(self, x):
        B, F, C, H, W = x.shape
        return self.conv(x.reshape(B * F, C, H, W)).reshape(B, F * self.conv_feat_dim)

    def forward(self, x):
        feats = self._frame_features(x)
        h1 = self.h1(feats)
        h2 = self.h2(self.d1(h1))
        h3 = self.h3(self.d2(h2))
        return self.out(self.d3(h3))

    def forward_with_activations(self, x):
        feats = self._frame_features(x)
        h1 = self.h1(feats)
        h2 = self.h2(self.d1(h1))
        h3 = self.h3(self.d2(h2))
        out = self.out(self.d3(h3))
        return out, {'h1': h1, 'h2': h2, 'h3': h3}
