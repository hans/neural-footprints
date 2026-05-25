"""SpatialSoftmaxV2 backbone — per-frame softmax-keypoint encoder.

Same architecture as the v2_128_temp_mlp config in scripts/eval_pp_cnn_softmax_sweep.py
that beat gridpool at ~21% the params (mean valid-dim R² 0.828 vs 0.796), with
two additions for use as the InverseModel backbone:

  * Dropout in the head's hidden Linears, so MC-dropout uncertainty samples
    can be drawn via ``predict_stochastic`` (the conv tower stays in eval —
    no dropout there → no point).
  * ``forward_with_activations`` returning the post-ReLU activations of each
    head layer (h1=keypoint coords, h2=first hidden, h3=second hidden), so
    the layer-tap protocol matches InverseMLPNet / InverseCNNNet.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SpatialSoftmaxV2(nn.Module):
    """Per-frame softmax-keypoint encoder with learnable temperature + MLP head.

    Output features per channel: E[x], E[y], and optionally E[x²], E[y²]
    (the latter pair captures keypoint spread/scale — useful when an object
    grows or rotates in-plane and the activation map widens).

    Layer protocol (consumed by ``extract_activations``):

      - ``h1`` — flat keypoint coords (post-softmax E[x],E[y] per channel per
        frame), shape ``(B, n_frames * n_filters * per_channel_feats)``.
      - ``h2`` — first head Linear, post-ReLU. Aliased to ``h1`` when
        ``head_depth == 1``.
      - ``h3`` — second head Linear, post-ReLU. Aliased to ``h2`` when
        ``head_depth == 2`` (and to ``h1`` when ``head_depth == 1``).
    """

    def __init__(self, n_frames, n_channels, image_size, output_dim, *,
                 n_filters=64, learned_temp=True, temp_per_channel=True,
                 include_variance=False, hidden_dim=128, head_depth=2,
                 dropout_rate=0.0):
        super().__init__()
        self.n_frames = n_frames
        self.n_channels = n_channels
        self.image_size = image_size
        self.output_dim = output_dim
        self.n_filters = n_filters
        self.learned_temp = learned_temp
        self.temp_per_channel = temp_per_channel
        self.include_variance = include_variance
        self.hidden_dim = hidden_dim
        self.head_depth = head_depth
        self.dropout_rate = dropout_rate

        mid = max(32, n_filters // 2)
        self.conv = nn.Sequential(
            nn.Conv2d(n_channels, 32, kernel_size=5, stride=2, padding=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, mid, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid, n_filters, kernel_size=3, padding=1),
        )

        sm_size = image_size // 4
        ys = torch.linspace(-1.0, 1.0, sm_size)
        xs = torch.linspace(-1.0, 1.0, sm_size)
        gy, gx = torch.meshgrid(ys, xs, indexing='ij')
        self.register_buffer('grid_x', gx.reshape(-1), persistent=False)
        self.register_buffer('grid_y', gy.reshape(-1), persistent=False)

        # Inverse temperature β so β=1 reproduces a standard softmax.
        # Parameterise as log_β so β stays positive without a clamp.
        if learned_temp:
            shape = (n_filters,) if temp_per_channel else (1,)
            self.log_beta = nn.Parameter(torch.zeros(shape))
        else:
            self.register_buffer('log_beta', torch.zeros(1), persistent=False)

        per_channel_feats = 4 if include_variance else 2
        feat_dim = n_frames * n_filters * per_channel_feats
        self.feat_dim = feat_dim

        # Head as a list of (Linear, ReLU, Dropout) blocks + final Linear.
        # Splitting it (vs a single nn.Sequential) lets forward_with_activations
        # tap post-ReLU activations of each hidden block by name.
        hidden_layers = max(0, head_depth - 1)
        self.head_hidden = nn.ModuleList()
        d = feat_dim
        for _ in range(hidden_layers):
            self.head_hidden.append(nn.Sequential(
                nn.Linear(d, hidden_dim), nn.ReLU(inplace=True)
            ))
            d = hidden_dim
        self.head_dropouts = nn.ModuleList(
            [nn.Dropout(dropout_rate) for _ in range(hidden_layers)]
        )
        self.head_out = nn.Linear(d, output_dim)

    def _keypoint_features(self, x):
        B, F_, C, H, W = x.shape
        feats = self.conv(x.reshape(B * F_, C, H, W))      # (B*F, K, H', W')
        K, Hs, Ws = feats.shape[1], feats.shape[2], feats.shape[3]
        flat = feats.reshape(B * F_, K, Hs * Ws)

        beta = self.log_beta.exp()
        if beta.numel() == 1:
            scaled = flat * beta
        else:
            scaled = flat * beta.reshape(1, K, 1)
        attn = F.softmax(scaled, dim=-1)                   # (B*F, K, H'*W')

        ex = (attn * self.grid_x).sum(dim=-1)              # (B*F, K)
        ey = (attn * self.grid_y).sum(dim=-1)
        if self.include_variance:
            ex2 = (attn * self.grid_x.pow(2)).sum(dim=-1)
            ey2 = (attn * self.grid_y.pow(2)).sum(dim=-1)
            coords = torch.stack([ex, ey, ex2, ey2], dim=-1)
        else:
            coords = torch.stack([ex, ey], dim=-1)
        return coords.reshape(B, -1)

    def _keypoint_coords_per_frame(self, x):
        """Returns per-frame keypoint coords as (B, F, K*per_channel_feats).

        Same conv pass as ``_keypoint_features`` but without the final flatten
        across frames, so the caller can compute temporal deltas.
        """
        B, F_, C, H, W = x.shape
        feats = self.conv(x.reshape(B * F_, C, H, W))      # (B*F, K, H', W')
        K, Hs, Ws = feats.shape[1], feats.shape[2], feats.shape[3]
        flat = feats.reshape(B * F_, K, Hs * Ws)

        beta = self.log_beta.exp()
        if beta.numel() == 1:
            scaled = flat * beta
        else:
            scaled = flat * beta.reshape(1, K, 1)
        attn = F.softmax(scaled, dim=-1)                   # (B*F, K, H'*W')

        ex = (attn * self.grid_x).sum(dim=-1)              # (B*F, K)
        ey = (attn * self.grid_y).sum(dim=-1)
        if self.include_variance:
            ex2 = (attn * self.grid_x.pow(2)).sum(dim=-1)
            ey2 = (attn * self.grid_y.pow(2)).sum(dim=-1)
            coords = torch.stack([ex, ey, ex2, ey2], dim=-1)  # (B*F, K, 4)
        else:
            coords = torch.stack([ex, ey], dim=-1)             # (B*F, K, 2)
        per_channel_feats = coords.shape[-1]
        return coords.reshape(B, F_, K * per_channel_feats)    # (B, F, K*pcf)

    def forward(self, x):
        h = self._keypoint_features(x)
        for block, drop in zip(self.head_hidden, self.head_dropouts):
            h = drop(block(h))
        return self.head_out(h)

    def forward_with_activations(self, x):
        """Returns (output, {'h1': ..., 'h2': ..., 'h3': ...}).

        h1 is always the flat keypoint feature vector. h2/h3 are post-ReLU
        outputs of the head's hidden Linears (pre-dropout). When head_depth
        is shallower, h2/h3 alias upward (h2→h1, h3→h2).
        """
        h1 = self._keypoint_features(x)
        h_acts = [h1]
        h = h1
        for block, drop in zip(self.head_hidden, self.head_dropouts):
            h = block(h)
            h_acts.append(h)
            h = drop(h)
        out = self.head_out(h)
        # h_acts has 1 + hidden_layers entries: [h1, head1?, head2?].
        # Pad up to length 3 by aliasing.
        while len(h_acts) < 3:
            h_acts.append(h_acts[-1])
        return out, {'h1': h_acts[0], 'h2': h_acts[1], 'h3': h_acts[2]}


class SpatialSoftmaxDepthGated(SpatialSoftmaxV2):
    """SpatialSoftmaxV2 with learned depth-gating to suppress background keypoints.

    The conv tower sees only RGBA (4 channels). A learnable gate weights each
    spatial position by exp(-γ · depth), so foreground (depth≈0) is preserved
    and background (depth≈1) is suppressed before the softmax.

    n_channels must be 5 (RGBA + depth); depth_gamma is learned in log-space
    so γ stays positive without clamping.
    """

    def __init__(self, n_frames, n_channels, image_size, output_dim, *,
                 depth_gamma_init=2.0, **kwargs):
        if n_channels != 5:
            raise ValueError(f"SpatialSoftmaxDepthGated requires n_channels=5, got {n_channels}")
        super().__init__(n_frames, 4, image_size, output_dim, **kwargs)
        self.depth_gamma = nn.Parameter(torch.log(torch.tensor(float(depth_gamma_init))))

    def _prepare_inputs(self, x):
        """Split (B,F,5,H,W) into RGBA (B*F,4,H,W) and depth_flat (B*F,1,H'*W')."""
        B, F_, _, H, W = x.shape
        x_rgba = x[:, :, :4].reshape(B * F_, 4, H, W)
        x_depth = x[:, :, 4:5].reshape(B * F_, 1, H, W)
        sm_h, sm_w = H // 4, W // 4
        depth_down = F.adaptive_avg_pool2d(x_depth, (sm_h, sm_w))   # (B*F, 1, H', W')
        depth_flat = depth_down.reshape(B * F_, 1, sm_h * sm_w)
        return x_rgba, depth_flat, B, F_

    def _keypoint_features(self, x_rgba, depth_flat, B):
        """Depth-gated keypoint extraction.

        x_rgba: (B*F, 4, H, W)
        depth_flat: (B*F, 1, H'*W')
        Returns: (B, feat_dim)
        """
        feats = self.conv(x_rgba)                                    # (B*F, K, H', W')
        K, Hs, Ws = feats.shape[1], feats.shape[2], feats.shape[3]
        flat = feats.reshape(feats.shape[0], K, Hs * Ws)

        beta = self.log_beta.exp()
        gate = torch.exp(-self.depth_gamma.exp() * depth_flat)       # (B*F, 1, H'*W')
        scaled = flat * (beta.reshape(1, K, 1) if beta.numel() > 1 else beta) * gate
        attn = F.softmax(scaled, dim=-1)                             # (B*F, K, H'*W')

        ex = (attn * self.grid_x).sum(dim=-1)
        ey = (attn * self.grid_y).sum(dim=-1)
        if self.include_variance:
            ex2 = (attn * self.grid_x.pow(2)).sum(dim=-1)
            ey2 = (attn * self.grid_y.pow(2)).sum(dim=-1)
            coords = torch.stack([ex, ey, ex2, ey2], dim=-1)
        else:
            coords = torch.stack([ex, ey], dim=-1)
        return coords.reshape(B, -1)

    def forward(self, x):
        x_rgba, depth_flat, B, _ = self._prepare_inputs(x)
        h = self._keypoint_features(x_rgba, depth_flat, B)
        for block, drop in zip(self.head_hidden, self.head_dropouts):
            h = drop(block(h))
        return self.head_out(h)

    def forward_with_activations(self, x):
        x_rgba, depth_flat, B, _ = self._prepare_inputs(x)
        h1 = self._keypoint_features(x_rgba, depth_flat, B)
        h_acts = [h1]
        h = h1
        for block, drop in zip(self.head_hidden, self.head_dropouts):
            h = block(h)
            h_acts.append(h)
            h = drop(h)
        out = self.head_out(h)
        while len(h_acts) < 3:
            h_acts.append(h_acts[-1])
        return out, {'h1': h_acts[0], 'h2': h_acts[1], 'h3': h_acts[2]}


class SpatialSoftmaxTemporalDelta(SpatialSoftmaxV2):
    """SpatialSoftmaxV2 with temporal keypoint-delta augmentation.

    After computing per-frame keypoint coords, the feature vector fed into the
    MLP head is:

        [kp_t0, kp_t1, ..., kp_tF, kp_t1 - kp_t0, kp_t2 - kp_t1, ...]

    This appends (F-1) velocity-like delta vectors for free, giving the head
    explicit motion information in keypoint space.

    The head input dimension is ``(2*n_frames - 1) * n_filters * per_channel_feats``
    instead of ``n_frames * n_filters * per_channel_feats``.
    """

    def __init__(self, n_frames, n_channels, image_size, output_dim, **kwargs):
        # Build base with standard feat_dim first (sets up conv / log_beta / grid).
        super().__init__(n_frames, n_channels, image_size, output_dim, **kwargs)

        # Re-compute head with augmented input dim.
        per_channel_feats = 4 if self.include_variance else 2
        frame_kp_dim = n_frames * self.n_filters * per_channel_feats
        delta_kp_dim = (n_frames - 1) * self.n_filters * per_channel_feats
        aug_feat_dim = frame_kp_dim + delta_kp_dim

        hidden_dim = self.hidden_dim
        head_depth = self.head_depth
        dropout_rate = self.dropout_rate

        hidden_layers = max(0, head_depth - 1)
        self.head_hidden = nn.ModuleList()
        d = aug_feat_dim
        for _ in range(hidden_layers):
            self.head_hidden.append(nn.Sequential(
                nn.Linear(d, hidden_dim), nn.ReLU(inplace=True)
            ))
            d = hidden_dim
        self.head_dropouts = nn.ModuleList(
            [nn.Dropout(dropout_rate) for _ in range(hidden_layers)]
        )
        self.head_out = nn.Linear(d, output_dim)

        # Store for reference
        self.aug_feat_dim = aug_feat_dim

    def _augmented_features(self, x):
        """Returns augmented feature vector [coords | deltas] as (B, aug_feat_dim)."""
        # coords: (B, F, K*per_channel_feats)
        coords = self._keypoint_coords_per_frame(x)
        deltas = coords[:, 1:] - coords[:, :-1]      # (B, F-1, K*per_channel_feats)
        B = coords.shape[0]
        return torch.cat([coords, deltas], dim=1).reshape(B, -1)

    def forward(self, x):
        h = self._augmented_features(x)
        for block, drop in zip(self.head_hidden, self.head_dropouts):
            h = drop(block(h))
        return self.head_out(h)

    def forward_with_activations(self, x):
        """Returns (output, {'h1': ..., 'h2': ..., 'h3': ...}).

        h1 is the augmented flat keypoint + delta feature vector.
        """
        h1 = self._augmented_features(x)
        h_acts = [h1]
        h = h1
        for block, drop in zip(self.head_hidden, self.head_dropouts):
            h = block(h)
            h_acts.append(h)
            h = drop(h)
        out = self.head_out(h)
        while len(h_acts) < 3:
            h_acts.append(h_acts[-1])
        return out, {'h1': h_acts[0], 'h2': h_acts[1], 'h3': h_acts[2]}


class SpatialSoftmaxDepthGatedTemporalDelta(SpatialSoftmaxTemporalDelta):
    """SpatialSoftmaxTemporalDelta with depth-gating.

    Combines per-frame depth-gated keypoint extraction (from SpatialSoftmaxDepthGated)
    with keypoint-velocity delta augmentation (from SpatialSoftmaxTemporalDelta).
    n_channels must be 5 (RGBA + depth).
    """

    def __init__(self, n_frames, n_channels, image_size, output_dim, *,
                 depth_gamma_init=2.0, **kwargs):
        if n_channels != 5:
            raise ValueError(
                f"SpatialSoftmaxDepthGatedTemporalDelta requires n_channels=5, got {n_channels}")
        super().__init__(n_frames, 4, image_size, output_dim, **kwargs)
        self.depth_gamma = nn.Parameter(torch.log(torch.tensor(float(depth_gamma_init))))

    def _prepare_inputs(self, x):
        B, F_, _, H, W = x.shape
        x_rgba = x[:, :, :4].reshape(B * F_, 4, H, W)
        x_depth = x[:, :, 4:5].reshape(B * F_, 1, H, W)
        sm_h, sm_w = H // 4, W // 4
        depth_down = F.adaptive_avg_pool2d(x_depth, (sm_h, sm_w))
        depth_flat = depth_down.reshape(B * F_, 1, sm_h * sm_w)
        return x_rgba, depth_flat, B, F_

    def _depth_gated_coords_per_frame(self, x_rgba, depth_flat, B, F_):
        """Depth-gated per-frame keypoint coords: (B, F, K*per_channel_feats)."""
        feats = self.conv(x_rgba)
        K, Hs, Ws = feats.shape[1], feats.shape[2], feats.shape[3]
        flat = feats.reshape(B * F_, K, Hs * Ws)

        beta = self.log_beta.exp()
        gate = torch.exp(-self.depth_gamma.exp() * depth_flat)
        scaled = flat * (beta.reshape(1, K, 1) if beta.numel() > 1 else beta) * gate
        attn = F.softmax(scaled, dim=-1)

        ex = (attn * self.grid_x).sum(dim=-1)
        ey = (attn * self.grid_y).sum(dim=-1)
        if self.include_variance:
            ex2 = (attn * self.grid_x.pow(2)).sum(dim=-1)
            ey2 = (attn * self.grid_y.pow(2)).sum(dim=-1)
            coords = torch.stack([ex, ey, ex2, ey2], dim=-1)
        else:
            coords = torch.stack([ex, ey], dim=-1)
        per_channel_feats = coords.shape[-1]
        return coords.reshape(B, F_, K * per_channel_feats)

    def forward(self, x):
        x_rgba, depth_flat, B, F_ = self._prepare_inputs(x)
        coords = self._depth_gated_coords_per_frame(x_rgba, depth_flat, B, F_)
        deltas = coords[:, 1:] - coords[:, :-1]
        h = torch.cat([coords, deltas], dim=1).reshape(B, -1)
        for block, drop in zip(self.head_hidden, self.head_dropouts):
            h = drop(block(h))
        return self.head_out(h)

    def forward_with_activations(self, x):
        x_rgba, depth_flat, B, F_ = self._prepare_inputs(x)
        coords = self._depth_gated_coords_per_frame(x_rgba, depth_flat, B, F_)
        deltas = coords[:, 1:] - coords[:, :-1]
        h1 = torch.cat([coords, deltas], dim=1).reshape(B, -1)
        h_acts = [h1]
        h = h1
        for block, drop in zip(self.head_hidden, self.head_dropouts):
            h = block(h)
            h_acts.append(h)
            h = drop(h)
        out = self.head_out(h)
        while len(h_acts) < 3:
            h_acts.append(h_acts[-1])
        return out, {'h1': h_acts[0], 'h2': h_acts[1], 'h3': h_acts[2]}
