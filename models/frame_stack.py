"""Shared 3-frame stack helpers for raw-pixel inverse-model backbones."""

import numpy as np

from config import IMAGE_SIZE


def build_frame_stack(scenes):
    """Stack initial / mid-or-early / late renders into ``(N, 3, C, H, W)``.

    Returns uint8. Float-normalize at the call site (typically
    ``frames.astype(np.float32) / 255.0``).
    """
    init = scenes["initial_renders"]
    if "mid_renders" in scenes and "late_renders" in scenes:
        f1, f2 = scenes["mid_renders"], scenes["late_renders"]
    elif "early_renders" in scenes and "late_renders" in scenes:
        f1, f2 = scenes["early_renders"], scenes["late_renders"]
    else:
        raise ValueError("scenes lacks 3 frames")
    n = init.shape[0]
    H = W = IMAGE_SIZE
    rgba_n = H * W * 4  # render vectors include depth+seg after RGBA; take only RGBA
    init, f1, f2 = init[:, :rgba_n], f1[:, :rgba_n], f2[:, :rgba_n]
    frames = np.stack([init, f1, f2], axis=1).astype(np.uint8)
    return frames.reshape(n, 3, H, W, 4).transpose(0, 1, 4, 2, 3)


def build_frame_stack_with_depth(scenes):
    """Stack 3 frames of RGBA + depth into (N, 3, 5, H, W) float32.

    Per-frame render layout: [RGBA_bytes (4*H*W) | depth_bytes (H*W*4 raw bytes of
    float32) | seg_bytes]. Depth is normalized to [0, 1] with sky pixels clipped at
    10 m so foreground variation is not compressed into a sliver of the range.

    Returns float32 (N, 3, 5, H, W): channels 0-3 are RGBA/255, channel 4 is depth.
    """
    init = scenes["initial_renders"]
    if "mid_renders" in scenes and "late_renders" in scenes:
        f1, f2 = scenes["mid_renders"], scenes["late_renders"]
    elif "early_renders" in scenes and "late_renders" in scenes:
        f1, f2 = scenes["early_renders"], scenes["late_renders"]
    else:
        raise ValueError("scenes lacks 3 frames")

    N = init.shape[0]
    H = W = IMAGE_SIZE
    rgba_n = H * W * 4
    depth_n_bytes = H * W * 4  # raw bytes of H*W float32 values

    def _extract(frame_data):
        rgba = frame_data[:, :rgba_n].astype(np.float32) / 255.0
        rgba = rgba.reshape(N, H, W, 4).transpose(0, 3, 1, 2)   # (N,4,H,W)
        raw = frame_data[:, rgba_n:rgba_n + depth_n_bytes]
        depth = raw.astype(np.uint8).view(np.float32)            # (N, H*W)
        return rgba, depth

    rgba_i, d_i = _extract(init)
    rgba_m, d_m = _extract(f1)
    rgba_l, d_l = _extract(f2)

    all_depth = np.clip(np.concatenate([d_i, d_m, d_l], axis=0), None, 10.0)
    d_min, d_range = all_depth.min(), max(all_depth.max() - all_depth.min(), 1e-6)

    def _norm(d):
        return ((np.clip(d, None, 10.0) - d_min) / d_range).reshape(N, 1, H, W)

    def _make5(rgba, d):
        return np.concatenate([rgba, _norm(d)], axis=1)          # (N,5,H,W)

    return np.stack([_make5(rgba_i, d_i), _make5(rgba_m, d_m), _make5(rgba_l, d_l)], axis=1)
