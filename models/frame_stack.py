"""Shared 3-frame stack helper for raw-pixel inverse-model backbones."""

import numpy as np

from config import IMAGE_SIZE


def build_frame_stack(scenes):
    """Stack initial / mid-or-early / late renders into ``(N, 3, C, H, W)``.

    Returns uint8. Float-normalize at the call site (typically
    ``frames.astype(np.float32) / 255.0``).
    """
    init = scenes['initial_renders']
    if 'mid_renders' in scenes and 'late_renders' in scenes:
        f1, f2 = scenes['mid_renders'], scenes['late_renders']
    elif 'early_renders' in scenes and 'late_renders' in scenes:
        f1, f2 = scenes['early_renders'], scenes['late_renders']
    else:
        raise ValueError("scenes lacks 3 frames")
    n = init.shape[0]
    H = W = IMAGE_SIZE
    frames = np.stack([init, f1, f2], axis=1).astype(np.uint8)
    return frames.reshape(n, 3, H, W, 4).transpose(0, 1, 4, 2, 3)
