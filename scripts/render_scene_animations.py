"""Render mp4 animations of a random sample of scenes (optional pipeline step).

Input: data/scenes.npz
Output: figures/scene_animations/scene_<idx>.mp4   (one per sampled scene)
        figures/scene_animations_grid.mp4          (combined grid montage)

Unlike the main pipeline (which renders only 4 frames per scene at 64x64), this
re-simulates each sampled scene and renders RGB at every intermediate physics
timestep at a presentation-friendly resolution. The grid montage plays all
sampled scenes in sync so the scene-to-scene variation (shape, size, color,
motion, lighting, background) is obvious at a glance.

Reuses scene_generator.render_scene_frames, mirroring the stored-data
reconstruction pattern in gen_forward_renders.py.
"""

import sys, os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import imageio.v2 as imageio

from io_utils import load_scenes
from load_config import load_config
from scene_generator import render_scene_frames


def _write_mp4(path, frames, fps, reps=1):
    """Write a uint8 [T, H, W, 3] array as an h264 mp4.

    `reps` repeats the whole sequence (appended frame-by-frame, so no large
    in-memory tiling) to produce a longer looped clip.
    """
    writer = imageio.get_writer(
        path,
        fps=fps,
        codec="libx264",
        quality=8,
        macro_block_size=16,
        pixelformat="yuv420p",
    )
    try:
        for _ in range(reps):
            for frame in frames:
                writer.append_data(frame)
    finally:
        writer.close()


def _reps_for_duration(n_frames, fps, seconds):
    """Number of sequence repeats so the clip lasts at least `seconds`."""
    if n_frames <= 0:
        return 1
    return max(1, int(np.ceil(seconds * fps / n_frames)))


def _build_grid(scene_frames, cols, decimate):
    """Tile per-scene frame stacks into a single [T, gridH, gridW, 3] montage.

    scene_frames: list of uint8 [T, H, W, 3] (all same T, H, W).
    Each cell is integer-decimated by `decimate` (e.g. 384 -> 192) to keep the
    montage a reasonable size. Empty cells (when n_scenes < cols*rows) are black.
    """
    cells = [f[:, ::decimate, ::decimate] for f in scene_frames]
    T, ch, cw, _ = cells[0].shape
    n = len(cells)
    rows = (n + cols - 1) // cols
    grid = np.zeros((T, rows * ch, cols * cw, 3), dtype=np.uint8)
    for k, cell in enumerate(cells):
        r, c = divmod(k, cols)
        grid[:, r * ch : (r + 1) * ch, c * cw : (c + 1) * cw] = cell
    return grid


cfg = load_config()
scenes = load_scenes(snakemake.input.scenes)

n_sample = int(cfg["animation_n_scenes"])
seed = int(cfg["animation_seed"])
res = int(cfg["animation_resolution"])
fps = int(cfg["animation_fps"])
stride = int(cfg["animation_stride"])
cols = int(cfg["animation_grid_cols"])
decimate = int(cfg["animation_montage_decimate"])
loop_seconds = int(cfg["animation_loop_seconds"])

scene_configs = scenes["scene_configs"]
initial_physics = scenes["initial_physics_labels"]
pillar_grays = scenes["pillar_grays"]
lightings = scenes["lightings"]
n_scenes = len(scene_configs)

rng = np.random.default_rng(seed)
n_sample = min(n_sample, n_scenes)
idx = np.sort(rng.choice(n_scenes, size=n_sample, replace=False))

out_dir = snakemake.output.scenes_dir
os.makedirs(out_dir, exist_ok=True)

print(
    f"Rendering {n_sample} scene animations at {res}x{res}, "
    f"stride={stride}, fps={fps} (indices: {idx.tolist()})"
)

scene_frames = []
for i in idx:
    frames = render_scene_frames(
        scene_configs[i],
        initial_physics[i],
        pillar_gray=pillar_grays[i],
        lighting=lightings[i],
        render_size=res,
        stride=stride,
    )
    reps = _reps_for_duration(frames.shape[0], fps, loop_seconds)
    path = os.path.join(out_dir, f"scene_{int(i):04d}.mp4")
    loop_path = os.path.join(out_dir, f"scene_{int(i):04d}_loop.mp4")
    _write_mp4(path, frames, fps)
    _write_mp4(loop_path, frames, fps, reps=reps)
    scene_frames.append(frames)
    print(
        f"  scene {int(i):04d}: {frames.shape[0]} frames -> {path} "
        f"(+ {reps}x loop ~{frames.shape[0] * reps / fps:.0f}s -> {loop_path})"
    )

grid = _build_grid(scene_frames, cols, decimate)
grid_reps = _reps_for_duration(grid.shape[0], fps, loop_seconds)
_write_mp4(snakemake.output.montage, grid, fps)
_write_mp4(snakemake.output.montage_loop, grid, fps, reps=grid_reps)
print(
    f"Saved montage {snakemake.output.montage}: "
    f"{grid.shape[0]} frames, {grid.shape[2]}x{grid.shape[1]}"
)
print(
    f"Saved looped montage {snakemake.output.montage_loop}: "
    f"{grid_reps}x ~{grid.shape[0] * grid_reps / fps:.0f}s"
)
