import sys, os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
from analyses.plot_figures import plot_pp, plot_pp_frames
from io_utils import load_scenes
from config import IMAGE_SIZE

fig_dir = os.path.dirname(snakemake.output.figure)
os.makedirs(fig_dir, exist_ok=True)

plot_data = dict(np.load(snakemake.input.plot_data, allow_pickle=False))
scenes = load_scenes(snakemake.input.scenes)
fwd = np.load(snakemake.input.forward_renders)
fwd_states = fwd["forward_program_states"]

# Extract the initial-frame RGBA from forward program states for visualization.
# frame_idx maps pp frame positions → scene indices (oracle_test_idx[:n_frame_samples]).
frame_idx = plot_data["frame_idx"].astype(int)
n_pp = len(frame_idx)
metadata = scenes["metadata"]
fri = metadata["frame_render_indices"]
rgba_n = metadata["target_pixel_indices"].stop - metadata["target_pixel_indices"].start
fwd_init_rgba = fwd_states[
    frame_idx, fri["initial"].start : fri["initial"].start + rgba_n
]
fwd_frame_imgs = (
    np.clip(fwd_init_rgba, 0, 255)
    .astype(np.uint8)
    .reshape(n_pp, IMAGE_SIZE, IMAGE_SIZE, 4)
)

plot_pp(plot_data, fig_dir=fig_dir)
plot_pp_frames(plot_data, fig_dir=fig_dir, fwd_frame_imgs=fwd_frame_imgs)
