"""Behavioral-dissociation slide figure using animation scenes.

Input:
  data/slide_anim_frames.npz   -- rendered by render_slide_anim_frames.py (Mac host)
  data/scenes.npz              -- for pixel model training data

Layout per row (one scene):
  [t=0]  [t=160]  [t=320]  →  [Pixels]  [Physics]  [Ground truth]
  <-----  observed  ----->     <-----  predict t=480  ----------->

Run:
    uv run python scripts/plot_slide_anim_dissociation.py
"""

import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from PIL import Image as _Image
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.neural_network import MLPRegressor

KEEP_SCENES = [0, 1, 2, 3]   # ← indices into SCENE_INDICES (0-3); edit after contact sheet

SCENE_INDICES = [1248, 1367, 1794, 1886]
IMAGE_SIZE = 64
HIRES = 256
BEH_PCA_DIM = 200   # fixed integer → randomized SVD (fast); 200 comps ≫ 90% var

OUT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                       "figures")
DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                        "data")

# ── load rendered frames ───────────────────────────────────────────────────────
print("Loading rendered frames…")
af = np.load(os.path.join(DATA_DIR, "slide_anim_frames.npz"))
input_frames    = af["input_frames"]     # (4, 3, 256, 256, 3) uint8  RGB
oracle_target   = af["oracle_target"]    # (4, 256, 256, 3) uint8
inferred_target = af["inferred_target"]  # (4, 256, 256, 3) uint8

# ── train pixel model on main 2000-scene data ─────────────────────────────────
print("Training pixel model on main data…")
raw = np.load(os.path.join(DATA_DIR, "scenes.npz"), allow_pickle=True)

import json
meta = json.loads(raw["metadata_json"].item())
tpi_start, tpi_stop = meta["target_pixel_indices"]   # [0, 16384]
rgba_bytes = tpi_stop - tpi_start

# 3-frame RGBA pixel input (concatenated initial + early + late)
pixel_3f_all = np.concatenate(
    [raw["initial_renders"], raw["early_renders"], raw["late_renders"]], axis=1
)   # (2000, 3*49152)

# Only keep the RGBA bytes of each frame (first 16384 per 49152-dim frame block)
D_frame = 49152
rgba_3f = np.concatenate(
    [
        raw["initial_renders"][:, tpi_start:tpi_stop],
        raw["early_renders"][:, tpi_start:tpi_stop],
        raw["late_renders"][:, tpi_start:tpi_stop],
    ],
    axis=1,
)   # (2000, 3*16384)

target_rgba = raw["target_renders"][:, tpi_start:tpi_stop]   # (2000, 16384)

# Fit input scaler + PCA
scaler_in = StandardScaler()
rgba_3f_scaled = scaler_in.fit_transform(rgba_3f)
pca_in = PCA(n_components=BEH_PCA_DIM, whiten=True, random_state=42,
             svd_solver="randomized")
input_pca = pca_in.fit_transform(rgba_3f_scaled)

# Fit target scaler + PCA
scaler_tgt = StandardScaler()
target_scaled = scaler_tgt.fit_transform(target_rgba)
pca_tgt = PCA(n_components=BEH_PCA_DIM, whiten=True, random_state=42,
              svd_solver="randomized")
target_pca = pca_tgt.fit_transform(target_scaled)

print(f"  Input PCA components: {pca_in.n_components_}, "
      f"Target PCA components: {pca_tgt.n_components_}")

# Train MLP
pixel_model = MLPRegressor(
    hidden_layer_sizes=(256, 256),
    max_iter=300,
    random_state=42,
    early_stopping=True,
    validation_fraction=0.1,
    n_iter_no_change=20,
)
pixel_model.fit(input_pca, target_pca)
print(f"  Pixel MLP trained ({pixel_model.n_iter_} iters)")

# ── apply pixel model to animation scenes ─────────────────────────────────────
print("Applying pixel model to animation scenes…")

def upscale_rgba(arr_64):
    """uint8 (64,64,4) → uint8 (256,256,4) bilinear."""
    return np.array(_Image.fromarray(arr_64).resize((HIRES, HIRES), _Image.BILINEAR))

pixel_preds = []
for i, scene_idx in enumerate(SCENE_INDICES):
    # Use the stored 64×64 renders (initial, early, late) from training data
    init_r  = raw["initial_renders"][scene_idx, tpi_start:tpi_stop]
    early_r = raw["early_renders"][scene_idx, tpi_start:tpi_stop]
    late_r  = raw["late_renders"][scene_idx, tpi_start:tpi_stop]
    x = np.concatenate([init_r, early_r, late_r])[None]   # (1, 3*16384)

    x_scaled = scaler_in.transform(x)
    x_pca    = pca_in.transform(x_scaled)
    pred_pca = pixel_model.predict(x_pca)

    # Inverse-transform to RGBA
    pred_scaled  = pca_tgt.inverse_transform(pred_pca)
    pred_pixels  = scaler_tgt.inverse_transform(pred_scaled)
    pred_rgba_64 = (
        np.clip(pred_pixels, 0, 255)
        .astype(np.uint8)
        .reshape(IMAGE_SIZE, IMAGE_SIZE, 4)
    )
    pixel_preds.append(upscale_rgba(pred_rgba_64))

pixel_preds = np.stack(pixel_preds)   # (4, 256, 256, 4) uint8 RGBA

# ── styling ───────────────────────────────────────────────────────────────────
INPUT_COLOR   = "#777777"
PIXEL_COLOR   = "#c0392b"
PHYSICS_COLOR = "#1a7abf"
GT_COLOR      = "#444444"


def _add_border(ax, color, lw=3.5):
    for spine in ax.spines.values():
        spine.set_edgecolor(color)
        spine.set_linewidth(lw)
        spine.set_visible(True)


def _make_figure(row_indices, contact=False):
    n = len(row_indices)
    col_ratios = [1, 1, 1, 0.22, 1, 1, 1]
    total_w = sum(col_ratios)

    fig_w = 13.5
    row_h = 2.4 if contact else 2.6
    header_h = 0.55
    fig_h = header_h + n * row_h + (n - 1) * 0.08

    fig = plt.figure(figsize=(fig_w, fig_h), facecolor="white")

    left_m  = 0.065
    right_m = 0.015
    usable_w = 1.0 - left_m - right_m
    unit = usable_w / total_w

    col_offsets = [0, 1, 2, 3, 3.22, 4.22, 5.22]
    col_widths  = [1, 1, 1, 0.22, 1, 1, 1]
    col_x_centres = [left_m + (o + w / 2) * unit
                     for o, w in zip(col_offsets, col_widths)]

    top_pad  = header_h / fig_h
    bot_pad  = 0.06 / fig_h
    row_gap  = 0.08 / fig_h
    row_frac = (1.0 - top_pad - bot_pad - row_gap * (n - 1)) / n

    outer_gs = gridspec.GridSpec(
        n, 1,
        figure=fig,
        top=1.0 - top_pad,
        bottom=bot_pad,
        hspace=row_gap / row_frac,
    )

    for row_pos, row_idx in enumerate(row_indices):
        gs = gridspec.GridSpecFromSubplotSpec(
            1, 7,
            subplot_spec=outer_gs[row_pos],
            width_ratios=col_ratios,
            wspace=0.04,
        )

        rgb_pixel   = pixel_preds[row_idx, :, :, :3]
        rgb_physics = inferred_target[row_idx]
        rgb_gt      = oracle_target[row_idx]
        rgb_in      = [input_frames[row_idx, t] for t in range(3)]  # t=0,160,320 RGB

        imgs         = rgb_in + [None, rgb_pixel, rgb_physics, rgb_gt]
        border_colors = [INPUT_COLOR]*3 + [None, PIXEL_COLOR, PHYSICS_COLOR, GT_COLOR]

        for col_idx, (img, bc) in enumerate(zip(imgs, border_colors)):
            ax = fig.add_subplot(gs[col_idx])
            if img is None:
                ax.set_xlim(0, 1); ax.set_ylim(0, 1)
                ax.annotate(
                    "", xy=(0.88, 0.5), xytext=(0.12, 0.5),
                    xycoords="axes fraction", textcoords="axes fraction",
                    arrowprops=dict(
                        arrowstyle="-|>", color="#333",
                        lw=2.2, mutation_scale=18,
                    ),
                )
                ax.axis("off")
                continue

            ax.imshow(img)
            ax.set_xticks([]); ax.set_yticks([])
            _add_border(ax, bc)

            if col_idx == 0 and contact:
                ax.set_ylabel(
                    f"scene\n{SCENE_INDICES[row_idx]}", fontsize=7.5,
                    rotation=0, labelpad=42, va="center",
                    color="#555", fontweight="bold",
                )

    # ── column sub-labels ────────────────────────────────────────────────────
    sublabel_y = 1.0 - top_pad * 0.45
    header_y   = 1.0 - 0.005

    sub_labels  = ["Frame 1", "Frame 2", "Frame 3", "", "Pixels", "Physics", "Ground truth"]
    sub_colors  = [INPUT_COLOR]*3 + ["", PIXEL_COLOR, PHYSICS_COLOR, GT_COLOR]
    sub_weights = ["normal"]*3 + ["", "bold", "bold", "normal"]
    for x, label, color, weight in zip(col_x_centres, sub_labels, sub_colors, sub_weights):
        if not label:
            continue
        fig.text(x, sublabel_y, label, ha="center", va="top",
                 fontsize=9.5 if not contact else 8.5,
                 color=color, fontweight=weight)

    # ── group headers ─────────────────────────────────────────────────────────
    for (xi, xf, label, color) in [
        (col_x_centres[0], col_x_centres[2], "Observed", INPUT_COLOR),
        (col_x_centres[4], col_x_centres[6], "Predict frame 4", "#222"),
    ]:
        xm = (xi + xf) / 2
        fig.text(xm, header_y, label, ha="center", va="top",
                 fontsize=10.5 if not contact else 9.5,
                 color=color, fontweight="bold")
        fig.add_artist(plt.Line2D(
            [xi - 0.01, xf + 0.01], [sublabel_y + 0.012, sublabel_y + 0.012],
            transform=fig.transFigure, color=color, lw=1.2,
        ))

    return fig


# ── contact sheet: all 4 scenes ───────────────────────────────────────────────
print("Generating contact sheet…")
fig_c = _make_figure(list(range(4)), contact=True)
path_c = os.path.join(OUT_DIR, "slide_anim_contact.png")
fig_c.savefig(path_c, dpi=130, bbox_inches="tight", facecolor="white")
plt.close(fig_c)
print(f"  → {path_c}")

# ── final figure ──────────────────────────────────────────────────────────────
print(f"Generating final figure (rows {KEEP_SCENES})…")
fig_f = _make_figure(KEEP_SCENES, contact=False)
path_f = os.path.join(OUT_DIR, "slide_anim_dissociation.png")
fig_f.savefig(path_f, dpi=150, bbox_inches="tight", facecolor="white")
plt.close(fig_f)
print(f"  → {path_f}")
