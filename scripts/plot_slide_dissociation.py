"""Generate behavioral-dissociation slide figure.

Two outputs:
  figures/slide_dissociation_contact.png  -- all 8 scenes, pick your rows
  figures/slide_dissociation.png          -- final figure (set KEEP_SCENES below)

Layout per row (one scene):
  [Frame 1] [Frame 2] [Frame 3]  ||  [Pixels] [Physics] [Ground truth]
  <--  observed input frames  -->  <--  predicted frame 4  -->

Run:
    uv run python scripts/plot_slide_dissociation.py
"""

import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrow
import matplotlib.gridspec as gridspec

# ── scenes to include in the final figure (0-indexed, 0–7 available) ──────────
KEEP_SCENES = [0, 1, 2, 3, 4, 5, 6, 7]   # ← edit after reviewing contact sheet

OUT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                       "figures")
os.makedirs(OUT_DIR, exist_ok=True)

# ── load data ─────────────────────────────────────────────────────────────────
sf  = np.load(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                            "data", "slide_frames.npz"))
diss = np.load(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                             "data", "dissociation_plot_data.npz"),
               allow_pickle=True)

init_imgs     = sf["init_imgs"]      # (8,256,256,4) true t=0
early_imgs    = sf["early_imgs"]     # (8,256,256,4) true t=early
late_imgs     = sf["late_imgs"]      # (8,256,256,4) true t=late
oracle_imgs   = sf["oracle_imgs"]    # (8,256,256,4) oracle target
inferred_imgs = sf["inferred_imgs"]  # (8,256,256,4) inferred-physics target
pixel_imgs    = diss["predicted_pixel_imgs"]  # (8,256,256,4) pixel model pred

# ── helpers ───────────────────────────────────────────────────────────────────
def rgba_to_rgb(arr):
    return arr[:, :, :3]

COL_LABELS = ["Frame 1", "Frame 2", "Frame 3", "Pixels", "Physics", "Ground truth"]
# visual groups: inputs (0–2) and predictions (3–5)
GROUP_SPANS = [(0, 3, "Observed"), (3, 6, "Predict frame 4")]
GROUP_COLORS = ["#444", "#444"]

INPUT_BORDER  = "#888888"
PIXEL_COLOR   = "#c0392b"   # red  — blurry / limited
PHYSICS_COLOR = "#1a7abf"   # blue — sharp physics
GT_COLOR      = "#444444"   # neutral

COL_BORDER_COLORS = [INPUT_BORDER] * 3 + [PIXEL_COLOR, PHYSICS_COLOR, GT_COLOR]


def _add_border(ax, color, lw=4):
    for spine in ax.spines.values():
        spine.set_edgecolor(color)
        spine.set_linewidth(lw)
        spine.set_visible(True)


def _make_figure(scene_indices, figsize_per_row=(13.5, 2.4), contact=False):
    n = len(scene_indices)
    n_cols = 7   # 3 input + 1 arrow + 3 output

    fig_w, fig_h_row = figsize_per_row
    fig_h = fig_h_row * n + (0.55 if not contact else 0.45)  # room for col headers

    fig = plt.figure(figsize=(fig_w, fig_h), facecolor="white")

    # Column width ratios: equal image cols with a narrow arrow col in middle
    col_ratios = [1, 1, 1, 0.22, 1, 1, 1]
    top_pad   = 0.52 / fig_h   # fractional space at top for headers
    bot_pad   = 0.08 / fig_h
    row_gap   = 0.06 / fig_h

    row_h = (1.0 - top_pad - bot_pad - row_gap * (n - 1)) / n

    for row_idx, scene_idx in enumerate(scene_indices):
        bottom = 1.0 - top_pad - (row_idx + 1) * row_h - row_idx * row_gap
        gs = gridspec.GridSpecFromSubplotSpec(
            1, n_cols,
            subplot_spec=gridspec.GridSpec(
                n, 1,
                figure=fig,
                top=1.0 - top_pad,
                bottom=bot_pad,
                hspace=row_gap / row_h,
            )[row_idx],
            width_ratios=col_ratios,
            wspace=0.04,
        )

        imgs = [
            rgba_to_rgb(init_imgs[scene_idx]),
            rgba_to_rgb(early_imgs[scene_idx]),
            rgba_to_rgb(late_imgs[scene_idx]),
            None,  # arrow col
            rgba_to_rgb(pixel_imgs[scene_idx]),
            rgba_to_rgb(inferred_imgs[scene_idx]),
            rgba_to_rgb(oracle_imgs[scene_idx]),
        ]
        border_colors = [INPUT_BORDER, INPUT_BORDER, INPUT_BORDER,
                         None,
                         PIXEL_COLOR, PHYSICS_COLOR, GT_COLOR]

        for col_idx, (img, bc) in enumerate(zip(imgs, border_colors)):
            ax = fig.add_subplot(gs[col_idx])
            if img is None:
                # Arrow cell
                ax.set_xlim(0, 1); ax.set_ylim(0, 1)
                ax.annotate(
                    "", xy=(0.85, 0.5), xytext=(0.15, 0.5),
                    xycoords="axes fraction", textcoords="axes fraction",
                    arrowprops=dict(
                        arrowstyle="-|>",
                        color="#333",
                        lw=2.2,
                        mutation_scale=18,
                    ),
                )
                ax.axis("off")
                continue

            ax.imshow(img)
            ax.set_xticks([]); ax.set_yticks([])
            _add_border(ax, bc, lw=3)

            # Row label (scene number) on leftmost image only
            if col_idx == 0 and contact:
                ax.set_ylabel(f"scene {scene_idx}", fontsize=7.5,
                              rotation=0, labelpad=38, va="center",
                              color="#555", fontweight="bold")

    # ── column headers (only once, at top) ────────────────────────────────────
    # Use figure-level text; position by computing fractional x of each col centre.
    # Approximate: 3 equal input cols + small arrow + 3 equal output cols.
    # total weight = 3 + 0.22 + 3 = 6.22; each image col = 1/6.22.
    total_w = 3 + 0.22 + 3
    left_margin  = 0.065
    right_margin = 0.02
    usable_w = 1.0 - left_margin - right_margin
    unit = usable_w / total_w

    # sub-label x centres (in figure coords)
    col_x = []
    offsets = [0, 1, 2, 3, 3.22, 4.22, 5.22]
    widths  = [1, 1, 1, 0.22, 1,    1,    1  ]
    for o, w in zip(offsets, widths):
        col_x.append(left_margin + (o + w / 2) * unit)

    sub_labels   = ["Frame 1", "Frame 2", "Frame 3", "", "Pixels", "Physics", "Ground truth"]
    sub_colors   = [INPUT_BORDER]*3 + ["none", PIXEL_COLOR, PHYSICS_COLOR, GT_COLOR]
    sub_weights  = ["normal"]*3 + ["normal", "bold", "bold", "normal"]

    header_y     = 1.0 - 0.005
    sublabel_y   = 1.0 - top_pad * 0.48

    for x, label, color, weight in zip(col_x, sub_labels, sub_colors, sub_weights):
        if not label:
            continue
        fig.text(x, sublabel_y, label, ha="center", va="top",
                 fontsize=9.5 if not contact else 8.5,
                 color=color, fontweight=weight)

    # Group headers
    group_defs = [
        (col_x[0], col_x[2], "Observed", INPUT_BORDER),
        (col_x[4], col_x[6], "Predict frame 4", "#222"),
    ]
    for x_start, x_end, label, color in group_defs:
        x_mid = (x_start + x_end) / 2
        fig.text(x_mid, header_y, label, ha="center", va="top",
                 fontsize=10.5 if not contact else 9.5,
                 color=color, fontweight="bold")
        # Underline bracket
        fig.add_artist(plt.Line2D(
            [x_start - 0.01, x_end + 0.01], [sublabel_y + 0.012, sublabel_y + 0.012],
            transform=fig.transFigure, color=color, lw=1.2,
        ))

    return fig


# ── contact sheet: all 8 scenes ───────────────────────────────────────────────
print("Generating contact sheet (all 8 scenes)…")
fig_contact = _make_figure(list(range(8)), figsize_per_row=(13.5, 2.3), contact=True)
contact_path = os.path.join(OUT_DIR, "slide_dissociation_contact.png")
fig_contact.savefig(contact_path, dpi=130, bbox_inches="tight", facecolor="white")
plt.close(fig_contact)
print(f"  → {contact_path}")

# ── final figure: only KEEP_SCENES ────────────────────────────────────────────
print(f"Generating final figure (scenes {KEEP_SCENES})…")
fig_final = _make_figure(KEEP_SCENES, figsize_per_row=(13.5, 2.55), contact=False)
final_path = os.path.join(OUT_DIR, "slide_dissociation.png")
fig_final.savefig(final_path, dpi=150, bbox_inches="tight", facecolor="white")
plt.close(fig_final)
print(f"  → {final_path}")
