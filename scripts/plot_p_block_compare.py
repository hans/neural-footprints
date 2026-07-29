"""
Snakemake script: cross-norm comparison figure for P block contribution.

Inputs:  data/zscore/p_block_plot_data.npz
         data/truncated_svd/p_block_plot_data.npz
Output:  figures/p_block_contribution_compare.pdf
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import json
import numpy as np
import matplotlib.pyplot as plt

from analyses.plot_style import paper_style, COL_WIDTH

os.makedirs("figures", exist_ok=True)

def _load(path):
    data = np.load(path, allow_pickle=False)
    return {
        "block_names": json.loads(str(data["block_names_json"])),
        "var_share": data["var_share"],
        "r2_signal": data["r2_P_from_block_signal"],
        "eff_contrib": data["effective_P_contribution"],
        "r2_total": float(data["r2_P_from_total_signal"]),
        "norm": str(data["norm"]),
    }


def _draw_panel(ax, d, shared_ymax):
    block_names = d["block_names"]
    var_share = d["var_share"]
    r2_signal = d["r2_signal"]
    eff_contrib = d["eff_contrib"]
    r2_total = d["r2_total"]
    norm = d["norm"]

    n_blocks = len(block_names)
    x = np.arange(n_blocks)
    width = 0.25

    ax.bar(x - width, var_share, width, label="var_share", color="#4878CF", alpha=0.85)
    ax.bar(x, r2_signal, width, label="r2_P_from_block_signal", color="#D65F5F", alpha=0.85)
    ax.bar(x + width, eff_contrib, width, label="eff_P_contrib", color="#6ACC65", alpha=0.85)

    ax.set_xticks(x)
    ax.set_xticklabels(block_names, rotation=20, ha="right")
    ax.set_ylim(0, shared_ymax)
    ax.set_title(norm, fontsize=7)

    ax.axhline(r2_total, color="black", linestyle="--", linewidth=0.7, alpha=0.6)
    ax.text(
        n_blocks - 0.5, r2_total * 1.05,
        f"r2_P_total={r2_total:.4f}",
        fontsize=5, color="black", ha="right", va="bottom"
    )


zscore_data = _load(snakemake.input.zscore)
tsvd_data = _load(snakemake.input.truncated_svd)

# Shared y-axis max across both panels
shared_ymax = max(
    max(zscore_data["var_share"].max(), zscore_data["r2_signal"].max(), zscore_data["eff_contrib"].max()),
    max(tsvd_data["var_share"].max(), tsvd_data["r2_signal"].max(), tsvd_data["eff_contrib"].max()),
) * 1.15

with paper_style():
    fig, axes = plt.subplots(1, 2, figsize=(COL_WIDTH * 2.0, 2.8), sharey=True)

    _draw_panel(axes[0], zscore_data, shared_ymax)
    _draw_panel(axes[1], tsvd_data, shared_ymax)

    axes[0].set_ylabel("Value")
    fig.suptitle("Per-block P contribution: zscore vs truncated_svd", fontsize=8)

    # Single legend below panels
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, fontsize=5,
               bbox_to_anchor=(0.5, -0.05))

    plt.tight_layout(rect=[0, 0.06, 1, 1])
    plt.savefig(snakemake.output.figure)
    plt.close()
