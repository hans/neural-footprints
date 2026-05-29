"""
Snakemake script: per-norm bar chart for P block contribution diagnostic.

Input:  data/{norm}/p_block_plot_data.npz
Output: figures/{norm}/p_block_contribution.pdf
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import json
import numpy as np
import matplotlib.pyplot as plt

from analyses.plot_style import paper_style, COL_WIDTH

fig_dir = os.path.dirname(snakemake.output.figure)
os.makedirs(fig_dir, exist_ok=True)

data = np.load(snakemake.input.plot_data, allow_pickle=False)
block_names = json.loads(str(data["block_names_json"]))
var_share = data["var_share"]
r2_signal = data["r2_P_from_block_signal"]
eff_contrib = data["effective_P_contribution"]
r2_total = float(data["r2_P_from_total_signal"])
norm = str(data["norm"])

n_blocks = len(block_names)
x = np.arange(n_blocks)
width = 0.25

with paper_style():
    fig, ax = plt.subplots(figsize=(COL_WIDTH * 1.4, 2.6))

    bars1 = ax.bar(x - width, var_share, width, label="var_share", color="#4878CF", alpha=0.85)
    bars2 = ax.bar(x, r2_signal, width, label="r2_P_from_block_signal", color="#D65F5F", alpha=0.85)
    bars3 = ax.bar(x + width, eff_contrib, width, label="effective_P_contribution", color="#6ACC65", alpha=0.85)

    ax.set_xticks(x)
    ax.set_xticklabels(block_names, rotation=20, ha="right")
    ax.set_ylabel("Value")
    ax.set_ylim(bottom=0)
    ax.set_title(f"P contribution by block — {norm} normalization")
    ax.legend(fontsize=5, loc="upper right")

    # Annotate r2_P_from_total_signal
    ax.axhline(r2_total, color="black", linestyle="--", linewidth=0.7, alpha=0.6)
    ax.text(
        n_blocks - 0.5, r2_total * 1.05,
        f"r2_P_total={r2_total:.4f}",
        fontsize=5, color="black", ha="right", va="bottom"
    )

    plt.tight_layout()
    plt.savefig(snakemake.output.figure)
    plt.close()
