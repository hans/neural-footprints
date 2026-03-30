"""
PCA Negative Analysis: variance-maximizing components miss physics.

Shows that PCA on neural activity converges on render-dominated dimensions.
Motion direction (a causally operative physics variable) is invisible in the
top PCs and barely decodable even from all PCs.
"""

import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegressionCV
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from analyses.plot_style import COLORS, paper_style


def run_pca_analysis(neural_activity, scenes, neural_meta, fig_dir="figures"):
    print("\n" + "=" * 60)
    print("PCA NEGATIVE ANALYSIS: Variance ≠ Information")
    print("=" * 60)

    n_scenes, n_neurons = neural_activity.shape

    # Extract motion direction: vx at t=0 is index 7 in physics labels
    vx = scenes['initial_physics_labels'][:, 7]
    motion_dir = (vx > 0).astype(int)  # 1 = rightward, 0 = leftward
    n_right = motion_dir.sum()
    print(f"  Motion direction: {n_scenes - n_right} left, {n_right} right")

    # Full PCA on standardized neural activity
    neural_scaled = StandardScaler().fit_transform(neural_activity)
    pca = PCA(n_components=n_neurons, random_state=42)
    neural_pca = pca.fit_transform(neural_scaled)
    cumvar = np.cumsum(pca.explained_variance_ratio_)

    # Decoding accuracy as function of number of PCs
    pc_counts = [k for k in [1, 2, 5, 10, 25, 50, 100, 200, n_neurons] if k <= n_neurons]
    decode_accs = []
    for k in pc_counts:
        scores = cross_val_score(
            LogisticRegressionCV(cv=5, max_iter=1000, random_state=42),
            neural_pca[:, :k], motion_dir, cv=5, scoring='accuracy',
        )
        acc = scores.mean()
        decode_accs.append(acc)
        print(f"    Top {k:>3d} PCs: accuracy = {acc:.2%}")

    all_pc_acc = decode_accs[-1]

    # --- Figure ---
    with paper_style():
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

        # Plot 1: Elbow + decoding reference
        ax1.plot(range(1, n_neurons + 1), cumvar, color=COLORS['pixels'], linewidth=2)
        ax1.axhline(all_pc_acc, color=COLORS['physics'], linestyle='--', linewidth=1.5,
                    label=f'Motion decoding (all PCs): {all_pc_acc:.1%}')
        ax1.axhline(0.5, color='gray', linestyle=':', alpha=0.5, label='Chance (50%)')
        ax1.set_xlabel('Number of principal components')
        ax1.set_ylabel('Cumulative explained variance')
        ax1.set_title('PCA Elbow Plot + Motion Decoding')
        ax1.legend(fontsize=9)
        ax1.set_xlim(1, n_neurons)

        # Plot 2: PC1/PC2 scatter colored by motion direction
        pc1, pc2 = neural_pca[:, 0], neural_pca[:, 1]
        # Clip to percentile range to remove outlier distortion
        lo, hi = 1, 99
        pc1_lim = np.percentile(pc1, [lo, hi])
        pc2_lim = np.percentile(pc2, [lo, hi])
        colors = np.where(motion_dir == 1, COLORS['physics'], COLORS['pixels'])
        ax2.scatter(pc1, pc2, c=colors, alpha=0.3, s=10, edgecolors='none')
        pad1 = 0.05 * (pc1_lim[1] - pc1_lim[0])
        pad2 = 0.05 * (pc2_lim[1] - pc2_lim[0])
        ax2.set_xlim(pc1_lim[0] - pad1, pc1_lim[1] + pad1)
        ax2.set_ylim(pc2_lim[0] - pad2, pc2_lim[1] + pad2)
        ax2.set_xlabel('PC1')
        ax2.set_ylabel('PC2')
        ax2.set_title('PC1 vs PC2 (colored by motion direction)')
        ax2.legend(handles=[
            Line2D([0], [0], marker='o', color='w', markerfacecolor=COLORS['pixels'],
                   markersize=8, label='Left'),
            Line2D([0], [0], marker='o', color='w', markerfacecolor=COLORS['physics'],
                   markersize=8, label='Right'),
        ], fontsize=9)

        # Inset: decoding accuracy vs number of PCs
        ax_inset = ax2.inset_axes([0.55, 0.05, 0.42, 0.40])
        ax_inset.plot(pc_counts, decode_accs, marker='o', color=COLORS['pixels'],
                      markersize=4, linewidth=1.5)
        ax_inset.axhline(0.5, color='gray', linestyle='--', alpha=0.5)
        ax_inset.set_xlabel('# PCs', fontsize=7)
        ax_inset.set_ylabel('Accuracy', fontsize=7)
        ax_inset.set_title('Motion decoding', fontsize=8)
        ax_inset.tick_params(labelsize=6)
        ax_inset.set_xscale('log')
        ax_inset.set_ylim(0.4, 1.0)

        plt.tight_layout()
        fig_path = f"{fig_dir}/pca_analysis.png"
        plt.savefig(fig_path)
        plt.close()
        print(f"\nFigure saved: {fig_path}")

    # --- Figure v2: elbow + decoding overlay with twinx ---
    with paper_style():
        fig, ax_var = plt.subplots(figsize=(6.5, 5))
        ax_dec = ax_var.twinx()

        # Cumulative variance (left y-axis)
        ax_var.plot(range(1, n_neurons + 1), cumvar,
                    color=COLORS['pixels'], linewidth=2, label='Cumulative variance')
        ax_var.set_xlabel('Number of principal components')
        ax_var.set_ylabel('Cumulative explained variance', color=COLORS['pixels'])
        ax_var.tick_params(axis='y', labelcolor=COLORS['pixels'])
        ax_var.set_xlim(1, n_neurons)
        ax_var.set_ylim(0, 1.05)

        # Decoding accuracy (right y-axis)
        ax_dec.plot(pc_counts, decode_accs, marker='o', color=COLORS['physics'],
                    linewidth=2, markersize=5, label='Motion decoding accuracy')
        ax_dec.axhline(0.5, color='gray', linestyle=':', alpha=0.5, linewidth=1)
        ax_dec.set_ylabel('Decoding accuracy', color=COLORS['physics'])
        ax_dec.tick_params(axis='y', labelcolor=COLORS['physics'])
        ax_dec.set_ylim(0.4, 1.05)

        # Combined legend
        lines_var, labels_var = ax_var.get_legend_handles_labels()
        lines_dec, labels_dec = ax_dec.get_legend_handles_labels()
        ax_var.legend(lines_var + lines_dec, labels_var + labels_dec,
                      loc='center right', fontsize=9)

        ax_var.set_title('PCA Variance vs Motion Decoding')

        fig_path2 = f"{fig_dir}/pca_variance_decoding.png"
        plt.savefig(fig_path2)
        plt.close()
        print(f"Figure saved: {fig_path2}")

    return {
        'cumulative_variance': cumvar.tolist(),
        'all_pc_decoding_accuracy': float(all_pc_acc),
        'pc_counts': pc_counts,
        'decode_accuracies': [float(a) for a in decode_accs],
    }
