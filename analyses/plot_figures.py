"""Plotting functions for all analyses. Separated from computation for fast iteration."""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from scipy.spatial.distance import squareform

from analyses.plot_style import COLORS, paper_style


def plot_encoding(plot_data, fig_dir="figures"):
    r2_pixels_only = plot_data['r2_pixels_only']
    r2_combined = plot_data['r2_combined']
    n_neurons = len(r2_pixels_only)
    mean_r2_pix = r2_pixels_only.mean()
    mean_r2_comb = r2_combined.mean()
    mean_delta = (r2_combined - r2_pixels_only).mean()
    neuron_counts = plot_data['neuron_counts']
    subsample_means = plot_data['subsample_means']
    subsample_sems = plot_data['subsample_sems']
    control_acc = float(plot_data['control_accuracy'])
    control_acc_std = float(plot_data['control_accuracy_std'])

    with paper_style():
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        ax = axes[0]
        ax.bar(['Pixels only', 'Pixels + Physics'],
               [mean_r2_pix, mean_r2_comb],
               yerr=[r2_pixels_only.std() / np.sqrt(n_neurons),
                     r2_combined.std() / np.sqrt(n_neurons)],
               color=[COLORS['pixels'], COLORS['physics']], capsize=5)
        ax.set_ylabel('Mean R²')
        ax.set_title('Encoding Model: R² ± Physics Labels')
        ymax = max(mean_r2_pix, mean_r2_comb) * 1.1
        ax.annotate(f'ΔR² = {mean_delta:.6f}', xy=(0.5, ymax),
                    ha='center', fontsize=10, style='italic')

        ax = axes[1]
        ax.errorbar(neuron_counts, subsample_means, yerr=subsample_sems,
                    marker='o', color=COLORS['pixels'], capsize=3)
        ax.axhline(0, color='gray', linestyle='--', alpha=0.5)
        ax.set_xlabel('Number of neurons sampled')
        ax.set_ylabel('Mean ΔR²')
        ax.set_title('ΔR² vs. Neuron Subsampling')

        ax = axes[2]
        ax.bar(['Physics → Behavior'], [control_acc],
               yerr=[control_acc_std], color=COLORS['control'], capsize=5)
        ax.axhline(0.5, color='gray', linestyle='--', alpha=0.5, label='Chance')
        ax.set_ylabel('Accuracy')
        ax.set_title('Control: Physics Labels Predict Behavior')
        ax.set_ylim(0, 1)
        ax.legend()

        plt.tight_layout()
        fig_path = f"{fig_dir}/encoding_analysis.png"
        plt.savefig(fig_path)
        plt.close()


def plot_rsa(plot_data, fig_dir="figures"):
    rdm_neural = plot_data['rdm_neural']
    rdm_render = plot_data['rdm_render']
    rdm_physics = plot_data['rdm_physics']
    n_sub = int(plot_data['n_sub'])
    corr_neural_render = float(plot_data['corr_neural_render'])
    corr_neural_physics = float(plot_data['corr_neural_physics'])
    partial_corr = float(plot_data['partial_corr'])

    with paper_style():
        fig, axes = plt.subplots(1, 4, figsize=(20, 5))

        n_show = min(100, n_sub)
        rdm_neural_sq = squareform(rdm_neural)[:n_show, :n_show]
        rdm_render_sq = squareform(rdm_render)[:n_show, :n_show]
        rdm_physics_sq = squareform(rdm_physics)[:n_show, :n_show]

        for ax, rdm, title in zip(axes[:3],
                                   [rdm_neural_sq, rdm_render_sq, rdm_physics_sq],
                                   ['Neural RDM', 'Render RDM', 'Physics RDM']):
            im = ax.imshow(rdm, cmap='viridis', aspect='equal')
            ax.set_title(title)
            ax.set_xlabel('Scene')
            ax.set_ylabel('Scene')
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

        ax = axes[3]
        labels = ['Neural↔Render', 'Neural↔Physics', 'Partial\nNeural↔Physics|Render']
        values = [corr_neural_render, corr_neural_physics, partial_corr]
        colors = [COLORS['pixels'], COLORS['physics'], COLORS['neutral']]
        bars = ax.bar(labels, values, color=colors)
        ax.set_ylabel('Spearman r')
        ax.set_title('RSA Correlations')
        ax.axhline(0, color='gray', linestyle='-', alpha=0.3)
        for bar, val in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                    f'{val:.3f}', ha='center', va='bottom', fontsize=9)

        plt.tight_layout()
        fig_path = f"{fig_dir}/rsa_analysis.png"
        plt.savefig(fig_path)
        plt.close()


def plot_dissociation(plot_data, fig_dir="figures"):
    r2_render = plot_data['r2_render']
    r2_physics = plot_data['r2_physics']
    n_neurons = len(r2_render)
    mean_r2_render = r2_render.mean()
    mean_r2_physics = r2_physics.mean()
    render_score = float(plot_data['render_score'])
    physics_score = float(plot_data['physics_score'])
    metric_label = str(plot_data['metric_label'])
    chance_val = float(plot_data['chance'])
    chance = None if np.isnan(chance_val) else chance_val

    with paper_style():
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))

        bar_width = 0.5
        colors = [COLORS['pixels'], COLORS['physics']]
        labels = ['Render\nmodel', 'Physics\nmodel']

        bars1 = ax1.bar(labels, [mean_r2_render, mean_r2_physics],
                        width=bar_width, color=colors,
                        yerr=[r2_render.std() / np.sqrt(n_neurons),
                              r2_physics.std() / np.sqrt(n_neurons)],
                        capsize=5)
        ax1.set_ylabel('Neural variance explained (R²)')
        ax1.set_title('Encoding model performance')
        for bar, val in zip(bars1, [mean_r2_render, mean_r2_physics]):
            ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                     f'{val:.3f}', ha='center', va='bottom', fontweight='bold')

        bars2 = ax2.bar(labels, [render_score, physics_score],
                        width=bar_width, color=colors, capsize=5)
        if chance is not None:
            ax2.axhline(chance, color='gray', linestyle='--', alpha=0.5, label='Chance')
            ax2.set_ylim(0, 1.1)
            ax2.legend()
        ax2.set_ylabel(metric_label)
        ax2.set_title('Behavioral sufficiency')
        for bar, val in zip(bars2, [render_score, physics_score]):
            ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                     f'{val:.3f}', ha='center', va='bottom', fontweight='bold')

        plt.tight_layout()
        fig_path = f"{fig_dir}/dissociation.png"
        plt.savefig(fig_path)
        plt.close()


def plot_predicted_frames(plot_data, fig_dir="figures"):
    init_imgs = plot_data['predicted_init_imgs']
    render_imgs = plot_data['predicted_render_imgs']
    physics_imgs = plot_data['predicted_physics_imgs']
    final_imgs = plot_data['predicted_final_imgs']
    n = len(init_imgs)

    col_titles = ['t=0 (input)', 'Render model\nprediction',
                  'Physics model\nprediction', 't=N (actual)']
    cols = [init_imgs, render_imgs, physics_imgs, final_imgs]

    with paper_style():
        fig, axes = plt.subplots(n, 4, figsize=(8, 2 * n))
        if n == 1:
            axes = axes[np.newaxis, :]

        for col_idx, (title, imgs) in enumerate(zip(col_titles, cols)):
            axes[0, col_idx].set_title(title, fontsize=8)
            for row_idx in range(n):
                axes[row_idx, col_idx].imshow(imgs[row_idx])
                axes[row_idx, col_idx].axis('off')

        plt.tight_layout(pad=0.3)
        fig_path = f"{fig_dir}/predicted_frames.png"
        plt.savefig(fig_path)
        plt.close()


def plot_dynamics(plot_data, fig_dir="figures"):
    r2_physics_forward = plot_data['r2_physics_forward']
    r2_pixel_forward = plot_data['r2_pixel_forward']
    n_neurons = len(r2_physics_forward)
    mean_r2_physics = r2_physics_forward.mean()
    mean_r2_pixel = r2_pixel_forward.mean()

    with paper_style():
        fig, ax1 = plt.subplots(1, 1, figsize=(5, 5))

        bar_width = 0.5
        colors = [COLORS['pixels'], COLORS['physics']]
        labels = ['Render\nmodel', 'Physics\nmodel']

        bars1 = ax1.bar(labels, [mean_r2_pixel, mean_r2_physics],
                        width=bar_width, color=colors,
                        yerr=[r2_pixel_forward.std() / np.sqrt(n_neurons),
                              r2_physics_forward.std() / np.sqrt(n_neurons)],
                        capsize=5)
        ax1.set_ylabel('Future neural R²')
        ax1.set_title('Future brain state prediction')
        for bar, val in zip(bars1, [mean_r2_pixel, mean_r2_physics]):
            ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                     f'{val:.3f}', ha='center', va='bottom', fontweight='bold')

        plt.tight_layout()
        fig_path = f"{fig_dir}/dynamics_analysis.png"
        plt.savefig(fig_path)
        plt.close()


def plot_pca(plot_data, fig_dir="figures"):
    cumvar = plot_data['cumvar']
    neural_pca_2d = plot_data['neural_pca_2d']
    motion_dir = plot_data['motion_dir']
    pc_counts = plot_data['pc_counts'].astype(int)
    decode_accs = plot_data['decode_accs']
    n_neurons = int(plot_data['n_neurons'])
    all_pc_acc = decode_accs[-1]
    pc1, pc2 = neural_pca_2d[:, 0], neural_pca_2d[:, 1]

    with paper_style():
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

        ax1.plot(range(1, n_neurons + 1), cumvar, color=COLORS['pixels'], linewidth=2)
        ax1.axhline(all_pc_acc, color=COLORS['physics'], linestyle='--', linewidth=1.5,
                    label=f'Motion decoding (all PCs): {all_pc_acc:.1%}')
        ax1.axhline(0.5, color='gray', linestyle=':', alpha=0.5, label='Chance (50%)')
        ax1.set_xlabel('Number of principal components')
        ax1.set_ylabel('Cumulative explained variance')
        ax1.set_title('PCA Elbow Plot + Motion Decoding')
        ax1.legend(fontsize=9)
        ax1.set_xlim(1, n_neurons)

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

    # Figure 2: elbow + decoding overlay with twinx
    with paper_style():
        fig, ax_var = plt.subplots(figsize=(6.5, 5))
        ax_dec = ax_var.twinx()

        ax_var.plot(range(1, n_neurons + 1), cumvar,
                    color=COLORS['pixels'], linewidth=2, label='Cumulative variance')
        ax_var.set_xlabel('Number of principal components')
        ax_var.set_ylabel('Cumulative explained variance', color=COLORS['pixels'])
        ax_var.tick_params(axis='y', labelcolor=COLORS['pixels'])
        ax_var.set_xlim(1, n_neurons)
        ax_var.set_ylim(0, 1.05)

        ax_dec.plot(pc_counts, decode_accs, marker='o', color=COLORS['physics'],
                    linewidth=2, markersize=5, label='Motion decoding accuracy')
        ax_dec.axhline(0.5, color='gray', linestyle=':', alpha=0.5, linewidth=1)
        ax_dec.set_ylabel('Decoding accuracy', color=COLORS['physics'])
        ax_dec.tick_params(axis='y', labelcolor=COLORS['physics'])
        ax_dec.set_ylim(0.4, 1.05)

        lines_var, labels_var = ax_var.get_legend_handles_labels()
        lines_dec, labels_dec = ax_dec.get_legend_handles_labels()
        ax_var.legend(lines_var + lines_dec, labels_var + labels_dec,
                      loc='center right', fontsize=9)

        ax_var.set_title('PCA Variance vs Motion Decoding')

        fig_path2 = f"{fig_dir}/pca_variance_decoding.png"
        plt.savefig(fig_path2)
        plt.close()


def plot_sample_scenes(initial_renders, program_states, pixel_indices,
                       image_size, n_timesteps, fig_dir="figures", n_samples=16):
    n = min(n_samples, len(initial_renders))
    with paper_style():
        fig, axes = plt.subplots(n, 2, figsize=(4, 2 * n))
        if n == 1:
            axes = axes[np.newaxis, :]

        axes[0, 0].set_title('t = 0 (initial)', fontsize=9)
        axes[0, 1].set_title(f't = {n_timesteps} (final)', fontsize=9)

        for i in range(n):
            init_rgba = initial_renders[i].astype(np.uint8).reshape(
                image_size, image_size, 4)
            final_rgba = program_states[i, pixel_indices].astype(np.uint8).reshape(
                image_size, image_size, 4)
            axes[i, 0].imshow(init_rgba)
            axes[i, 0].axis('off')
            axes[i, 1].imshow(final_rgba)
            axes[i, 1].axis('off')

        plt.tight_layout(pad=0.3)
        fig_path = f"{fig_dir}/sample_scenes.png"
        plt.savefig(fig_path)
        plt.close()
