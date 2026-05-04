"""Plotting functions for all analyses. Separated from computation for fast iteration."""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import ticker as mticker
from matplotlib.lines import Line2D
from scipy.cluster.hierarchy import leaves_list, linkage
from scipy.spatial.distance import squareform

from analyses.plot_style import COLORS, COL_WIDTH, FULL_WIDTH, paper_style


# Model names
LABEL_RENDER = "Sensory"
LABEL_PHYSICS = "Physics"
LABEL_RENDER_PLUS_PHYSICS = "Full"


def plot_encoding(plot_data, fig_dir="figures"):
    r2_pixel_only = plot_data['r2_pixel_only']
    r2_combined = plot_data['r2_combined']
    n_neurons = len(r2_pixel_only)
    mean_r2_pixel = r2_pixel_only.mean()
    mean_r2_comb = r2_combined.mean()
    mean_delta = (r2_combined - r2_pixel_only).mean()
    neuron_counts = plot_data['neuron_counts']
    subsample_means = plot_data['subsample_means']
    subsample_sems = plot_data['subsample_sems']
    control_acc = float(plot_data['control_accuracy'])
    control_acc_std = float(plot_data['control_accuracy_std'])

    with paper_style():
        fig, axes = plt.subplots(3, 1, figsize=(COL_WIDTH, 5.5))

        # Panel A: R² bar plot
        ax = axes[0]
        ax.bar([LABEL_RENDER, LABEL_RENDER_PLUS_PHYSICS],
               [mean_r2_pixel, mean_r2_comb],
               yerr=[r2_pixel_only.std() / np.sqrt(n_neurons),
                     r2_combined.std() / np.sqrt(n_neurons)],
               color=[COLORS['pixels'], COLORS['physics']], capsize=3,
               width=0.6)
        ax.set_ylabel('Mean R\u00b2')
        ax.set_title('Encoding model: R\u00b2 \u00b1 physics labels')
        ymax = max(mean_r2_pixel, mean_r2_comb) * 1.12
        ax.annotate(f'\u0394R\u00b2 = {mean_delta:.6f}', xy=(0.5, ymax),
                    ha='center', style='italic')

        # Panel B: Subsampling curve
        ax = axes[1]
        ax.errorbar(neuron_counts, subsample_means, yerr=subsample_sems,
                    marker='o', color=COLORS['pixels'], capsize=2)
        ax.axhline(0, color='gray', linestyle='--', alpha=0.5)
        ax.set_xlabel('Number of neurons sampled')
        ax.set_ylabel('Mean \u0394R\u00b2')
        ax.set_title('\u0394R\u00b2 vs. neuron subsampling')

        # Panel C: Control accuracy
        ax = axes[2]
        ax.bar(['Physics \u2192 Behavior'], [control_acc],
               yerr=[control_acc_std], color=COLORS['control'], capsize=3,
               width=0.5)
        ax.axhline(0.5, color='gray', linestyle='--', alpha=0.5, label='Chance')
        ax.set_ylabel('Accuracy')
        ax.set_title('Control: physics labels predict behavior')
        ax.set_ylim(0, 1)
        ax.legend()

        fig.align_ylabels(axes)
        plt.tight_layout()
        fig_path = f"{fig_dir}/encoding_analysis.pdf"
        plt.savefig(fig_path)
        plt.close()


def plot_rsa(plot_data, fig_dir="figures"):
    rdm_neural = plot_data['rdm_neural']
    rdm_pixel = plot_data['rdm_pixel']
    rdm_physics = plot_data['rdm_physics']
    n_sub = int(plot_data['n_sub'])
    corr_neural_pixel = float(plot_data['corr_neural_pixel'])
    corr_neural_physics = float(plot_data['corr_neural_physics'])

    with paper_style():
        # Reorder scenes via hierarchical clustering on pixel RDM
        # so shared structure between Neural and Sensory is visible
        rdm_pixel_full = squareform(rdm_pixel)
        Z = linkage(rdm_pixel, method='average')
        order = leaves_list(Z)

        n_show = min(40, n_sub)
        order = order[:n_show]
        rdm_neural_sq = squareform(rdm_neural)[np.ix_(order, order)]
        rdm_pixel_sq = rdm_pixel_full[np.ix_(order, order)]
        rdm_physics_sq = squareform(rdm_physics)[np.ix_(order, order)]

        # Shared color range across all RDMs
        all_rdms = [rdm_neural_sq, rdm_pixel_sq, rdm_physics_sq]
        vmin = min(r.min() for r in all_rdms)
        vmax = max(r.max() for r in all_rdms)

        # Top row: RDM | colorbar | RDM; bottom row: RDM | bar chart
        fig = plt.figure(figsize=(COL_WIDTH, COL_WIDTH * 0.9))
        gs = fig.add_gridspec(2, 3, width_ratios=[1, 0.05, 1],
                              hspace=0.45, wspace=0.15)

        # Top-left: Neural RDM
        ax_nr = fig.add_subplot(gs[0, 0])
        im = ax_nr.imshow(rdm_neural_sq, cmap='viridis', aspect='equal',
                          vmin=vmin, vmax=vmax)
        ax_nr.set_title('Neural RDM')
        ax_nr.set_xticks([]); ax_nr.set_yticks([])

        # Top-right: Pixel RDM
        ax_rr = fig.add_subplot(gs[0, 2])
        ax_rr.imshow(rdm_pixel_sq, cmap='viridis', aspect='equal',
                     vmin=vmin, vmax=vmax)
        ax_rr.set_title(f'{LABEL_RENDER} RDM')
        ax_rr.set_xticks([]); ax_rr.set_yticks([])

        # Vertical colorbar between top two RDMs
        cax = fig.add_subplot(gs[0, 1])
        cb = fig.colorbar(im, cax=cax, orientation='vertical')
        cb.ax.tick_params(labelsize=4, length=1.5, pad=1)
        cb.set_ticks([vmin, (vmin + vmax) / 2, vmax])
        cb.set_ticklabels([f'{vmin:.1f}', f'{(vmin+vmax)/2:.1f}',
                           f'{vmax:.1f}'])

        # Bottom-left: Physics RDM
        ax_pr = fig.add_subplot(gs[1, 0])
        ax_pr.imshow(rdm_physics_sq, cmap='viridis', aspect='equal',
                     vmin=vmin, vmax=vmax)
        ax_pr.set_title(f'{LABEL_PHYSICS} RDM')
        ax_pr.set_xticks([]); ax_pr.set_yticks([])

        # Correlation bar plot spanning bottom middle+right
        ax = fig.add_subplot(gs[1, 2])
        labels = [f'Neural\u2013\n{LABEL_RENDER}', f'Neural\u2013\n{LABEL_PHYSICS}']
        values = [corr_neural_pixel, corr_neural_physics]
        colors = [COLORS['pixels'], COLORS['physics']]
        bars = ax.bar(labels, values, color=colors, width=0.6)
        ax.set_ylabel('Spearman r')
        ax.set_title('RSA correlations')
        ax.axhline(0, color='gray', linestyle='-', alpha=0.3)
        for bar, val in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                    f'{val:.3f}', ha='center', va='bottom')

        fig_path = f"{fig_dir}/rsa_analysis.pdf"
        plt.savefig(fig_path)
        plt.close()


def plot_dissociation(plot_data, fig_dir="figures"):
    r2_pixel = plot_data['r2_pixel']
    r2_physics = plot_data['r2_physics']
    n_neurons = len(r2_pixel)
    mean_r2_pixel = r2_pixel.mean()
    mean_r2_physics = r2_physics.mean()
    pixel_score = float(plot_data['pixel_score'])
    physics_score = float(plot_data['physics_score'])
    metric_label = str(plot_data['metric_label'])
    chance_val = float(plot_data['chance'])
    chance = None if np.isnan(chance_val) else chance_val

    with paper_style():
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(COL_WIDTH, 2.0))

        bar_width = 0.5
        colors = [COLORS['pixels'], COLORS['physics']]
        labels = [LABEL_RENDER, LABEL_PHYSICS]

        bars1 = ax1.bar(labels, [mean_r2_pixel, mean_r2_physics],
                        width=bar_width, color=colors,
                        yerr=[r2_pixel.std() / np.sqrt(n_neurons),
                              r2_physics.std() / np.sqrt(n_neurons)],
                        capsize=3)
        ax1.set_ylabel('Neural R\u00b2')
        ax1.set_title('Encoding performance')
        for bar, val in zip(bars1, [mean_r2_pixel, mean_r2_physics]):
            ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                     f'{val:.3f}', ha='center', va='bottom', fontweight='bold')

        bars2 = ax2.bar(labels, [pixel_score, physics_score],
                        width=bar_width, color=colors, capsize=3)
        if chance is not None:
            ax2.axhline(chance, color='gray', linestyle='--', alpha=0.5, label='Chance')
            ax2.set_ylim(0, 1.1)
            ax2.legend()
        ax2.set_ylabel(metric_label)
        ax2.set_title('Computational sufficiency')
        for bar, val in zip(bars2, [pixel_score, physics_score]):
            ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                     f'{val:.3f}', ha='center', va='bottom', fontweight='bold')

        plt.tight_layout()
        fig_path = f"{fig_dir}/dissociation.pdf"
        plt.savefig(fig_path)
        plt.close()


def plot_dissociation_combined(plot_data, fig_dir="figures"):
    """Dissociation figure comparing pixel vs. pixel+physics models.

    The two encoding bars are nearly identical height, showing that adding
    physics to the pixel model barely changes neural R² — even though
    physics dramatically improves behavioral prediction.
    """
    r2_pixel = plot_data['r2_pixel']
    r2_combined = plot_data['r2_combined']
    n_neurons = len(r2_pixel)
    mean_r2_pixel = r2_pixel.mean()
    mean_r2_combined = r2_combined.mean()
    pixel_score = float(plot_data['pixel_score'])
    combined_score = float(plot_data['combined_score'])
    metric_label = str(plot_data['metric_label'])
    chance_val = float(plot_data['chance'])
    chance = None if np.isnan(chance_val) else chance_val

    with paper_style():
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(COL_WIDTH, 2.0))

        bar_width = 0.5
        colors = [COLORS['pixels'], COLORS['combined']]
        labels = [LABEL_RENDER, LABEL_RENDER_PLUS_PHYSICS]

        bars1 = ax1.bar(labels, [mean_r2_pixel, mean_r2_combined],
                        width=bar_width, color=colors,
                        yerr=[r2_pixel.std() / np.sqrt(n_neurons),
                              r2_combined.std() / np.sqrt(n_neurons)],
                        capsize=3)
        ax1.set_ylabel('Neural R\u00b2')
        ax1.set_title('Encoding performance')
        for bar, val in zip(bars1, [mean_r2_pixel, mean_r2_combined]):
            ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                     f'{val:.3f}', ha='center', va='bottom', fontweight='bold')

        bars2 = ax2.bar(labels, [pixel_score, combined_score],
                        width=bar_width, color=colors, capsize=3)
        if chance is not None:
            ax2.axhline(chance, color='gray', linestyle='--', alpha=0.5, label='Chance')
            ax2.set_ylim(0, 1.1)
            ax2.legend()
        ax2.set_ylabel(metric_label)
        ax2.set_title('Computational sufficiency')
        for bar, val in zip(bars2, [pixel_score, combined_score]):
            ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                     f'{val:.3f}', ha='center', va='bottom', fontweight='bold')

        plt.tight_layout()
        fig_path = f"{fig_dir}/dissociation_combined.pdf"
        plt.savefig(fig_path)
        plt.close()


def plot_predicted_frames(plot_data, fig_dir="figures"):
    """Predicted frames grid — uses figure* (full width) for legibility."""
    init_imgs = plot_data['predicted_init_imgs']
    pixel_imgs = plot_data['predicted_pixel_imgs']
    physics_imgs = plot_data['predicted_physics_imgs']
    final_imgs = plot_data['predicted_final_imgs']
    n = len(init_imgs)

    col_titles = ['t=0 (input)', f'{LABEL_RENDER} model\nprediction',
                  'Physics model\nprediction', 't=N (actual)']
    cols = [init_imgs, pixel_imgs, physics_imgs, final_imgs]

    with paper_style():
        fig, axes = plt.subplots(n, 4, figsize=(FULL_WIDTH, 1.6 * n))
        if n == 1:
            axes = axes[np.newaxis, :]

        for col_idx, (title, imgs) in enumerate(zip(col_titles, cols)):
            axes[0, col_idx].set_title(title)
            for row_idx in range(n):
                axes[row_idx, col_idx].imshow(imgs[row_idx])
                axes[row_idx, col_idx].axis('off')

        plt.tight_layout(pad=0.3)
        fig_path = f"{fig_dir}/predicted_frames.pdf"
        plt.savefig(fig_path)
        plt.close()


def plot_predicted_frames_compact(plot_data, fig_dir="figures", scene_idx=1):
    """Predicted frames as a 2x2 grid for a single-column figure."""
    init_imgs = plot_data['predicted_init_imgs']
    pixel_imgs = plot_data['predicted_pixel_imgs']
    physics_imgs = plot_data['predicted_physics_imgs']
    final_imgs = plot_data['predicted_final_imgs']

    titles = ['t=0 (input)', 'Sensory model\nprediction',
              'Physics model\nprediction', 't=N (actual)']
    imgs = [init_imgs[scene_idx], pixel_imgs[scene_idx],
            physics_imgs[scene_idx], final_imgs[scene_idx]]

    with paper_style():
        fig, axes = plt.subplots(2, 2, figsize=(COL_WIDTH, COL_WIDTH * 0.95),
                                 gridspec_kw={'hspace': 0.15, 'wspace': 0.05})

        for ax, title, img in zip(axes.flat, titles, imgs):
            ax.imshow(img)
            ax.set_title(title, pad=2)
            ax.axis('off')

        fig_path = f"{fig_dir}/predicted_frames_compact.pdf"
        plt.savefig(fig_path)
        plt.close()


def plot_dynamics(plot_data, fig_dir="figures"):
    r2_physics_forward = plot_data['r2_physics_forward']
    r2_pixel_forward = plot_data['r2_pixel_forward']
    n_neurons = len(r2_physics_forward)
    mean_r2_physics = r2_physics_forward.mean()
    mean_r2_pixel = r2_pixel_forward.mean()

    with paper_style():
        fig, ax1 = plt.subplots(1, 1, figsize=(COL_WIDTH * 0.6, 2.2))

        bar_width = 0.5
        # Bar 1 is the pixel forward model (RGBA-only prediction per glossary),
        # not the full render slice — label accordingly.
        colors = [COLORS['pixels'], COLORS['physics']]
        labels = ['Pixel', LABEL_PHYSICS]

        bars1 = ax1.bar(labels, [mean_r2_pixel, mean_r2_physics],
                        width=bar_width, color=colors,
                        yerr=[r2_pixel_forward.std() / np.sqrt(n_neurons),
                              r2_physics_forward.std() / np.sqrt(n_neurons)],
                        capsize=3)
        ax1.set_ylabel('Future neural R\u00b2')
        ax1.set_title('Future brain state prediction')
        for bar, val in zip(bars1, [mean_r2_pixel, mean_r2_physics]):
            ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                     f'{val:.3f}', ha='center', va='bottom', fontweight='bold')

        plt.tight_layout()
        fig_path = f"{fig_dir}/dynamics_analysis.pdf"
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

    # Figure 1: elbow + scatter (stacked vertically)
    with paper_style():
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(COL_WIDTH, 4.5))

        # Elbow + decoding reference
        ax1.plot(range(1, n_neurons + 1), cumvar, color=COLORS['pixels'])
        ax1.axhline(all_pc_acc, color=COLORS['physics'], linestyle='--',
                    linewidth=1.0,
                    label=f'Motion decoding (all PCs): {all_pc_acc:.1%}')
        ax1.axhline(0.5, color='gray', linestyle=':', alpha=0.5,
                    label='Chance (50%)')
        ax1.set_xlabel('Number of principal components')
        ax1.set_ylabel('Cumulative explained variance')
        ax1.set_title('PCA elbow plot + motion decoding')
        ax1.legend()
        ax1.set_xlim(1, n_neurons)

        # PC1/PC2 scatter colored by motion direction
        lo, hi = 1, 99
        pc1_lim = np.percentile(pc1, [lo, hi])
        pc2_lim = np.percentile(pc2, [lo, hi])
        colors = np.where(motion_dir == 1, COLORS['physics'], COLORS['pixels'])
        ax2.scatter(pc1, pc2, c=colors, alpha=0.3, s=4, edgecolors='none')
        pad1 = 0.05 * (pc1_lim[1] - pc1_lim[0])
        pad2 = 0.05 * (pc2_lim[1] - pc2_lim[0])
        ax2.set_xlim(pc1_lim[0] - pad1, pc1_lim[1] + pad1)
        ax2.set_ylim(pc2_lim[0] - pad2, pc2_lim[1] + pad2)
        ax2.set_xlabel('PC1')
        ax2.set_ylabel('PC2')
        ax2.set_title('PC1 vs PC2 (colored by motion direction)')
        ax2.legend(handles=[
            Line2D([0], [0], marker='o', color='w',
                   markerfacecolor=COLORS['pixels'], markersize=5,
                   label='Left'),
            Line2D([0], [0], marker='o', color='w',
                   markerfacecolor=COLORS['physics'], markersize=5,
                   label='Right'),
        ])

        # Inset: decoding accuracy vs number of PCs
        ax_inset = ax2.inset_axes([0.58, 0.05, 0.40, 0.42])
        ax_inset.plot(pc_counts, decode_accs, marker='o',
                      color=COLORS['pixels'], markersize=3)
        ax_inset.axhline(0.5, color='gray', linestyle='--', alpha=0.5)
        ax_inset.set_xlabel('# PCs', fontsize=5)
        ax_inset.set_ylabel('Accuracy', fontsize=5)
        ax_inset.set_title('Motion decoding', fontsize=6)
        ax_inset.tick_params(labelsize=5)
        ax_inset.set_xscale('log')
        ax_inset.set_ylim(0.4, 1.0)

        fig.align_ylabels([ax1, ax2])
        plt.tight_layout()
        fig_path = f"{fig_dir}/pca_analysis.pdf"
        plt.savefig(fig_path)
        plt.close()

    # Figure 2: elbow + decoding overlay with twinx
    with paper_style():
        fig, ax_var = plt.subplots(figsize=(COL_WIDTH * 0.7, 1.6))
        ax_dec = ax_var.twinx()

        ax_var.plot(range(1, n_neurons + 1), cumvar,
                    color=COLORS['pixels'], label='Cumulative\nvariance')
        ax_var.set_xlabel('Number of principal components')
        ax_var.set_ylabel('Cumulative explained variance',
                          color=COLORS['pixels'])
        ax_var.tick_params(axis='y', labelcolor=COLORS['pixels'])
        ax_var.set_xlim(1, n_neurons)
        ax_var.set_ylim(0, 1.05)
        ax_var.set_xscale("log")
        ax_var.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1.0))

        ax_dec.plot(pc_counts, decode_accs, marker='o',
                    color=COLORS['physics'], markersize=3,
                    label='Motion decoding\naccuracy')
        ax_dec.axhline(0.5, color='gray', linestyle=':', alpha=0.5)
        ax_dec.set_ylabel('Decoding accuracy', color=COLORS['physics'])
        ax_dec.tick_params(axis='y', labelcolor=COLORS['physics'])
        ax_dec.set_ylim(0.4, 1.05)
        ax_dec.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1.0))

        lines_var, labels_var = ax_var.get_legend_handles_labels()
        lines_dec, labels_dec = ax_dec.get_legend_handles_labels()
        ax_var.legend(lines_var + lines_dec, labels_var + labels_dec,
                      loc='upper left')

        # ax_var.set_title('PCA variance vs motion decoding')

        fig_path2 = f"{fig_dir}/pca_variance_decoding.pdf"
        plt.savefig(fig_path2)
        plt.close()


def plot_sample_scenes(initial_renders, target_renders, rgba_bytes,
                       image_size, n_timesteps, fig_dir="figures",
                       n_samples=6):
    n = min(n_samples, len(initial_renders))
    with paper_style():
        fig, axes = plt.subplots(n, 2, figsize=(COL_WIDTH, 1.6 * n))
        if n == 1:
            axes = axes[np.newaxis, :]

        axes[0, 0].set_title('t = 0 (initial)')
        axes[0, 1].set_title(f't = {n_timesteps} (target)')

        for i in range(n):
            init_rgba = initial_renders[i, :rgba_bytes].astype(np.uint8).reshape(
                image_size, image_size, 4)
            target_rgba = target_renders[i, :rgba_bytes].astype(np.uint8).reshape(
                image_size, image_size, 4)
            axes[i, 0].imshow(init_rgba)
            axes[i, 0].axis('off')
            axes[i, 1].imshow(target_rgba)
            axes[i, 1].axis('off')

        plt.tight_layout(pad=0.3)
        fig_path = f"{fig_dir}/sample_scenes.pdf"
        plt.savefig(fig_path)
        plt.close()


def plot_residual(plot_data, fig_dir="figures"):
    r2_raw_pixel = plot_data['r2_raw_pixel']
    r2_raw_physics_gt = plot_data['r2_raw_physics_gt']
    r2_resid_pixel = plot_data['r2_resid_pixel']
    r2_resid_physics_gt = plot_data['r2_resid_physics_gt']
    var_kept = float(plot_data['residual_variance_fraction'])

    n = len(r2_raw_physics_gt)

    raw_means = np.array([r2_raw_pixel.mean(), r2_raw_physics_gt.mean()])
    resid_means = np.array([r2_resid_pixel.mean(), r2_resid_physics_gt.mean()])
    raw_sems = np.array([r2_raw_pixel.std() / np.sqrt(n),
                         r2_raw_physics_gt.std() / np.sqrt(n)])
    resid_sems = np.array([r2_resid_pixel.std() / np.sqrt(n),
                           r2_resid_physics_gt.std() / np.sqrt(n)])

    with paper_style():
        fig, axes = plt.subplots(1, 2, figsize=(FULL_WIDTH, 2.4))

        # Panel A: per-neuron scatter, raw vs residualized GT physics R²
        ax = axes[0]
        ax.scatter(r2_raw_physics_gt, r2_resid_physics_gt,
                   s=6, alpha=0.5, color=COLORS['physics'],
                   edgecolors='none')
        lo = float(min(r2_raw_physics_gt.min(), r2_resid_physics_gt.min()))
        hi = float(max(r2_raw_physics_gt.max(), r2_resid_physics_gt.max()))
        pad = 0.05 * (hi - lo if hi > lo else 1.0)
        ax.plot([lo - pad, hi + pad], [lo - pad, hi + pad],
                color='gray', linestyle='--', linewidth=0.8, label='y = x')
        ax.axhline(0, color='gray', linestyle=':', linewidth=0.6)
        ax.set_xlim(lo - pad, hi + pad)
        ax.set_ylim(lo - pad, hi + pad)
        ax.set_xlabel('R² (raw neural ~ GT physics)')
        ax.set_ylabel('R² (residualized neural ~ GT physics)')
        ax.set_title('Per-neuron collapse after pixel-residualization')
        ax.legend(loc='upper left')

        # Panel B: grouped bars — raw vs residualized for each predictor set
        ax = axes[1]
        x = np.arange(2)
        width = 0.35
        ax.bar(x - width / 2, raw_means, width, yerr=raw_sems,
               color=COLORS['pixels'], capsize=3, label='Raw neural')
        ax.bar(x + width / 2, resid_means, width, yerr=resid_sems,
               color=COLORS['physics'], capsize=3, label='Residualized neural')
        ax.axhline(0, color='gray', linestyle='--', linewidth=0.6)
        ax.set_xticks(x)
        ax.set_xticklabels(['Pixel', 'GT Physics'])
        ax.set_ylabel('Mean R²')
        ax.set_title(f'Encoding R² (residual var fraction = {var_kept:.2f})')
        ax.legend()

        plt.tight_layout()
        fig_path = f"{fig_dir}/residual_analysis.pdf"
        plt.savefig(fig_path)
        plt.close()
