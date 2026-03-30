"""
Simulation 2: RSA dominated by render structure.

Shows that neural RDM tracks the render RDM (high correlation)
but not the physics RDM, and partial correlation removes any residual physics signal.
"""

import numpy as np
from scipy.spatial.distance import pdist, squareform
from scipy.stats import spearmanr
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

from analyses.plot_style import COLORS, paper_style

from config import RSA_SUBSAMPLE as _CFG_RSA_SUBSAMPLE, PIXEL_PCA_DIM as _CFG_PIXEL_PCA_DIM


def _compute_rdm(data):
    """Compute representational dissimilarity matrix using correlation distance."""
    return pdist(data, metric='correlation')


def _partial_spearman(x, y, z):
    """
    Partial Spearman correlation between x and y, controlling for z.
    Uses rank-based residualization.
    """
    from scipy.stats import rankdata

    rx = rankdata(x)
    ry = rankdata(y)
    rz = rankdata(z)

    # Residualize x and y on z using linear regression
    # residual_x = rx - (rx . rz / rz . rz) * rz
    def residualize(a, b):
        b_centered = b - b.mean()
        a_centered = a - a.mean()
        beta = np.dot(a_centered, b_centered) / np.dot(b_centered, b_centered)
        return a_centered - beta * b_centered

    res_x = residualize(rx, rz)
    res_y = residualize(ry, rz)

    corr, pval = spearmanr(res_x, res_y)
    return corr, pval


def run_rsa_analysis(neural_activity, scenes, neural_meta, fig_dir="figures",
                     *, rsa_subsample=None, pixel_pca_dim=None):
    """
    Run RSA analysis on a subsample of scenes.

    1. Compute RDMs for neural, render, and physics spaces
    2. Spearman correlations: neural<->render (high), neural<->physics (low)
    3. Partial correlation: neural<->physics | render -> near zero
    """
    if rsa_subsample is None:
        rsa_subsample = _CFG_RSA_SUBSAMPLE
    if pixel_pca_dim is None:
        pixel_pca_dim = _CFG_PIXEL_PCA_DIM

    print("\n" + "=" * 60)
    print("SIMULATION 2: RSA Dominated by Render Structure")
    print("=" * 60)

    program_states = scenes['program_states']
    physics_labels = scenes['physics_labels']
    render_indices = scenes['metadata']['render_indices']

    n_scenes = program_states.shape[0]
    n_sub = min(rsa_subsample, n_scenes)

    # Subsample scenes for tractability
    rng = np.random.default_rng(123)
    sub_idx = rng.choice(n_scenes, size=n_sub, replace=False)
    sub_idx.sort()

    neural_sub = neural_activity[sub_idx]
    pixel_sub = program_states[sub_idx][:, render_indices]
    physics_sub = physics_labels[sub_idx]

    # PCA-reduce pixel data for tractability
    print(f"\nSubsampled {n_sub} scenes for RSA.")
    print(f"PCA-reducing pixel data to {pixel_pca_dim} components...")
    scaler = StandardScaler()
    pixel_scaled = scaler.fit_transform(pixel_sub)
    pca = PCA(n_components=min(pixel_pca_dim, pixel_scaled.shape[0] - 1), random_state=42)
    pixel_pca = pca.fit_transform(pixel_scaled)

    # Standardize physics
    scaler_phys = StandardScaler()
    physics_scaled = scaler_phys.fit_transform(physics_sub)

    # Compute RDMs
    print("Computing RDMs...")
    rdm_neural = _compute_rdm(neural_sub)
    rdm_render = _compute_rdm(pixel_pca)
    rdm_physics = _compute_rdm(physics_scaled)

    # Handle NaN in RDMs (constant rows produce NaN in correlation distance)
    for rdm in [rdm_neural, rdm_render, rdm_physics]:
        rdm[np.isnan(rdm)] = 0.0

    # Spearman correlations
    corr_neural_render, p_nr = spearmanr(rdm_neural, rdm_render)
    corr_neural_physics, p_np = spearmanr(rdm_neural, rdm_physics)
    corr_render_physics, p_rp = spearmanr(rdm_render, rdm_physics)

    print(f"\n  Spearman neural<->render:  r={corr_neural_render:.4f}  (p={p_nr:.2e})")
    print(f"  Spearman neural<->physics: r={corr_neural_physics:.4f}  (p={p_np:.2e})")
    print(f"  Spearman render<->physics: r={corr_render_physics:.4f}  (p={p_rp:.2e})")

    # Partial correlation: neural<->physics | render
    partial_corr, partial_p = _partial_spearman(rdm_neural, rdm_physics, rdm_render)
    print(f"  Partial neural<->physics | render: r={partial_corr:.4f}  (p={partial_p:.2e})")

    # --- Figure: RDM heatmaps + correlation bar plot ---
    with paper_style():
        fig, axes = plt.subplots(1, 4, figsize=(20, 5))

        # RDM heatmaps (show square form of first 100 scenes for visibility)
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

        # Correlation bar plot
        ax = axes[3]
        labels = ['Neural↔Render', 'Neural↔Physics', 'Partial\nNeural↔Physics|Render']
        values = [corr_neural_render, corr_neural_physics, partial_corr]
        colors = [COLORS['pixels'], COLORS['physics'], COLORS['neutral']]
        bars = ax.bar(labels, values, color=colors)
        ax.set_ylabel('Spearman r')
        ax.set_title('RSA Correlations')
        ax.axhline(0, color='gray', linestyle='-', alpha=0.3)
        # Annotate values
        for bar, val in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                    f'{val:.3f}', ha='center', va='bottom', fontsize=9)

        plt.tight_layout()
        fig_path = f"{fig_dir}/rsa_analysis.png"
        plt.savefig(fig_path)
        plt.close()
        print(f"\nFigure saved: {fig_path}")

    return {
        'corr_neural_render': corr_neural_render,
        'corr_neural_physics': corr_neural_physics,
        'corr_render_physics': corr_render_physics,
        'partial_neural_physics': partial_corr,
    }
