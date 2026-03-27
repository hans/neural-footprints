"""
Simulation 4: Future brain state prediction — reverse dissociation.

Compares two forward models that attempt to predict future neural activity
from t=0 information:

  Physics forward model (oracle): resimulate each scene via PyBullet from
    initial physics state → reconstruct full program_state → encoding R².
    Since PyBullet is deterministic, this recovers ~the original encoding R².

  Pixel forward model (learned): MLP predicts final-frame pixels from initial
    pixels → use predicted pixels as encoding features → encoding R².
    Fails because initial pixels lack velocity and occluded-state information.

This is the reverse of the encoding analysis: physics (invisible to standard
encoding) is exactly what you need for temporal prediction.
"""

import numpy as np
from sklearn.decomposition import PCA
from sklearn.model_selection import cross_val_predict
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

from analyses.encoding import pca_reduce_pixels, ridge_r2_per_neuron
from analyses.dissociation import _make_mlp
from config import PIXEL_PCA_DIM as _CFG_PIXEL_PCA_DIM
from config import BEHAVIORAL_PCA_DIM as _CFG_BEHAVIORAL_PCA_DIM


def run_dynamics_analysis(neural_activity, scenes, neural_meta,
                          encoding_delta_r2, *,
                          fig_dir="figures",
                          pixel_pca_dim=None, behavioral_pca_dim=None):
    """
    Future brain state prediction via physics vs. pixel forward models.

    Parameters
    ----------
    neural_activity : ndarray [n_scenes x n_neurons]
        Neural activity generated from final-frame program_state (the target).
    scenes : dict
        Output of generate_scenes / load_scenes.
    neural_meta : dict
        Neural generation metadata.
    encoding_delta_r2 : float
        Mean ΔR² from the encoding analysis (for comparison plot).
    fig_dir : str
        Directory for output figures.
    pixel_pca_dim : int
        PCA components for encoding features.
    behavioral_pca_dim : int
        PCA components for MLP prediction target.
    """
    if pixel_pca_dim is None:
        pixel_pca_dim = _CFG_PIXEL_PCA_DIM
    if behavioral_pca_dim is None:
        behavioral_pca_dim = _CFG_BEHAVIORAL_PCA_DIM

    print("\n" + "=" * 60)
    print("SIMULATION 4: Future Brain State Prediction")
    print("=" * 60)

    from scene_generator import resimulate_scene

    program_states = scenes['program_states']
    initial_physics_labels = scenes['initial_physics_labels']
    initial_renders = scenes['initial_renders']
    scene_configs = scenes['scene_configs']
    pillar_grays = scenes['pillar_grays']
    pixel_indices = scenes['metadata']['pixel_indices']

    n_scenes, n_neurons = neural_activity.shape

    # ------------------------------------------------------------------
    # Physics forward model: resimulate each scene → full program_state
    # ------------------------------------------------------------------
    print("\nPhysics forward model: resimulating scenes from initial state...")
    resim_program_states = np.zeros_like(program_states)
    for i in range(n_scenes):
        if (i + 1) % 100 == 0 or i == 0:
            print(f"  Resimulating scene {i+1}/{n_scenes}...")
        resim_program_states[i] = resimulate_scene(
            scene_configs[i], initial_physics_labels[i],
            return_program_state=True,
            pillar_gray=pillar_grays[i],
        )

    print("  PCA-reducing resimulated pixels for encoding...")
    resim_pixel_data = resim_program_states[:, pixel_indices]
    resim_pixel_pca, _ = pca_reduce_pixels(resim_pixel_data, pixel_pca_dim)

    print("  Ridge regression: resimulated pixels → neural activity...")
    r2_physics_forward = ridge_r2_per_neuron(resim_pixel_pca, neural_activity)
    mean_r2_physics = r2_physics_forward.mean()
    print(f"  Physics forward model mean R²: {mean_r2_physics:.4f}")

    # ------------------------------------------------------------------
    # Pixel forward model: MLP predicts final pixels from initial pixels
    # ------------------------------------------------------------------
    print("\nPixel forward model: training MLP on initial → final pixels...")

    # PCA for MLP target (behavioral dim, whitened — matches dissociation)
    final_pixel_data = program_states[:, pixel_indices]
    scaler_final = StandardScaler()
    final_scaled = scaler_final.fit_transform(final_pixel_data)
    pca_final = PCA(n_components=behavioral_pca_dim, whiten=True, random_state=42)
    final_pixel_pca = pca_final.fit_transform(final_scaled)

    scaler_init = StandardScaler()
    init_scaled = scaler_init.fit_transform(initial_renders)
    pca_init = PCA(n_components=behavioral_pca_dim, whiten=True, random_state=42)
    init_pixel_pca = pca_init.fit_transform(init_scaled)

    # Cross-validated MLP predictions (out-of-fold to avoid double-dipping)
    pred_final_pca = cross_val_predict(_make_mlp(), init_pixel_pca, final_pixel_pca, cv=5)

    # Inverse-transform back to pixel space
    pred_final_scaled = pca_final.inverse_transform(pred_final_pca)
    pred_final_pixels = scaler_final.inverse_transform(pred_final_scaled)

    # PCA-reduce predicted pixels for encoding (same dim as physics model)
    print("  PCA-reducing MLP-predicted pixels for encoding...")
    pred_pixel_pca, _ = pca_reduce_pixels(pred_final_pixels, pixel_pca_dim)

    print("  Ridge regression: predicted pixels → neural activity...")
    r2_pixel_forward = ridge_r2_per_neuron(pred_pixel_pca, neural_activity)
    mean_r2_pixel = r2_pixel_forward.mean()
    print(f"  Pixel forward model mean R²: {mean_r2_pixel:.4f}")

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    gap = mean_r2_physics - mean_r2_pixel
    print(f"\n  FUTURE BRAIN STATE DISSOCIATION:")
    print(f"    Physics forward model R²: {mean_r2_physics:.4f}")
    print(f"    Pixel forward model R²:   {mean_r2_pixel:.4f}")
    print(f"    Gap:                      {gap:.4f}")
    print(f"    (cf. encoding ΔR² for current brain: {encoding_delta_r2:.4f})")

    # ------------------------------------------------------------------
    # Figure
    # ------------------------------------------------------------------
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))

    colors = ['#D65F5F', '#4878CF']
    labels = ['Physics\nforward', 'Pixel\nforward']

    # Left panel: future brain state R²
    bars1 = ax1.bar(labels, [mean_r2_physics, mean_r2_pixel],
                    width=0.5, color=colors,
                    yerr=[r2_physics_forward.std() / np.sqrt(n_neurons),
                          r2_pixel_forward.std() / np.sqrt(n_neurons)],
                    capsize=5)
    ax1.set_ylabel('Future neural R²', fontsize=12)
    ax1.set_title('Future brain state prediction', fontsize=13)
    for bar, val in zip(bars1, [mean_r2_physics, mean_r2_pixel]):
        ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                 f'{val:.3f}', ha='center', va='bottom', fontsize=11,
                 fontweight='bold')

    # Right panel: ΔR² comparison (current vs future brain)
    delta_labels = ['Current brain\n(encoding ΔR²)', 'Future brain\n(forward gap)']
    delta_vals = [encoding_delta_r2, gap]
    bars2 = ax2.bar(delta_labels, delta_vals,
                    width=0.5, color=['#999999', '#6ACC65'])
    ax2.set_ylabel('ΔR² / Gap', fontsize=12)
    ax2.set_title('Reverse dissociation', fontsize=13)
    ax2.axhline(0, color='gray', linestyle='--', alpha=0.3)
    for bar, val in zip(bars2, delta_vals):
        ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                 f'{val:.4f}', ha='center', va='bottom', fontsize=11,
                 fontweight='bold')

    plt.tight_layout()
    fig_path = f"{fig_dir}/dynamics_analysis.png"
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\nFigure saved: {fig_path}")

    return {
        'r2_physics_forward': r2_physics_forward,
        'r2_pixel_forward': r2_pixel_forward,
        'mean_r2_physics_forward': mean_r2_physics,
        'mean_r2_pixel_forward': mean_r2_pixel,
        'encoding_delta_r2': encoding_delta_r2,
        'forward_gap': gap,
    }
