"""
Simulation 3: R² vs. Behavioral Sufficiency Dissociation.

Directly visualizes the disconnect: the feature set that explains most neural
variance (render) is useless for predicting behavior, while the feature set
that predicts behavior (physics) adds essentially nothing to the encoding model.

Two behavioral sufficiency objectives are supported (set in config.py):
  "next_frame_pixels": Ridge R² predicting final-frame pixels from t=0 features.
      Physics wins (has velocities + occluded state); render fails (no velocity,
      blind to occluded objects).
  "kinetic_energy":    Logistic accuracy on KE binary label.
      Physics wins (~100%, KE is deterministic from mass+velocity);
      render at chance (~50%, pixels carry no velocity signal).
"""

import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import RidgeCV, LogisticRegressionCV
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

from config import PIXEL_PCA_DIM, BEHAVIORAL_OBJECTIVE


# ---------------------------------------------------------------------------
# Behavioral sufficiency scorers
# ---------------------------------------------------------------------------

def _score_next_frame_pixels(pixel_pca_initial, physics_initial_scaled, final_pixel_pca):
    """
    Ridge regression R² predicting final-frame pixel PCs from t=0 features.

    Features (t=0):
      - pixel_pca_initial:        PCA of initial RGBA pixels
      - physics_initial_scaled:   standardized initial physics labels

    Target: final_pixel_pca — PCA of final RGBA pixels.

    Returns (render_score, physics_score, metric_label, chance_line).
    """
    ridge = RidgeCV(alphas=np.logspace(-2, 6, 20))
    render_r2 = cross_val_score(ridge, pixel_pca_initial, final_pixel_pca, cv=5, scoring='r2').mean()
    physics_r2 = cross_val_score(
        ridge,
        np.concatenate([physics_initial_scaled, pixel_pca_initial], axis=1),
        final_pixel_pca,
        cv=5,
        scoring='r2'
    ).mean()
    return render_r2, physics_r2, "Next-frame pred. R²", None


def _save_predicted_frames(
    pixel_pca_initial, physics_initial_scaled,
    final_pixel_pca, scaler_pix, pca_final,
    initial_renders, program_states, pixel_indices,
    fig_dir, n_samples=8
):
    """
    Train each model on all data, reconstruct predicted final frames to pixel
    space, and save a grid figure.

    Grid: n_samples rows × 4 cols
      Col 0: initial frame (t=0, model input)
      Col 1: render model prediction
      Col 2: physics model prediction
      Col 3: actual final frame (t=N, ground truth)
    Saved to {fig_dir}/predicted_frames.png.
    """
    from config import IMAGE_SIZE

    n = min(n_samples, len(initial_renders))
    phys_plus_pix = np.concatenate([physics_initial_scaled, pixel_pca_initial], axis=1)

    ridge = RidgeCV(alphas=np.logspace(-2, 6, 20))

    ridge.fit(pixel_pca_initial, final_pixel_pca)
    render_pred_pca = ridge.predict(pixel_pca_initial[:n])

    ridge.fit(phys_plus_pix, final_pixel_pca)
    physics_pred_pca = ridge.predict(phys_plus_pix[:n])

    def pca_to_img(pred_pca):
        pred_scaled = pca_final.inverse_transform(pred_pca)
        pred_pixels = scaler_pix.inverse_transform(pred_scaled)
        return np.clip(pred_pixels, 0, 255).astype(np.uint8).reshape(n, IMAGE_SIZE, IMAGE_SIZE, 4)

    render_imgs = pca_to_img(render_pred_pca)
    physics_imgs = pca_to_img(physics_pred_pca)
    init_imgs = initial_renders[:n].astype(np.uint8).reshape(n, IMAGE_SIZE, IMAGE_SIZE, 4)
    final_imgs = program_states[:n, pixel_indices].astype(np.uint8).reshape(n, IMAGE_SIZE, IMAGE_SIZE, 4)

    col_titles = ['t=0 (input)', 'Render model\nprediction', 'Physics model\nprediction', 't=N (actual)']
    cols = [init_imgs, render_imgs, physics_imgs, final_imgs]

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
    plt.savefig(fig_path, dpi=120, bbox_inches='tight')
    plt.close()
    print(f"Predicted frames saved: {fig_path}")


def _score_kinetic_energy(pixel_pca, physics_scaled, behavior_labels):
    """
    Logistic regression accuracy predicting KE binary label.

    Returns (render_score, physics_score, metric_label, chance_line).
    """
    log_reg = LogisticRegressionCV(cv=5, max_iter=1000, random_state=42)
    render_r2 = cross_val_score(log_reg, pixel_pca, behavior_labels, cv=5, scoring='accuracy').mean()
    physics_r2 = cross_val_score(log_reg, physics_scaled, behavior_labels, cv=5, scoring='accuracy').mean()
    return render_r2, physics_r2, "Behavior accuracy", 0.5


# ---------------------------------------------------------------------------
# Main analysis
# ---------------------------------------------------------------------------

def run_dissociation_analysis(neural_activity, scenes, neural_meta,
                               objective=None, fig_dir="figures"):
    """
    Compute and plot the R² vs. behavioral sufficiency dissociation.

    Two models:
      Render model:  pixel PCA features (PIXEL_PCA_DIM dims)
      Physics model: API physics labels (15*N_OBJECTS dims)

    For each, measure:
      - Neural R² (how well it predicts neural activity)
      - Behavioral sufficiency score (determined by `objective`)
    """
    if objective is None:
        objective = BEHAVIORAL_OBJECTIVE

    print("\n" + "=" * 60)
    print("SIMULATION 3: R² vs. Behavioral Sufficiency Dissociation")
    print(f"  Objective: {objective}")
    print("=" * 60)

    program_states = scenes['program_states']
    physics_labels = scenes['physics_labels']
    initial_physics_labels = scenes['initial_physics_labels']
    initial_renders = scenes['initial_renders']
    behavior_labels = scenes['behavior_labels']
    pixel_indices = scenes['metadata']['pixel_indices']

    n_scenes, n_neurons = neural_activity.shape

    # --- Prepare final-frame render features (used for neural R²) ---
    print("\nPreparing render features (pixel PCA of final frame)...")
    pixel_data = program_states[:, pixel_indices]
    scaler_pix = StandardScaler()
    pixel_scaled = scaler_pix.fit_transform(pixel_data)
    pca = PCA(n_components=PIXEL_PCA_DIM, random_state=42)
    pixel_pca = pca.fit_transform(pixel_scaled)

    scaler_phys = StandardScaler()
    physics_scaled = scaler_phys.fit_transform(physics_labels)

    # --- Neural R² ---
    print("Computing neural R² for each model...")
    alphas = np.logspace(-2, 6, 20)

    r2_render = np.zeros(n_neurons)
    for j in range(n_neurons):
        ridge = RidgeCV(alphas=alphas)
        ridge.fit(pixel_pca, neural_activity[:, j])
        r2_render[j] = ridge.score(pixel_pca, neural_activity[:, j])

    r2_physics = np.zeros(n_neurons)
    for j in range(n_neurons):
        ridge = RidgeCV(alphas=alphas)
        ridge.fit(physics_scaled, neural_activity[:, j])
        r2_physics[j] = ridge.score(physics_scaled, neural_activity[:, j])

    mean_r2_render = r2_render.mean()
    mean_r2_physics = r2_physics.mean()

    print(f"  Render model mean R²:  {mean_r2_render:.4f}")
    print(f"  Physics model mean R²: {mean_r2_physics:.4f}")

    # --- Behavioral sufficiency ---
    print(f"Computing behavioral sufficiency ({objective})...")

    if objective == "next_frame_pixels":
        # Features: initial state (t=0). Target: final-frame pixel PCs.
        scaler_init_pix = StandardScaler()
        init_pix_scaled = scaler_init_pix.fit_transform(initial_renders)
        pca_init = PCA(n_components=PIXEL_PCA_DIM, random_state=42)
        pixel_pca_initial = pca_init.fit_transform(init_pix_scaled)

        scaler_init_phys = StandardScaler()
        physics_initial_scaled = scaler_init_phys.fit_transform(initial_physics_labels)

        # Final-frame pixel target (same PCA space as above for consistency)
        pca_final = PCA(n_components=PIXEL_PCA_DIM, random_state=42)
        final_pixel_pca = pca_final.fit_transform(pixel_scaled)

        render_score, physics_score, metric_label, chance = _score_next_frame_pixels(
            pixel_pca_initial, physics_initial_scaled, final_pixel_pca
        )

        _save_predicted_frames(
            pixel_pca_initial, physics_initial_scaled,
            final_pixel_pca, scaler_pix, pca_final,
            initial_renders, program_states, pixel_indices,
            fig_dir
        )

    elif objective == "kinetic_energy":
        render_score, physics_score, metric_label, chance = _score_kinetic_energy(
            pixel_pca, physics_scaled, behavior_labels
        )

    else:
        raise ValueError(f"Unknown objective: {objective!r}. "
                         "Use 'next_frame_pixels' or 'kinetic_energy'.")

    print(f"  Render  → {metric_label}: {render_score:.4f}")
    print(f"  Physics → {metric_label}: {physics_score:.4f}")

    print("\n  DISSOCIATION:")
    print(f"    Render model:  R² = {mean_r2_render:.4f}  |  {metric_label} = {render_score:.4f}")
    print(f"    Physics model: R² = {mean_r2_physics:.4f}  |  {metric_label} = {physics_score:.4f}")

    # --- Figure ---
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))

    bar_width = 0.5
    colors = ['#4878CF', '#D65F5F']
    labels = ['Render\nmodel', 'Physics\nmodel']

    # Left panel: Neural R²
    bars1 = ax1.bar(labels, [mean_r2_render, mean_r2_physics],
                    width=bar_width, color=colors,
                    yerr=[r2_render.std() / np.sqrt(n_neurons),
                          r2_physics.std() / np.sqrt(n_neurons)],
                    capsize=5)
    ax1.set_ylabel('Neural variance explained (R²)', fontsize=12)
    ax1.set_title('Encoding model performance', fontsize=13)
    for bar, val in zip(bars1, [mean_r2_render, mean_r2_physics]):
        ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                 f'{val:.3f}', ha='center', va='bottom', fontsize=11, fontweight='bold')

    # Right panel: behavioral sufficiency
    bars2 = ax2.bar(labels, [render_score, physics_score],
                    width=bar_width, color=colors, capsize=5)
    if chance is not None:
        ax2.axhline(chance, color='gray', linestyle='--', alpha=0.5, label='Chance')
        ax2.set_ylim(0, 1.1)
        ax2.legend(fontsize=10)
    ax2.set_ylabel(metric_label, fontsize=12)
    ax2.set_title('Behavioral sufficiency', fontsize=13)
    for bar, val in zip(bars2, [render_score, physics_score]):
        ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                 f'{val:.3f}', ha='center', va='bottom', fontsize=11, fontweight='bold')

    plt.tight_layout()
    fig_path = f"{fig_dir}/dissociation.png"
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\nFigure saved: {fig_path}")

    return {
        'mean_r2_render': mean_r2_render,
        'mean_r2_physics': mean_r2_physics,
        'render_behavioral_score': render_score,
        'physics_behavioral_score': physics_score,
        'metric_label': metric_label,
        'objective': objective,
    }
