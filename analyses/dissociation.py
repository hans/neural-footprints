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
from sklearn.model_selection import cross_val_score, KFold
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
    alphas = np.logspace(-2, 6, 20)
    kf = KFold(n_splits=5, shuffle=True, random_state=42)

    def mean_cv_r2(X, Y):
        r2s = []
        for train, test in kf.split(X):
            ridge = RidgeCV(alphas=alphas)
            ridge.fit(X[train], Y[train])
            r2s.append(ridge.score(X[test], Y[test]))
        return float(np.mean(r2s))

    render_r2 = mean_cv_r2(pixel_pca_initial, final_pixel_pca)
    physics_r2 = mean_cv_r2(physics_initial_scaled, final_pixel_pca)
    return render_r2, physics_r2, "Next-frame pred. R²", None


def _score_kinetic_energy(pixel_pca, physics_scaled, behavior_labels):
    """
    Logistic regression accuracy predicting KE binary label.

    Returns (render_score, physics_score, metric_label, chance_line).
    """
    log_reg_render = LogisticRegressionCV(cv=5, max_iter=1000, random_state=42)
    render_scores = cross_val_score(
        log_reg_render, pixel_pca, behavior_labels, cv=5, scoring='accuracy'
    )

    log_reg_physics = LogisticRegressionCV(cv=5, max_iter=1000, random_state=42)
    physics_scores = cross_val_score(
        log_reg_physics, physics_scaled, behavior_labels, cv=5, scoring='accuracy'
    )

    return render_scores.mean(), physics_scores.mean(), "Behavior accuracy", 0.5


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
        render_std = physics_std = None  # std not returned for R² scorer

    elif objective == "kinetic_energy":
        render_score, physics_score, metric_label, chance = _score_kinetic_energy(
            pixel_pca, physics_scaled, behavior_labels
        )
        render_std = physics_std = None

    else:
        raise ValueError(f"Unknown objective: {objective!r}. "
                         "Use 'next_frame_pixels' or 'kinetic_energy'.")

    print(f"  Render  → {metric_label}: {render_score:.4f}")
    print(f"  Physics → {metric_label}: {physics_score:.4f}")

    print(f"\n  DISSOCIATION:")
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
