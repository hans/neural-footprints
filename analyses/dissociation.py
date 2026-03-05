"""
Simulation 3: R² vs. Behavioral Sufficiency Dissociation.

Directly visualizes the disconnect: the feature set that explains most neural
variance (render) is useless for predicting behavior, while the feature set
that predicts behavior (physics) adds essentially nothing to the encoding model.

The behavior label uses peak displacement during the simulation (not final
positions) and depends on occluded objects — making it inaccessible from pixels.
"""

import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import RidgeCV, LogisticRegressionCV
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

from config import PIXEL_PCA_DIM


def run_dissociation_analysis(neural_activity, scenes, neural_meta, fig_dir="figures"):
    """
    Compute and plot the R² vs. behavioral sufficiency dissociation.

    Two models:
      Render model:  pixel PCA features (200 dims)
      Physics model: API physics labels (75 dims)

    For each, measure:
      - Neural R² (how well it predicts neural activity)
      - Behavior prediction accuracy (how well it predicts the behavioral outcome)
    """
    print("\n" + "=" * 60)
    print("SIMULATION 3: R² vs. Behavioral Sufficiency Dissociation")
    print("=" * 60)

    program_states = scenes['program_states']
    physics_labels = scenes['physics_labels']
    behavior_labels = scenes['behavior_labels']
    pixel_indices = scenes['metadata']['pixel_indices']

    n_scenes, n_neurons = neural_activity.shape

    # --- Prepare feature sets ---
    print("\nPreparing render features (pixel PCA)...")
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

    # --- Behavioral prediction ---
    print("Computing behavior prediction accuracy for each model...")

    log_reg_render = LogisticRegressionCV(cv=5, max_iter=1000, random_state=42)
    render_scores = cross_val_score(
        log_reg_render, pixel_pca, behavior_labels, cv=5, scoring='accuracy'
    )
    render_behavior_acc = render_scores.mean()
    render_behavior_std = render_scores.std()

    log_reg_physics = LogisticRegressionCV(cv=5, max_iter=1000, random_state=42)
    physics_scores = cross_val_score(
        log_reg_physics, physics_scaled, behavior_labels, cv=5, scoring='accuracy'
    )
    physics_behavior_acc = physics_scores.mean()
    physics_behavior_std = physics_scores.std()

    print(f"  Render → behavior:  {render_behavior_acc:.2%} (±{render_behavior_std:.2%})")
    print(f"  Physics → behavior: {physics_behavior_acc:.2%} (±{physics_behavior_std:.2%})")

    # --- Dissociation summary ---
    print(f"\n  DISSOCIATION:")
    print(f"    Render model:  R² = {mean_r2_render:.4f}  |  Behavior = {render_behavior_acc:.2%}")
    print(f"    Physics model: R² = {mean_r2_physics:.4f}  |  Behavior = {physics_behavior_acc:.2%}")

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

    # Right panel: Behavioral prediction
    bars2 = ax2.bar(labels, [render_behavior_acc, physics_behavior_acc],
                    width=bar_width, color=colors,
                    yerr=[render_behavior_std, physics_behavior_std],
                    capsize=5)
    ax2.axhline(0.5, color='gray', linestyle='--', alpha=0.5, label='Chance')
    ax2.set_ylabel('Behavior prediction accuracy', fontsize=12)
    ax2.set_title('Behavioral sufficiency', fontsize=13)
    ax2.set_ylim(0, 1.1)
    ax2.legend(fontsize=10)
    for bar, val in zip(bars2, [render_behavior_acc, physics_behavior_acc]):
        ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                 f'{val:.1%}', ha='center', va='bottom', fontsize=11, fontweight='bold')

    plt.tight_layout()
    fig_path = f"{fig_dir}/dissociation.png"
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\nFigure saved: {fig_path}")

    return {
        'mean_r2_render': mean_r2_render,
        'mean_r2_physics': mean_r2_physics,
        'render_behavior_acc': render_behavior_acc,
        'physics_behavior_acc': physics_behavior_acc,
    }
