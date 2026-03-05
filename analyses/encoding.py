"""
Simulation 1: Encoding model false negatives.

Demonstrates that adding physics labels to a pixel-based encoding model
produces negligible improvement in R², despite physics being causally operative.
"""

import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import RidgeCV, LogisticRegressionCV
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

from config import PIXEL_PCA_DIM


def run_encoding_analysis(neural_activity, scenes, neural_meta, fig_dir="figures"):
    """
    Run encoding model analysis.

    1. PCA-reduce pixel slice to PIXEL_PCA_DIM components
    2. Ridge regression: neural ~ pixel_PCA -> R² per neuron
    3. Ridge regression: neural ~ pixel_PCA + physics_labels -> R²
    4. DeltaR² should be tiny
    5. Subsampling curve: vary neurons sampled, plot DeltaR² + significance
    6. Control: logistic regression physics_labels -> behavior_label
    """
    print("\n" + "=" * 60)
    print("SIMULATION 1: Encoding Model False Negatives")
    print("=" * 60)

    program_states = scenes['program_states']
    physics_labels = scenes['physics_labels']
    behavior_labels = scenes['behavior_labels']
    pixel_indices = scenes['metadata']['pixel_indices']

    n_scenes, n_neurons = neural_activity.shape

    # --- Extract and PCA-reduce pixel slice ---
    print(f"\nExtracting pixel slice and reducing to {PIXEL_PCA_DIM} PCA components...")
    pixel_data = program_states[:, pixel_indices]
    scaler_pix = StandardScaler()
    pixel_scaled = scaler_pix.fit_transform(pixel_data)
    pca = PCA(n_components=PIXEL_PCA_DIM, random_state=42)
    pixel_pca = pca.fit_transform(pixel_scaled)
    print(f"  PCA explained variance: {pca.explained_variance_ratio_.sum():.2%}")

    # --- Standardize physics labels ---
    scaler_phys = StandardScaler()
    physics_scaled = scaler_phys.fit_transform(physics_labels)

    # --- Encoding model: pixels only ---
    print("\nFitting encoding model: neural ~ pixel_PCA ...")
    alphas = np.logspace(-2, 6, 20)
    r2_pixels_only = np.zeros(n_neurons)
    for j in range(n_neurons):
        ridge = RidgeCV(alphas=alphas)
        ridge.fit(pixel_pca, neural_activity[:, j])
        r2_pixels_only[j] = ridge.score(pixel_pca, neural_activity[:, j])

    # --- Encoding model: pixels + physics labels ---
    print("Fitting encoding model: neural ~ pixel_PCA + physics_labels ...")
    combined = np.hstack([pixel_pca, physics_scaled])
    r2_combined = np.zeros(n_neurons)
    for j in range(n_neurons):
        ridge = RidgeCV(alphas=alphas)
        ridge.fit(combined, neural_activity[:, j])
        r2_combined[j] = ridge.score(combined, neural_activity[:, j])

    delta_r2 = r2_combined - r2_pixels_only
    mean_r2_pix = r2_pixels_only.mean()
    mean_r2_comb = r2_combined.mean()
    mean_delta = delta_r2.mean()

    print(f"\n  Mean R² (pixels only):    {mean_r2_pix:.4f}")
    print(f"  Mean R² (pixels+physics): {mean_r2_comb:.4f}")
    print(f"  Mean ΔR²:                 {mean_delta:.6f}")

    # --- Control: physics_labels -> behavior_label ---
    print("\nControl: logistic regression physics_labels -> behavior_label ...")
    log_reg = LogisticRegressionCV(cv=5, max_iter=1000, random_state=42)
    log_scores = cross_val_score(log_reg, physics_scaled, behavior_labels, cv=5,
                                 scoring='accuracy')
    control_acc = log_scores.mean()
    print(f"  Behavior prediction accuracy: {control_acc:.2%} (±{log_scores.std():.2%})")
    print("  (High accuracy proves physics labels are informative)")

    # --- Subsampling curve ---
    print("\nComputing subsampling curve...")
    neuron_counts = [10, 25, 50, 100, 200, 300, 400, n_neurons]
    neuron_counts = [n for n in neuron_counts if n <= n_neurons]
    rng = np.random.default_rng(42)

    subsample_means = []
    subsample_sems = []
    subsample_sig_fracs = []

    for n_sub in neuron_counts:
        # Random subsample of neurons, repeated 20 times
        deltas = []
        for _ in range(20):
            idx = rng.choice(n_neurons, size=n_sub, replace=False)
            sub_delta = delta_r2[idx]
            deltas.append(sub_delta.mean())
        deltas = np.array(deltas)
        subsample_means.append(deltas.mean())
        subsample_sems.append(deltas.std() / np.sqrt(len(deltas)))
        # Fraction of subsamples where mean ΔR² > 0
        subsample_sig_fracs.append((deltas > 0).mean())

    # --- Figure 1: R² bar plot ---
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    ax = axes[0]
    bars = ax.bar(['Pixels only', 'Pixels + Physics'],
                  [mean_r2_pix, mean_r2_comb],
                  yerr=[r2_pixels_only.std() / np.sqrt(n_neurons),
                        r2_combined.std() / np.sqrt(n_neurons)],
                  color=['#4878CF', '#D65F5F'], capsize=5)
    ax.set_ylabel('Mean R²')
    ax.set_title('Encoding Model: R² ± Physics Labels')
    # Annotate ΔR²
    ymax = max(mean_r2_pix, mean_r2_comb) * 1.1
    ax.annotate(f'ΔR² = {mean_delta:.6f}', xy=(0.5, ymax),
                ha='center', fontsize=10, style='italic')

    # --- Figure 2: Subsampling curve ---
    ax = axes[1]
    ax.errorbar(neuron_counts, subsample_means, yerr=subsample_sems,
                marker='o', color='#4878CF', capsize=3)
    ax.axhline(0, color='gray', linestyle='--', alpha=0.5)
    ax.set_xlabel('Number of neurons sampled')
    ax.set_ylabel('Mean ΔR²')
    ax.set_title('ΔR² vs. Neuron Subsampling')

    # --- Figure 3: Control accuracy ---
    ax = axes[2]
    ax.bar(['Physics → Behavior'], [control_acc],
           yerr=[log_scores.std()], color='#6ACC65', capsize=5)
    ax.axhline(0.5, color='gray', linestyle='--', alpha=0.5, label='Chance')
    ax.set_ylabel('Accuracy')
    ax.set_title('Control: Physics Labels Predict Behavior')
    ax.set_ylim(0, 1)
    ax.legend()

    plt.tight_layout()
    fig_path = f"{fig_dir}/encoding_analysis.png"
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\nFigure saved: {fig_path}")

    return {
        'r2_pixels_only': r2_pixels_only,
        'r2_combined': r2_combined,
        'delta_r2': delta_r2,
        'control_accuracy': control_acc,
        'subsample_means': subsample_means,
        'subsample_neuron_counts': neuron_counts,
    }
