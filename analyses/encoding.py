"""
Simulation 1: Encoding model false negatives.

Demonstrates that adding physics labels to a pixel-based encoding model
produces negligible improvement in R², despite physics being causally operative.
"""

import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import cross_val_predict, cross_val_score
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from config import PIXEL_PCA_DIM as _CFG_PIXEL_PCA_DIM


def pca_reduce_pixels(pixel_data, n_components, random_state=42):
    """StandardScaler + PCA on raw pixel data. Returns (pixel_pca, pca, scaler)."""
    scaler = StandardScaler()
    pixel_scaled = scaler.fit_transform(pixel_data)
    pca = PCA(n_components=n_components, random_state=random_state)
    pixel_pca = pca.fit_transform(pixel_scaled)
    return pixel_pca, pca, scaler


def ridge_r2_per_neuron(X, neural_activity, alphas=None, cv=5):
    """Cross-validated ridge regression, returns R² array of shape [n_neurons]."""
    if alphas is None:
        alphas = np.logspace(-2, 6, 20)
    ridge = RidgeCV(alphas=alphas, alpha_per_target=True)
    predictions = cross_val_predict(ridge, X, neural_activity, cv=cv)
    ss_res = ((neural_activity - predictions) ** 2).sum(axis=0)
    ss_tot = ((neural_activity - neural_activity.mean(axis=0)) ** 2).sum(axis=0)
    return 1 - ss_res / ss_tot


def run_encoding_analysis(neural_activity, scenes, neural_meta,
                          *, pixel_pca_dim=None):
    """
    Run encoding model analysis.

    1. PCA-reduce pixel slice to PIXEL_PCA_DIM components
    2. Ridge regression: neural ~ pixel_PCA -> R² per neuron
    3. Ridge regression: neural ~ pixel_PCA + physics_labels -> R²
    4. DeltaR² should be tiny
    5. Subsampling curve: vary neurons sampled, plot DeltaR² + significance
    6. Control: MLP physics_labels -> behavior_label
    """
    if pixel_pca_dim is None:
        pixel_pca_dim = _CFG_PIXEL_PCA_DIM

    print("\n" + "=" * 60)
    print("SIMULATION 1: Encoding Model False Negatives")
    print("=" * 60)

    program_states = scenes['program_states']
    physics_labels = scenes['physics_labels']
    behavior_labels = scenes['behavior_labels']
    render_indices = scenes['metadata']['render_indices']

    n_scenes, n_neurons = neural_activity.shape

    # --- Extract and PCA-reduce render slice (RGBA + depth + seg) ---
    print(f"\nExtracting render slice and reducing to {pixel_pca_dim} PCA components...")
    pixel_data = program_states[:, render_indices]
    pixel_pca, pca, pixel_scaler = pca_reduce_pixels(pixel_data, pixel_pca_dim)
    print(f"  PCA explained variance: {pca.explained_variance_ratio_.sum():.2%}")

    # --- Standardize physics labels ---
    scaler_phys = StandardScaler()
    physics_scaled = scaler_phys.fit_transform(physics_labels)

    # --- Encoding model: pixels only ---
    print("\nFitting encoding model: neural ~ pixel_PCA ...")
    r2_pixels_only = ridge_r2_per_neuron(pixel_pca, neural_activity)

    # --- Encoding model: pixels + physics labels ---
    print("Fitting encoding model: neural ~ pixel_PCA + physics_labels ...")
    combined = np.hstack([pixel_pca, physics_scaled])
    r2_combined = ridge_r2_per_neuron(combined, neural_activity)

    delta_r2 = r2_combined - r2_pixels_only
    mean_r2_pix = r2_pixels_only.mean()
    mean_r2_comb = r2_combined.mean()
    mean_delta = delta_r2.mean()

    print(f"\n  Mean R² (pixels only):    {mean_r2_pix:.4f}")
    print(f"  Mean R² (pixels+physics): {mean_r2_comb:.4f}")
    print(f"  Mean ΔR²:                 {mean_delta:.6f}")

    # --- Control: physics_labels -> behavior_label ---
    # MLP because KE = 0.5*m*v² is nonlinear in the physics label features.
    print("\nControl: MLP physics_labels -> behavior_label ...")
    mlp_clf = MLPClassifier(hidden_layer_sizes=(64,), max_iter=500,
                            random_state=42, early_stopping=True)
    log_scores = cross_val_score(mlp_clf, physics_scaled, behavior_labels, cv=5,
                                 scoring='accuracy')
    control_acc = log_scores.mean()
    print(f"  Behavior prediction accuracy: {control_acc:.2%} (±{log_scores.std():.2%})")
    print("  (High accuracy expected: KE label is a deterministic function of physics labels)")

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

    # --- Fit full encoder on all data (for downstream dynamics analysis) ---
    print("\nFitting full encoder for downstream use...")
    alphas = np.logspace(-2, 6, 20)
    encoder_ridge = RidgeCV(alphas=alphas, alpha_per_target=True)
    encoder_ridge.fit(pixel_pca, neural_activity)

    return {
        'r2_pixels_only': r2_pixels_only,
        'r2_combined': r2_combined,
        'delta_r2': delta_r2,
        'control_accuracy': control_acc,
        'control_accuracy_std': log_scores.std(),
        'subsample_means': subsample_means,
        'subsample_sems': subsample_sems,
        'subsample_neuron_counts': neuron_counts,
        'encoder': {
            'scaler': pixel_scaler,
            'pca': pca,
            'ridge': encoder_ridge,
        },
    }
