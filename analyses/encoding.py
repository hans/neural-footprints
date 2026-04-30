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
from config import RENDER_PCA_DIM as _CFG_RENDER_PCA_DIM


def pca_reduce_render(render_data, n_components, random_state=42):
    """StandardScaler + PCA on render data (RGBA+depth+seg). Returns (render_pca, pca, scaler)."""
    scaler = StandardScaler()
    render_scaled = scaler.fit_transform(render_data)
    pca = PCA(n_components=n_components, random_state=random_state)
    render_pca = pca.fit_transform(render_scaled)
    return render_pca, pca, scaler


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
                          *, render_pca_dim=None, inferred_physics=None):
    """
    Run encoding model analysis.

    1. PCA-reduce render slice to RENDER_PCA_DIM components
    2. Ridge regression: neural ~ render_PCA -> R² per neuron
    3. Ridge regression: neural ~ render_PCA + physics_labels -> R²
    4. DeltaR² should be tiny
    5. Subsampling curve: vary neurons sampled, plot DeltaR² + significance
    6. Control: MLP physics_labels -> behavior_label

    If `inferred_physics` is provided (output of the predictive-processing
    InverseModel, [n_scenes × physics_dim]), three extra Ridge fits are run:
      - neural ~ inferred_physics              -> r2_inferred
      - neural ~ render_PCA + inferred_physics -> r2_inferred_combined
    The resulting `delta_r2_inferred = combined - pixels_only` is the same
    quantity the evaluation pipeline already gates on.
    """
    if render_pca_dim is None:
        render_pca_dim = _CFG_RENDER_PCA_DIM

    print("\n" + "=" * 60)
    print("SIMULATION 1: Encoding Model False Negatives")
    print("=" * 60)

    program_states = scenes['program_states']
    physics_labels = scenes['physics_labels']
    behavior_labels = scenes['behavior_labels']
    render_indices = scenes['metadata']['render_indices']

    n_scenes, n_neurons = neural_activity.shape

    # --- Extract and PCA-reduce render slice (RGBA + depth + seg) ---
    print(f"\nExtracting render slice and reducing to {render_pca_dim} PCA components...")
    render_data = program_states[:, render_indices]
    render_pca, pca, render_scaler = pca_reduce_render(render_data, render_pca_dim)
    print(f"  PCA explained variance: {pca.explained_variance_ratio_.sum():.2%}")

    # --- Standardize physics labels ---
    scaler_phys = StandardScaler()
    physics_scaled = scaler_phys.fit_transform(physics_labels)

    # --- Encoding model: render only ---
    print("\nFitting encoding model: neural ~ render_PCA ...")
    r2_pixels_only = ridge_r2_per_neuron(render_pca, neural_activity)

    # --- Encoding model: physics only ---
    print("Fitting encoding model: neural ~ physics_labels ...")
    r2_physics_only = ridge_r2_per_neuron(physics_scaled, neural_activity)

    # --- Encoding model: render + physics labels ---
    print("Fitting encoding model: neural ~ render_PCA + physics_labels ...")
    combined = np.hstack([render_pca, physics_scaled])
    r2_combined = ridge_r2_per_neuron(combined, neural_activity)

    delta_r2 = r2_combined - r2_pixels_only
    mean_r2_pix = r2_pixels_only.mean()
    mean_r2_phys = r2_physics_only.mean()
    mean_r2_comb = r2_combined.mean()
    mean_delta = delta_r2.mean()

    print(f"\n  Mean R² (pixels only):    {mean_r2_pix:.4f}")
    print(f"  Mean R² (physics only):   {mean_r2_phys:.4f}")
    print(f"  Mean R² (pixels+physics): {mean_r2_comb:.4f}")
    print(f"  Mean ΔR²:                 {mean_delta:.6f}")

    # --- Optional: same comparison with PP-inferred physics ---
    r2_inferred = None
    r2_inferred_combined = None
    delta_r2_inferred = None
    inferred_scaled = None
    if inferred_physics is not None:
        print("\nFitting encoding model: neural ~ inferred_physics ...")
        scaler_inf = StandardScaler()
        inferred_scaled = scaler_inf.fit_transform(inferred_physics)
        r2_inferred = ridge_r2_per_neuron(inferred_scaled, neural_activity)
        print("Fitting encoding model: neural ~ render_PCA + inferred_physics ...")
        combined_inf = np.hstack([render_pca, inferred_scaled])
        r2_inferred_combined = ridge_r2_per_neuron(combined_inf, neural_activity)
        delta_r2_inferred = r2_inferred_combined - r2_pixels_only
        print(f"  Mean R² (inferred only):           {r2_inferred.mean():.4f}")
        print(f"  Mean R² (pixels+inferred):         {r2_inferred_combined.mean():.4f}")
        print(f"  Mean ΔR² (inferred):               {delta_r2_inferred.mean():.6f}")

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

    # --- Control (inferred): inferred_physics -> behavior_label ---
    control_acc_inferred = None
    control_acc_inferred_std = None
    if inferred_scaled is not None:
        print("\nControl: MLP inferred_physics -> behavior_label ...")
        mlp_inf = MLPClassifier(hidden_layer_sizes=(64,), max_iter=500,
                                random_state=42, early_stopping=True)
        log_scores_inf = cross_val_score(mlp_inf, inferred_scaled, behavior_labels, cv=5,
                                         scoring='accuracy')
        control_acc_inferred = log_scores_inf.mean()
        control_acc_inferred_std = log_scores_inf.std()
        print(f"  Behavior prediction accuracy (inferred): "
              f"{control_acc_inferred:.2%} (±{control_acc_inferred_std:.2%})")

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
    encoder_ridge.fit(render_pca, neural_activity)

    return {
        'r2_pixels_only': r2_pixels_only,
        'r2_physics_only': r2_physics_only,
        'r2_combined': r2_combined,
        'delta_r2': delta_r2,
        'r2_inferred': r2_inferred,
        'r2_inferred_combined': r2_inferred_combined,
        'delta_r2_inferred': delta_r2_inferred,
        'control_accuracy': control_acc,
        'control_accuracy_std': log_scores.std(),
        'control_accuracy_inferred': control_acc_inferred,
        'control_accuracy_inferred_std': control_acc_inferred_std,
        'subsample_means': subsample_means,
        'subsample_sems': subsample_sems,
        'subsample_neuron_counts': neuron_counts,
        'encoder': {
            'scaler': render_scaler,
            'pca': pca,
            'ridge': encoder_ridge,
            'scaler_phys': scaler_phys,
        },
    }
