"""
Simulation 2: RSA dominated by pixel structure.

Shows that neural RDM tracks the pixel RDM (high correlation)
but not the physics RDM, and partial correlation removes any residual physics signal.
The sensory RDM is built from 3-frame brain pixels — the same data the
encoding/residual/dynamics analyses see.
"""

import numpy as np
from scipy.spatial.distance import pdist, squareform
from scipy.stats import spearmanr
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from config import (
    RSA_SUBSAMPLE as _CFG_RSA_SUBSAMPLE,
    PIXEL_PCA_DIM as _CFG_PIXEL_PCA_DIM,
)
from scene_generator import extract_brain_pixels


def _compute_rdm(data):
    """Compute representational dissimilarity matrix using correlation distance."""
    return pdist(data, metric="correlation")


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


def run_rsa_analysis(
    neural_activity,
    scenes,
    neural_meta,
    *,
    rsa_subsample=None,
    pixel_pca_dim=None,
    predicted_pixel_pca=None,
):
    """
    Run RSA analysis on a subsample of scenes.

    1. Compute RDMs for neural, pixel, and physics spaces
    2. Spearman correlations: neural<->pixel (high), neural<->physics (low)
    3. Partial correlation: neural<->physics | pixel -> near zero
    """
    if rsa_subsample is None:
        rsa_subsample = _CFG_RSA_SUBSAMPLE
    if pixel_pca_dim is None:
        pixel_pca_dim = _CFG_PIXEL_PCA_DIM

    print("\n" + "=" * 60)
    print("SIMULATION 2: RSA Dominated by Pixel Structure")
    print("=" * 60)

    program_states = scenes["program_states"]
    physics_labels = scenes["physics_labels"]
    metadata = scenes["metadata"]

    n_scenes = program_states.shape[0]
    n_sub = min(rsa_subsample, n_scenes)

    # Subsample scenes for tractability
    rng = np.random.default_rng(123)
    sub_idx = rng.choice(n_scenes, size=n_sub, replace=False)
    sub_idx.sort()

    neural_sub = neural_activity[sub_idx]
    pixel_sub = extract_brain_pixels(program_states[sub_idx], metadata)
    physics_sub = physics_labels[sub_idx]
    predicted_sub = (
        predicted_pixel_pca[sub_idx] if predicted_pixel_pca is not None else None
    )

    # PCA-reduce pixel data for tractability
    print(f"\nSubsampled {n_sub} scenes for RSA.")
    print(f"PCA-reducing pixel data to {pixel_pca_dim} components...")
    scaler = StandardScaler()
    pixel_scaled = scaler.fit_transform(pixel_sub)
    pca = PCA(
        n_components=min(pixel_pca_dim, pixel_scaled.shape[0] - 1), random_state=42
    )
    pixel_pca = pca.fit_transform(pixel_scaled)

    # Standardize physics
    scaler_phys = StandardScaler()
    physics_scaled = scaler_phys.fit_transform(physics_sub)

    # Compute RDMs
    print("Computing RDMs...")
    rdm_neural = _compute_rdm(neural_sub)
    rdm_pixel = _compute_rdm(pixel_pca)
    rdm_physics = _compute_rdm(physics_scaled)

    # Handle NaN in RDMs (constant rows produce NaN in correlation distance)
    for rdm in [rdm_neural, rdm_pixel, rdm_physics]:
        rdm[np.isnan(rdm)] = 0.0

    # Predicted-S RDM (forward-model render)
    rdm_predicted = None
    corr_neural_predicted = None
    if predicted_sub is not None:
        scaler_pred = StandardScaler()
        predicted_scaled = scaler_pred.fit_transform(predicted_sub)
        pca_pred = PCA(
            n_components=min(pixel_pca_dim, predicted_scaled.shape[0] - 1),
            random_state=42,
        )
        predicted_pca_sub = pca_pred.fit_transform(predicted_scaled)
        rdm_predicted = _compute_rdm(predicted_pca_sub)
        rdm_predicted[np.isnan(rdm_predicted)] = 0.0
        corr_neural_predicted, _ = spearmanr(rdm_neural, rdm_predicted)

    # Spearman correlations
    corr_neural_pixel, p_nr = spearmanr(rdm_neural, rdm_pixel)
    corr_neural_physics, p_np = spearmanr(rdm_neural, rdm_physics)
    corr_pixel_physics, p_rp = spearmanr(rdm_pixel, rdm_physics)

    print(f"\n  Spearman neural<->pixel:   r={corr_neural_pixel:.4f}  (p={p_nr:.2e})")
    print(f"  Spearman neural<->physics: r={corr_neural_physics:.4f}  (p={p_np:.2e})")
    print(f"  Spearman pixel<->physics:  r={corr_pixel_physics:.4f}  (p={p_rp:.2e})")
    if corr_neural_predicted is not None:
        print(f"  Spearman neural<->predicted_S: r={corr_neural_predicted:.4f}")

    # Partial correlation: neural<->physics | pixel
    partial_corr, partial_p = _partial_spearman(rdm_neural, rdm_physics, rdm_pixel)
    print(
        f"  Partial neural<->physics | pixel: r={partial_corr:.4f}  (p={partial_p:.2e})"
    )

    result = {
        "corr_neural_pixel": corr_neural_pixel,
        "corr_neural_physics": corr_neural_physics,
        "corr_pixel_physics": corr_pixel_physics,
        "partial_neural_physics": partial_corr,
        "rdm_neural": rdm_neural,
        "rdm_pixel": rdm_pixel,
        "rdm_physics": rdm_physics,
        "n_sub": n_sub,
    }
    if rdm_predicted is not None:
        result["rdm_predicted"] = rdm_predicted
        result["corr_neural_predicted"] = corr_neural_predicted
    return result
