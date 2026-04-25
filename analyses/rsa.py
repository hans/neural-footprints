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
from config import RSA_SUBSAMPLE as _CFG_RSA_SUBSAMPLE, RENDER_PCA_DIM as _CFG_RENDER_PCA_DIM


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


def run_rsa_analysis(neural_activity, scenes, neural_meta,
                     *, rsa_subsample=None, render_pca_dim=None,
                     inferred_physics=None):
    """
    Run RSA analysis on a subsample of scenes.

    1. Compute RDMs for neural, render, and physics spaces
    2. Spearman correlations: neural<->render (high), neural<->physics (low)
    3. Partial correlation: neural<->physics | render -> near zero

    If `inferred_physics` is provided ([n_scenes × physics_dim] from the
    PP InverseModel), the same correlations are computed against an
    inferred-physics RDM (corr_neural_inferred, partial_neural_inferred).
    """
    if rsa_subsample is None:
        rsa_subsample = _CFG_RSA_SUBSAMPLE
    if render_pca_dim is None:
        render_pca_dim = _CFG_RENDER_PCA_DIM

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
    render_sub = program_states[sub_idx][:, render_indices]
    physics_sub = physics_labels[sub_idx]

    # PCA-reduce render data for tractability
    print(f"\nSubsampled {n_sub} scenes for RSA.")
    print(f"PCA-reducing render data to {render_pca_dim} components...")
    scaler = StandardScaler()
    render_scaled = scaler.fit_transform(render_sub)
    pca = PCA(n_components=min(render_pca_dim, render_scaled.shape[0] - 1), random_state=42)
    render_pca = pca.fit_transform(render_scaled)

    # Standardize physics
    scaler_phys = StandardScaler()
    physics_scaled = scaler_phys.fit_transform(physics_sub)

    # Compute RDMs
    print("Computing RDMs...")
    rdm_neural = _compute_rdm(neural_sub)
    rdm_render = _compute_rdm(render_pca)
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

    corr_neural_inferred = None
    partial_neural_inferred = None
    rdm_inferred = None
    if inferred_physics is not None:
        scaler_inf = StandardScaler()
        inferred_sub = scaler_inf.fit_transform(inferred_physics[sub_idx])
        rdm_inferred = _compute_rdm(inferred_sub)
        rdm_inferred[np.isnan(rdm_inferred)] = 0.0
        corr_neural_inferred, p_ni = spearmanr(rdm_neural, rdm_inferred)
        partial_neural_inferred, p_pni = _partial_spearman(
            rdm_neural, rdm_inferred, rdm_render
        )
        print(f"  Spearman neural<->inferred:        r={corr_neural_inferred:.4f}  (p={p_ni:.2e})")
        print(f"  Partial neural<->inferred | render: r={partial_neural_inferred:.4f}  (p={p_pni:.2e})")

    return {
        'corr_neural_render': corr_neural_render,
        'corr_neural_physics': corr_neural_physics,
        'corr_render_physics': corr_render_physics,
        'partial_neural_physics': partial_corr,
        'corr_neural_inferred': corr_neural_inferred,
        'partial_neural_inferred': partial_neural_inferred,
        'rdm_neural': rdm_neural,
        'rdm_render': rdm_render,
        'rdm_physics': rdm_physics,
        'rdm_inferred': rdm_inferred,
        'n_sub': n_sub,
    }
