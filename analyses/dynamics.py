"""
Simulation 4: Future brain state prediction — reverse dissociation.

Compares two forward models that attempt to predict future neural activity
from t=0 information, scored through the same learned pixel-based encoding
model (fitted on 3-frame brain pixels in the encoding analysis):

  Physics forward model (oracle): resimulate each scene via PyBullet from
    initial physics state → render the 3 brain frames → extract pixels →
    encoder.predict().
    Since PyBullet is deterministic, the encoder sees pixel-perfect input.

  Pixel forward model (learned): MLP predicts the 3-frame pixel vector from
    the initial frame's pixels → encoder.predict().
    Fails because the initial frame lacks velocity and occluded-state info,
    so the predicted later frames diverge from the true ones.

Both models consume the same input the encoder was trained on (3-frame RGBA),
matching what a scientist would actually have access to. This is the reverse
of the encoding analysis: physics (invisible to standard encoding) is exactly
what you need for temporal prediction.
"""

import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import KFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from analyses.dissociation import _make_mlp
from config import DYNAMICS_PCA_DIM as _CFG_DYNAMICS_PCA_DIM
from scene_generator import extract_brain_pixels, extract_frame_pixels


def run_dynamics_analysis(
    neural_activity,
    scenes,
    neural_meta,
    encoding_delta_r2,
    encoder,
    *,
    inferred_physics=None,
    dynamics_pca_dim=None,
):
    """
    Future brain state prediction via physics vs. pixel forward models.

    Both forward models are scored through the same learned encoding model
    (scaler → PCA → ridge) fitted on actual final-frame render data.

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
    encoder : dict
        Fitted encoder from encoding analysis: {'scaler', 'pca', 'ridge'}.
    fig_dir : str
        Directory for output figures.
    dynamics_pca_dim : int
        PCA components for MLP pixel prediction target.
    """
    if dynamics_pca_dim is None:
        dynamics_pca_dim = _CFG_DYNAMICS_PCA_DIM

    print("\n" + "=" * 60)
    print("SIMULATION 4: Future Brain State Prediction")
    print("=" * 60)

    from scene_generator import resimulate_scene

    program_states = scenes["program_states"]
    initial_physics_labels = scenes["initial_physics_labels"]
    initial_renders = scenes["initial_renders"]
    scene_configs = scenes["scene_configs"]
    pillar_grays = scenes["pillar_grays"]
    lightings = scenes["lightings"]
    metadata = scenes["metadata"]
    initial_rgba = extract_frame_pixels(initial_renders, metadata)

    enc_scaler = encoder["scaler"]
    enc_pca = encoder["pca"]
    enc_ridge = encoder["ridge"]

    n_scenes, n_neurons = neural_activity.shape

    # ------------------------------------------------------------------
    # Physics forward model: resimulate → 3-frame brain pixels → encoder
    # ------------------------------------------------------------------
    print("\nPhysics forward model: resimulating scenes from initial state...")
    resim_program_states = np.zeros_like(program_states)
    for i in range(n_scenes):
        if (i + 1) % 100 == 0 or i == 0:
            print(f"  Resimulating scene {i+1}/{n_scenes}...")
        resim_program_states[i] = resimulate_scene(
            scene_configs[i],
            initial_physics_labels[i],
            return_program_state=True,
            pillar_gray=pillar_grays[i],
            lighting=lightings[i],
        )

    pixel_data = extract_brain_pixels(program_states, metadata)
    resim_pixels = extract_brain_pixels(resim_program_states, metadata)

    print("  Cross-validating encoder on resimulated pixels (5-fold)...")
    kf = KFold(n_splits=5, shuffle=False)
    pred_neural_physics = np.zeros_like(neural_activity)

    for fold, (train_idx, test_idx) in enumerate(kf.split(pixel_data), 1):
        pipe = make_pipeline(
            StandardScaler(),
            PCA(n_components=enc_pca.n_components_, random_state=42),
            RidgeCV(alphas=np.logspace(-2, 6, 20), alpha_per_target=True),
        )
        pipe.fit(pixel_data[train_idx], neural_activity[train_idx])
        pred_neural_physics[test_idx] = pipe.predict(resim_pixels[test_idx])

    r2_physics_forward = _r2_per_neuron(pred_neural_physics, neural_activity)
    mean_r2_physics = r2_physics_forward.mean()
    print(f"  Physics forward model mean R² (CV): {mean_r2_physics:.4f}")

    # ------------------------------------------------------------------
    # Inferred-physics forward model (cognitive PP): same resimulation
    # path, driven by the InverseModel's per-scene output instead of GT.
    # Mean-fill of non-observable dims (mass/friction/orientation/ang_vel)
    # is dynamically safe in this scene generator — see InverseModel docstring.
    # ------------------------------------------------------------------
    r2_inferred_forward = None
    mean_r2_inferred = None
    if inferred_physics is not None:
        print("\nInferred-physics forward model: resimulating from PP estimates...")
        resim_inferred_states = np.zeros_like(program_states)
        for i in range(n_scenes):
            if (i + 1) % 100 == 0 or i == 0:
                print(f"  Resimulating scene {i+1}/{n_scenes}...")
            resim_inferred_states[i] = resimulate_scene(
                scene_configs[i],
                inferred_physics[i],
                return_program_state=True,
                pillar_gray=pillar_grays[i],
                lighting=lightings[i],
            )
        resim_inferred_pixels = extract_brain_pixels(resim_inferred_states, metadata)

        print("  Cross-validating encoder on inferred-resimulated pixels (5-fold)...")
        pred_neural_inferred = np.zeros_like(neural_activity)
        for fold, (train_idx, test_idx) in enumerate(kf.split(pixel_data), 1):
            pipe = make_pipeline(
                StandardScaler(),
                PCA(n_components=enc_pca.n_components_, random_state=42),
                RidgeCV(alphas=np.logspace(-2, 6, 20), alpha_per_target=True),
            )
            pipe.fit(pixel_data[train_idx], neural_activity[train_idx])
            pred_neural_inferred[test_idx] = pipe.predict(
                resim_inferred_pixels[test_idx]
            )
        r2_inferred_forward = _r2_per_neuron(pred_neural_inferred, neural_activity)
        mean_r2_inferred = r2_inferred_forward.mean()
        print(f"  Inferred forward model mean R² (CV): {mean_r2_inferred:.4f}")

    # ------------------------------------------------------------------
    # Pixel forward model: MLP predicts 3-frame pixels from initial → encoder
    # ------------------------------------------------------------------
    print("\nPixel forward model: training MLP on initial → 3-frame brain pixels...")

    # PCA for MLP target: 3-frame brain pixels (behavioral dim, whitened —
    # matches dissociation). The encoder consumes raw 3-frame pixels through
    # its own scaler+PCA, so we inverse-transform predictions back to pixel
    # space before feeding it.
    scaler_target = StandardScaler()
    target_scaled = scaler_target.fit_transform(pixel_data)
    pca_target = PCA(n_components=dynamics_pca_dim, whiten=True, random_state=42)
    target_pixel_pca = pca_target.fit_transform(target_scaled)

    scaler_init = StandardScaler()
    init_scaled = scaler_init.fit_transform(initial_rgba)
    pca_init = PCA(n_components=dynamics_pca_dim, whiten=True, random_state=42)
    init_pixel_pca = pca_init.fit_transform(init_scaled)

    # Cross-validated MLP predictions (out-of-fold to avoid double-dipping)
    pred_target_pca = cross_val_predict(
        _make_mlp(), init_pixel_pca, target_pixel_pca, cv=5
    )

    # Inverse-transform back to raw 3-frame pixel space, then through the encoder.
    pred_target_scaled = pca_target.inverse_transform(pred_target_pca)
    pred_target_pixels = scaler_target.inverse_transform(pred_target_scaled)

    pred_pixel_pca = enc_pca.transform(enc_scaler.transform(pred_target_pixels))
    pred_neural_pixel = enc_ridge.predict(pred_pixel_pca)

    r2_pixel_forward = _r2_per_neuron(pred_neural_pixel, neural_activity)
    mean_r2_pixel = r2_pixel_forward.mean()
    print(f"  Pixel forward model mean R²: {mean_r2_pixel:.4f}")

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    gap = mean_r2_physics - mean_r2_pixel
    print(f"\n  FUTURE BRAIN STATE DISSOCIATION:")
    print(f"    Physics forward model R²:   {mean_r2_physics:.4f}")
    if mean_r2_inferred is not None:
        print(f"    Inferred forward model R²:  {mean_r2_inferred:.4f}")
    print(f"    Pixel forward model R²:     {mean_r2_pixel:.4f}")
    print(f"    Gap (physics − pixel):      {gap:.4f}")
    print(f"    (cf. encoding ΔR² for current brain: {encoding_delta_r2:.4f})")

    return {
        "r2_physics_forward": r2_physics_forward,
        "r2_pixel_forward": r2_pixel_forward,
        "r2_inferred_forward": r2_inferred_forward,
        "mean_r2_physics_forward": mean_r2_physics,
        "mean_r2_pixel_forward": mean_r2_pixel,
        "mean_r2_inferred_forward": mean_r2_inferred,
        "encoding_delta_r2": encoding_delta_r2,
        "forward_gap": gap,
    }


def _r2_per_neuron(predicted, actual):
    """Per-neuron R² between predicted and actual neural activity."""
    ss_res = ((actual - predicted) ** 2).sum(axis=0)
    ss_tot = ((actual - actual.mean(axis=0)) ** 2).sum(axis=0)
    r2 = 1.0 - ss_res / ss_tot
    r2[ss_tot == 0] = 0.0
    return r2
