"""
Simulation 3: R² vs. Behavioral Sufficiency Dissociation.

Directly visualizes the disconnect: the feature set that explains most neural
variance (pixels) is useless for predicting behavior, while the feature set
that predicts behavior (physics) adds essentially nothing to the encoding model.

Two behavioral sufficiency objectives are supported (set in config.py):
  "next_frame_pixels": Ridge R² predicting final-frame pixels from t=0 features.
      Physics wins (has velocities + occluded state); pixels fail (no velocity,
      blind to occluded objects).
  "kinetic_energy":    Logistic accuracy on KE binary label.
      Physics wins (~100%, KE is deterministic from mass+velocity);
      pixels at chance (~50%, RGBA carries no velocity signal).
"""

import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegressionCV
from sklearn.model_selection import cross_val_predict, cross_val_score
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from config import BEHAVIORAL_PCA_DIM as _CFG_BEHAVIORAL_PCA_DIM
from config import BEHAVIORAL_OBJECTIVE as _CFG_BEHAVIORAL_OBJECTIVE


# ---------------------------------------------------------------------------
# Foreground-masked R² utilities
# ---------------------------------------------------------------------------

def _foreground_pixel_mask(pixel_array, threshold_quantile=0.75):
    """
    Boolean mask of pixels with above-threshold variance across scenes.
    High-variance pixels are in the dynamic region where the object moves.
    pixel_array: [n_scenes, n_pixels] float
    """
    variances = pixel_array.var(axis=0)
    threshold = np.quantile(variances, threshold_quantile)
    return variances > threshold


def _masked_pixel_r2(y_true, y_pred, mask):
    """
    R² over foreground pixels only.
    y_true, y_pred: [n_scenes, n_pixels]; mask: [n_pixels] boolean.
    """
    y_t = y_true[:, mask]
    y_p = y_pred[:, mask]
    ss_res = np.sum((y_t - y_p) ** 2)
    ss_tot = np.sum((y_t - y_t.mean(axis=0, keepdims=True)) ** 2)
    return float(1.0 - ss_res / ss_tot) if ss_tot > 0 else 1.0


# ---------------------------------------------------------------------------
# Behavioral sufficiency scorers
# ---------------------------------------------------------------------------

def _make_mlp():
    """Two-layer MLP pipeline with internal scaling. Captures nonlinear mappings."""
    return make_pipeline(
        StandardScaler(),
        MLPRegressor(hidden_layer_sizes=(256, 256), activation='relu',
                     max_iter=500, random_state=42,
                     early_stopping=True, validation_fraction=0.1),
    )

def _pixel_prediction_r2(predicted_raw, actual_pca, scaler, pca):
    """R² of raw pixel predictions against ground-truth, measured in PCA space."""
    pred_pca = pca.transform(scaler.transform(predicted_raw))
    ss_res = np.sum((actual_pca - pred_pca) ** 2)
    ss_tot = np.sum((actual_pca - actual_pca.mean(axis=0, keepdims=True)) ** 2)
    return float(1.0 - ss_res / ss_tot) if ss_tot > 0 else 1.0


def _score_next_frame_pixels(pixel_input_pca, target_pixel_pca,
                              scene_configs, initial_physics_labels,
                              scaler_target, pca_target, pillar_grays=None,
                              lightings=None, n_oracle=200, target_raw=None,
                              initial_raw=None):
    """
    Behavioral sufficiency for next-frame pixel prediction.

    The brain sees three frames (t=0, t=PP_EARLY_FRAME, t=PP_LATE_FRAME).
    The behavioral target is the RGBA at t=N_TIMESTEPS — strictly *beyond*
    the brain's view, so the test is genuine extrapolation. Both models are
    scored in the same target PCA space (scaler_target → pca_target), where
    pca_target retains enough components to explain behavioral_pca_dim of the
    total pixel variance (default 90%).

    Pixel model score:  MLP cross-val R² predicting target-frame pixel PCA
                         from the brain's 3-frame pixel PCA.
    Physics model score: Oracle R² — re-simulate each scene from the initial
                         physics state, render at t=N_TIMESTEPS, project into
                         the target PCA. Deterministic → ~1.0.

    If target_raw is provided ([n_scenes, n_pixels] float in original pixel
    space), also computes foreground-masked R² in raw pixel space: variance
    across scenes identifies the dynamic region, and R² is measured only there.

    If initial_raw is also provided ([n_scenes, n_pixels] float), additionally
    computes delta-frame R²: R² on (fourth_frame − first_frame) in raw pixel
    space, i.e. the difference between the behavioral target (t=N_TIMESTEPS)
    and the t=0 render. Static background pixels are zero in the delta,
    removing their inflation of R²; the only residual signal is the
    object-displacement pattern, which a blurry prediction cannot match
    precisely.

    Returns (pixel_score, physics_score, metric_label, chance_line,
             fg_pixel_score, fg_physics_score, delta_pixel_score, delta_physics_score).
    fg_* and delta_* are None when the corresponding raw arrays are not provided.
    """
    import pybullet as _p
    from scene_generator import resimulate_scene, open_render_client

    # Use cross_val_predict so OOF predictions can be inverted to pixel space
    # for the foreground-masked metric (also avoids fitting twice).
    pixel_pred_pca = cross_val_predict(_make_mlp(), pixel_input_pca, target_pixel_pca, cv=5)
    ss_res = np.sum((target_pixel_pca - pixel_pred_pca) ** 2)
    ss_tot = np.sum((target_pixel_pca - target_pixel_pca.mean(axis=0, keepdims=True)) ** 2)
    pixel_r2 = float(1.0 - ss_res / ss_tot) if ss_tot > 0 else 1.0

    n = min(n_oracle, len(scene_configs))
    _pc = open_render_client(use_gui=True)
    try:
        oracle_raw = np.stack([
            resimulate_scene(scene_configs[i], initial_physics_labels[i],
                             pillar_gray=pillar_grays[i] if pillar_grays is not None else 0.5,
                             lighting=lightings[i] if lightings is not None else None,
                             use_gui=True, physics_client=_pc,
                             ).reshape(-1).astype(np.float32)
            for i in range(n)
        ])
    finally:
        _p.disconnect(_pc)
    physics_r2 = _pixel_prediction_r2(oracle_raw, target_pixel_pca[:n],
                                       scaler_target, pca_target)

    fg_pixel_r2 = None
    fg_physics_r2 = None
    pixel_pred_raw = None
    if target_raw is not None:
        mask = _foreground_pixel_mask(target_raw.astype(float))
        pixel_pred_raw = scaler_target.inverse_transform(
            pca_target.inverse_transform(pixel_pred_pca))
        fg_pixel_r2 = _masked_pixel_r2(target_raw.astype(float), pixel_pred_raw, mask)
        fg_physics_r2 = _masked_pixel_r2(
            target_raw[:n].astype(float), oracle_raw.astype(float), mask)

    delta_pixel_r2 = None
    delta_physics_r2 = None
    if target_raw is not None and initial_raw is not None:
        init_f = initial_raw.astype(float)
        delta_target = target_raw.astype(float) - init_f
        delta_pred = pixel_pred_raw - init_f
        delta_oracle = oracle_raw.astype(float) - init_f[:n]
        ss_res = np.sum((delta_target - delta_pred) ** 2)
        ss_tot = np.sum((delta_target - delta_target.mean(axis=0, keepdims=True)) ** 2)
        delta_pixel_r2 = float(1.0 - ss_res / ss_tot) if ss_tot > 0 else 1.0
        ss_res_o = np.sum((delta_target[:n] - delta_oracle) ** 2)
        ss_tot_o = np.sum((delta_target[:n] - delta_target[:n].mean(axis=0, keepdims=True)) ** 2)
        delta_physics_r2 = float(1.0 - ss_res_o / ss_tot_o) if ss_tot_o > 0 else 1.0

    return pixel_r2, physics_r2, "Next-frame pred. R²", None, fg_pixel_r2, fg_physics_r2, delta_pixel_r2, delta_physics_r2


HIRES_VIZ = 256  # render resolution for visualization plots only


def _compute_predicted_frames(
    pixel_input_pca, target_pixel_pca, scaler_target, pca_target,
    initial_renders, target_renders, rgba_bytes,
    scene_configs, initial_physics_labels,
    n_samples=8, pillar_grays=None, lightings=None
):
    """
    Compute pixel model vs. physics model predicted frame images.

    Returns (init_imgs, pixel_imgs, physics_imgs, target_imgs) as uint8 arrays.
    Physics oracle, init, and target are re-rendered at HIRES_VIZ resolution
    with the OpenGL renderer (shadows). Pixel model predictions are bilinearly
    upscaled from their native 64×64 since they are learned outputs.
    `target_imgs` is the ground-truth t=N_TIMESTEPS RGBA.
    """
    import pybullet as _p
    from PIL import Image as _Image
    from config import IMAGE_SIZE
    from scene_generator import resimulate_scene, open_render_client

    n = min(n_samples, len(initial_renders))

    # Pixel model: MLP trained on 3-frame pixel PCA → target RGBA PCA.
    # Bilinearly upscale to HIRES_VIZ for display (it's a learned prediction,
    # not re-renderable at higher resolution).
    pixel_model = _make_mlp()
    pixel_model.fit(pixel_input_pca, target_pixel_pca)
    pixel_pred_pca = pixel_model.predict(pixel_input_pca[:n])

    def pca_to_img_upscaled(pred_pca):
        pred_scaled = pca_target.inverse_transform(pred_pca)
        pred_pixels = scaler_target.inverse_transform(pred_scaled)
        raw = np.clip(pred_pixels, 0, 255).astype(np.uint8).reshape(
            n, IMAGE_SIZE, IMAGE_SIZE, 4)
        upscaled = np.stack([
            np.array(_Image.fromarray(raw[i]).resize(
                (HIRES_VIZ, HIRES_VIZ), _Image.BILINEAR))
            for i in range(n)
        ])
        return upscaled

    pixel_imgs = pca_to_img_upscaled(pixel_pred_pca)

    # Physics oracle, init, and target: re-render at HIRES_VIZ with OpenGL.
    _pc = open_render_client(use_gui=True)
    try:
        init_imgs = np.stack([
            resimulate_scene(scene_configs[j], initial_physics_labels[j],
                             n_timesteps=0,
                             pillar_gray=pillar_grays[j] if pillar_grays is not None else 0.5,
                             lighting=lightings[j] if lightings is not None else None,
                             use_gui=True, physics_client=_pc,
                             render_size=HIRES_VIZ)
            for j in range(n)
        ])
        physics_imgs = np.stack([
            resimulate_scene(scene_configs[j], initial_physics_labels[j],
                             pillar_gray=pillar_grays[j] if pillar_grays is not None else 0.5,
                             lighting=lightings[j] if lightings is not None else None,
                             use_gui=True, physics_client=_pc,
                             render_size=HIRES_VIZ)
            for j in range(n)
        ])
        target_imgs = np.stack([
            resimulate_scene(scene_configs[j], initial_physics_labels[j],
                             pillar_gray=pillar_grays[j] if pillar_grays is not None else 0.5,
                             lighting=lightings[j] if lightings is not None else None,
                             use_gui=True, physics_client=_pc,
                             render_size=HIRES_VIZ)
            for j in range(n)
        ])
    finally:
        _p.disconnect(_pc)

    return init_imgs, pixel_imgs, physics_imgs, target_imgs


def _score_kinetic_energy(pixel_pca, physics_scaled, behavior_labels):
    """
    Logistic regression accuracy predicting KE binary label.

    Returns (pixel_score, physics_score, metric_label, chance_line).
    """
    log_reg = LogisticRegressionCV(cv=5, max_iter=1000, random_state=42)
    pixel_r2 = cross_val_score(log_reg, pixel_pca, behavior_labels, cv=5, scoring='accuracy').mean()
    physics_r2 = cross_val_score(log_reg, physics_scaled, behavior_labels, cv=5, scoring='accuracy').mean()
    return pixel_r2, physics_r2, "Behavior accuracy", 0.5


# ---------------------------------------------------------------------------
# Main analysis
# ---------------------------------------------------------------------------

def run_dissociation_analysis(neural_activity, scenes, neural_meta,
                               encoder, encoding_results,
                               objective=None,
                               *, behavioral_pca_dim=None):
    """
    Compute and plot the R² vs. behavioral sufficiency dissociation.

    Two models:
      Pixel model:   pixel PCA features (PIXEL_PCA_DIM dims)
      Physics model: API physics labels (15*N_OBJECTS dims)

    For each, measure:
      - Neural R² (how well it predicts neural activity)
      - Behavioral sufficiency score (determined by `objective`)

    Parameters
    ----------
    encoder : dict
        Fitted encoder from encoding analysis: {'scaler', 'pca', 'ridge', 'scaler_phys'}.
        Reused for pixel PCA and physics scaling to avoid redundant fitting.
    """
    if objective is None:
        objective = _CFG_BEHAVIORAL_OBJECTIVE
    if behavioral_pca_dim is None:
        behavioral_pca_dim = _CFG_BEHAVIORAL_PCA_DIM

    print("\n" + "=" * 60)
    print("SIMULATION 3: R² vs. Behavioral Sufficiency Dissociation")
    print(f"  Objective: {objective}")
    print("=" * 60)

    program_states = scenes['program_states']
    physics_labels = scenes['physics_labels']
    initial_physics_labels = scenes['initial_physics_labels']
    initial_renders = scenes['initial_renders']
    target_renders = scenes['target_renders']
    behavior_labels = scenes['behavior_labels']
    metadata = scenes['metadata']
    target_pixel_indices = metadata['target_pixel_indices']
    scene_configs = scenes['scene_configs']
    pillar_grays = scenes['pillar_grays']
    lightings = scenes['lightings']
    rgba_bytes = target_pixel_indices.stop - target_pixel_indices.start

    n_scenes, n_neurons = neural_activity.shape

    # --- Reuse neural R² from encoding analysis ---
    print("\nReusing neural R² from encoding analysis...")
    r2_pixel = encoding_results['r2_pixel_only']
    r2_physics = encoding_results['r2_physics_only']
    r2_combined = encoding_results['r2_combined']

    mean_r2_pixel = r2_pixel.mean()
    mean_r2_physics = r2_physics.mean()
    mean_r2_combined = r2_combined.mean()

    print(f"  Pixel model mean R²:          {mean_r2_pixel:.4f}")
    print(f"  Physics model mean R²:        {mean_r2_physics:.4f}")
    print(f"  Pixel+physics model mean R²:  {mean_r2_combined:.4f}")

    # --- Prepare features for behavioral sufficiency scoring ---
    pixel_data = np.concatenate(
        [scenes['initial_renders'], scenes['early_renders'], scenes['late_renders']],
        axis=1,
    )
    pixel_pca = encoder['pca'].transform(encoder['scaler'].transform(pixel_data))
    physics_scaled = encoder['scaler_phys'].transform(physics_labels)

    # 3-frame pixel input for the behavioral pixel model. Uses a
    # behavioral-sized PCA (whitened) so pixel and physics models are
    # scored at comparable input dimensionality, separate from the high-dim
    # encoder PCA used for neural R².
    scaler_pixel_3f = StandardScaler()
    pixel_3f_scaled = scaler_pixel_3f.fit_transform(pixel_data)
    pca_pixel_3f = PCA(n_components=behavioral_pca_dim, whiten=True, random_state=42)
    pixel_input_pca = pca_pixel_3f.fit_transform(pixel_3f_scaled)

    # Behavioral target: RGBA at t=N_TIMESTEPS — strictly outside the brain's
    # three observed frames (the brain cannot trivially memorize this).
    # PCA retains behavioral_pca_dim of the variance (default 0.90), covering
    # high-frequency components where blur is visible while avoiding
    # static-background inflation from near-zero-variance pixels.
    target_rgba = target_renders[:, target_pixel_indices]
    initial_rgba = initial_renders[:, target_pixel_indices]
    scaler_target = StandardScaler()
    target_scaled = scaler_target.fit_transform(target_rgba)
    pca_target = PCA(n_components=behavioral_pca_dim, whiten=True, random_state=42)
    target_pixel_pca = pca_target.fit_transform(target_scaled)

    # --- Behavioral sufficiency ---
    print(f"Computing behavioral sufficiency ({objective})...")

    if objective == "next_frame_pixels":
        pixel_score, physics_score, metric_label, chance, fg_pixel_score, fg_physics_score, \
            delta_pixel_score, delta_physics_score = \
            _score_next_frame_pixels(
                pixel_input_pca, target_pixel_pca,
                scene_configs, initial_physics_labels,
                scaler_target, pca_target, pillar_grays=pillar_grays,
                lightings=lightings,
                target_raw=target_rgba,
                initial_raw=initial_rgba,
            )
        # Combined model: same oracle as physics-only — if you have the
        # physics state you can re-simulate deterministically.
        combined_score = physics_score

    elif objective == "kinetic_energy":
        pixel_score, physics_score, metric_label, chance = _score_kinetic_energy(
            pixel_pca, physics_scaled, behavior_labels
        )
        fg_pixel_score = None
        fg_physics_score = None
        delta_pixel_score = None
        delta_physics_score = None
        combined_features = np.hstack([pixel_pca, physics_scaled])
        log_reg = LogisticRegressionCV(cv=5, max_iter=1000, random_state=42)
        combined_score = cross_val_score(
            log_reg, combined_features, behavior_labels,
            cv=5, scoring='accuracy').mean()

    else:
        raise ValueError(f"Unknown objective: {objective!r}. "
                         "Use 'next_frame_pixels' or 'kinetic_energy'.")

    # --- Compute predicted frame images for visualization ---
    print("Computing predicted frame images...")
    init_imgs, pixel_imgs, physics_imgs, final_imgs = _compute_predicted_frames(
        pixel_input_pca, target_pixel_pca, scaler_target, pca_target,
        initial_renders, target_renders, rgba_bytes,
        scene_configs, initial_physics_labels,
        pillar_grays=pillar_grays, lightings=lightings,
    )

    print(f"  Pixel          → {metric_label}: {pixel_score:.4f}")
    print(f"  Physics        → {metric_label}: {physics_score:.4f}")
    print(f"  Pixel+physics  → {metric_label}: {combined_score:.4f}")
    if fg_pixel_score is not None:
        print(f"  Foreground-masked pixel  R²:  {fg_pixel_score:.4f}")
        print(f"  Foreground-masked physics R²: {fg_physics_score:.4f}")
    if delta_pixel_score is not None:
        print(f"  Delta-frame pixel  R²:        {delta_pixel_score:.4f}")
        print(f"  Delta-frame physics R²:       {delta_physics_score:.4f}")

    print("\n  DISSOCIATION:")
    print(f"    Pixel model:          R² = {mean_r2_pixel:.4f}  |  {metric_label} = {pixel_score:.4f}")
    print(f"    Physics model:        R² = {mean_r2_physics:.4f}  |  {metric_label} = {physics_score:.4f}")
    print(f"    Pixel+physics model:  R² = {mean_r2_combined:.4f}  |  {metric_label} = {combined_score:.4f}")

    return {
        'mean_r2_pixel': mean_r2_pixel,
        'mean_r2_physics': mean_r2_physics,
        'mean_r2_combined': mean_r2_combined,
        'r2_pixel': r2_pixel,
        'r2_physics': r2_physics,
        'r2_combined': r2_combined,
        'pixel_behavioral_score': pixel_score,
        'physics_behavioral_score': physics_score,
        'combined_behavioral_score': combined_score,
        'fg_pixel_behavioral_score': float(fg_pixel_score) if fg_pixel_score is not None else float('nan'),
        'fg_physics_behavioral_score': float(fg_physics_score) if fg_physics_score is not None else float('nan'),
        'delta_pixel_behavioral_score': float(delta_pixel_score) if delta_pixel_score is not None else float('nan'),
        'delta_physics_behavioral_score': float(delta_physics_score) if delta_physics_score is not None else float('nan'),
        'metric_label': metric_label,
        'objective': objective,
        'chance': chance if chance is not None else float('nan'),
        'predicted_init_imgs': init_imgs,
        'predicted_pixel_imgs': pixel_imgs,
        'predicted_physics_imgs': physics_imgs,
        'predicted_final_imgs': final_imgs,
    }
