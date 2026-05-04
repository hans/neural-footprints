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
from sklearn.model_selection import cross_val_score
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from config import BEHAVIORAL_PCA_DIM as _CFG_BEHAVIORAL_PCA_DIM
from config import BEHAVIORAL_OBJECTIVE as _CFG_BEHAVIORAL_OBJECTIVE
from scene_generator import extract_brain_pixels


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
    """R² of raw pixel predictions against ground-truth, measured in PCA space.

    All pixel-level comparisons must go through this helper so that both models
    are scored in the same feature space.
    """
    pred_pca = pca.transform(scaler.transform(predicted_raw))
    ss_res = np.sum((actual_pca - pred_pca) ** 2)
    ss_tot = np.sum((actual_pca - actual_pca.mean(axis=0, keepdims=True)) ** 2)
    return float(1.0 - ss_res / ss_tot) if ss_tot > 0 else 1.0


def _score_next_frame_pixels(pixel_input_pca, target_pixel_pca,
                              scene_configs, initial_physics_labels,
                              scaler_target, pca_target, pillar_grays=None,
                              lightings=None, n_oracle=200):
    """
    Behavioral sufficiency for next-frame pixel prediction.

    The brain sees three frames (t=0, t=PP_EARLY_FRAME, t=PP_LATE_FRAME).
    The behavioral target is the RGBA at t=N_TIMESTEPS — strictly *beyond*
    the brain's view, so the test is genuine extrapolation. Both models
    are scored in the same target PCA space (scaler_target → pca_target).

    Pixel model score:  MLP cross-val R² predicting target-frame pixel PCA
                         from the brain's 3-frame pixel PCA. Even with three
                         frames, occlusion + RGBA→world ambiguity prevent a
                         perfect parabola fit out to t=N_TIMESTEPS.
    Physics model score: Oracle R² — re-simulate each scene from the initial
                         physics state, render at t=N_TIMESTEPS, project into
                         the target PCA. Deterministic, so approaches 1.0.

    Returns (pixel_score, physics_score, metric_label, chance_line).
    """
    from scene_generator import resimulate_scene

    # Pixel model: MLP cross-val R² in target PCA space
    pixel_r2 = cross_val_score(_make_mlp(), pixel_input_pca, target_pixel_pca,
                                cv=5, scoring='r2').mean()

    # Physics model: oracle re-simulation R² in same target PCA space.
    # `resimulate_scene` (return_program_state=False) renders the t=N_TIMESTEPS
    # behavioral target frame, matching how `target_renders` is built upstream.
    n = min(n_oracle, len(scene_configs))
    oracle_raw = np.stack([
        resimulate_scene(scene_configs[i], initial_physics_labels[i],
                         pillar_gray=pillar_grays[i] if pillar_grays is not None else 0.5,
                         lighting=lightings[i] if lightings is not None else None,
                         ).reshape(-1).astype(np.float32)
        for i in range(n)
    ])
    physics_r2 = _pixel_prediction_r2(oracle_raw, target_pixel_pca[:n],
                                       scaler_target, pca_target)

    return pixel_r2, physics_r2, "Next-frame pred. R²", None


def _compute_predicted_frames(
    pixel_input_pca, target_pixel_pca, scaler_target, pca_target,
    initial_renders, target_renders, rgba_bytes,
    scene_configs, initial_physics_labels,
    n_samples=8, pillar_grays=None, lightings=None
):
    """
    Compute pixel model vs. physics model predicted frame images.

    Returns (init_imgs, pixel_imgs, physics_imgs, target_imgs) as uint8 arrays.
    `target_imgs` is the ground-truth t=N_TIMESTEPS RGBA — the behavioral
    prediction target.
    """
    from config import IMAGE_SIZE
    from scene_generator import resimulate_scene

    n = min(n_samples, len(initial_renders))

    # Pixel model: MLP trained on 3-frame pixel PCA → target RGBA PCA
    pixel_model = _make_mlp()
    pixel_model.fit(pixel_input_pca, target_pixel_pca)
    pixel_pred_pca = pixel_model.predict(pixel_input_pca[:n])

    def pca_to_img(pred_pca):
        pred_scaled = pca_target.inverse_transform(pred_pca)
        pred_pixels = scaler_target.inverse_transform(pred_scaled)
        return np.clip(pred_pixels, 0, 255).astype(np.uint8).reshape(n, IMAGE_SIZE, IMAGE_SIZE, 4)

    pixel_imgs = pca_to_img(pixel_pred_pca)

    # Physics oracle: re-simulate to t=N_TIMESTEPS, return target RGBA frame
    physics_imgs = np.stack([
        resimulate_scene(scene_configs[j], initial_physics_labels[j],
                         pillar_gray=pillar_grays[j] if pillar_grays is not None else 0.5,
                         lighting=lightings[j] if lightings is not None else None)
        for j in range(n)
    ])

    # initial_renders / target_renders hold full RGBA+depth+seg per frame;
    # slice the leading RGBA bytes for visualization.
    init_imgs = initial_renders[:n, :rgba_bytes].astype(np.uint8).reshape(
        n, IMAGE_SIZE, IMAGE_SIZE, 4)
    target_imgs = target_renders[:n, :rgba_bytes].astype(np.uint8).reshape(
        n, IMAGE_SIZE, IMAGE_SIZE, 4)

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
    pixel_data = extract_brain_pixels(program_states, metadata)
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
    target_rgba = target_renders[:, target_pixel_indices]
    scaler_target = StandardScaler()
    target_scaled = scaler_target.fit_transform(target_rgba)
    pca_target = PCA(n_components=behavioral_pca_dim, whiten=True, random_state=42)
    target_pixel_pca = pca_target.fit_transform(target_scaled)

    # --- Behavioral sufficiency ---
    print(f"Computing behavioral sufficiency ({objective})...")

    if objective == "next_frame_pixels":
        pixel_score, physics_score, metric_label, chance = _score_next_frame_pixels(
            pixel_input_pca, target_pixel_pca,
            scene_configs, initial_physics_labels,
            scaler_target, pca_target, pillar_grays=pillar_grays,
            lightings=lightings,
        )
        # Combined model: same oracle as physics-only — if you have the
        # physics state you can re-simulate deterministically.
        combined_score = physics_score

    elif objective == "kinetic_energy":
        pixel_score, physics_score, metric_label, chance = _score_kinetic_energy(
            pixel_pca, physics_scaled, behavior_labels
        )
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
        'metric_label': metric_label,
        'objective': objective,
        'chance': chance if chance is not None else float('nan'),
        'predicted_init_imgs': init_imgs,
        'predicted_pixel_imgs': pixel_imgs,
        'predicted_physics_imgs': physics_imgs,
        'predicted_final_imgs': final_imgs,
    }
