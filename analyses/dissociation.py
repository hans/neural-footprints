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
from sklearn.linear_model import LogisticRegressionCV
from sklearn.model_selection import cross_val_score
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from analyses.encoding import ridge_r2_per_neuron
from config import BEHAVIORAL_PCA_DIM as _CFG_BEHAVIORAL_PCA_DIM
from config import BEHAVIORAL_OBJECTIVE as _CFG_BEHAVIORAL_OBJECTIVE


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


def _score_next_frame_pixels(pixel_pca_initial, final_pixel_pca,
                              scene_configs, initial_physics_labels,
                              scaler_pix, pca_final, pillar_grays=None,
                              lightings=None, n_oracle=200):
    """
    Behavioral sufficiency for next-frame pixel prediction.

    Both models are scored in the same PCA space (scaler_pix → pca_final).

    Render model score:  MLP cross-val R² predicting final-frame pixel PCA
                         from initial pixel PCA (missing velocity/occlusion info).
    Physics model score: Oracle R² — re-simulate each scene from initial physics
                         state, project rendered pixels into PCA space, compare.
                         Since physics is deterministic, this approaches 1.0.

    Returns (render_score, physics_score, metric_label, chance_line).
    """
    from scene_generator import resimulate_scene

    # Render model: MLP cross-val R² in PCA space
    render_r2 = cross_val_score(_make_mlp(), pixel_pca_initial, final_pixel_pca,
                                cv=5, scoring='r2').mean()

    # Physics model: oracle re-simulation R² in same PCA space
    n = min(n_oracle, len(scene_configs))
    oracle_raw = np.stack([
        resimulate_scene(scene_configs[i], initial_physics_labels[i],
                         pillar_gray=pillar_grays[i] if pillar_grays is not None else 0.5,
                         lighting=lightings[i] if lightings is not None else None,
                         ).reshape(-1).astype(np.float32)
        for i in range(n)
    ])
    physics_r2 = _pixel_prediction_r2(oracle_raw, final_pixel_pca[:n],
                                       scaler_pix, pca_final)

    return render_r2, physics_r2, "Next-frame pred. R²", None


def _compute_predicted_frames(
    pixel_pca_initial, final_pixel_pca, scaler_pix, pca_final,
    initial_renders, program_states, pixel_indices,
    scene_configs, initial_physics_labels,
    n_samples=8, pillar_grays=None, lightings=None
):
    """
    Compute render model vs. physics model predicted frame images.

    Returns (init_imgs, render_imgs, physics_imgs, final_imgs) as uint8 arrays.
    """
    from config import IMAGE_SIZE
    from scene_generator import resimulate_scene

    n = min(n_samples, len(initial_renders))

    # Render model: MLP trained on initial pixel PCA → final pixel PCA
    render_model = _make_mlp()
    render_model.fit(pixel_pca_initial, final_pixel_pca)
    render_pred_pca = render_model.predict(pixel_pca_initial[:n])

    def pca_to_img(pred_pca):
        pred_scaled = pca_final.inverse_transform(pred_pca)
        pred_pixels = scaler_pix.inverse_transform(pred_scaled)
        return np.clip(pred_pixels, 0, 255).astype(np.uint8).reshape(n, IMAGE_SIZE, IMAGE_SIZE, 4)

    render_imgs = pca_to_img(render_pred_pca)

    # Physics model: oracle re-simulation from stored initial state
    physics_imgs = np.stack([
        resimulate_scene(scene_configs[j], initial_physics_labels[j],
                         pillar_gray=pillar_grays[j] if pillar_grays is not None else 0.5,
                         lighting=lightings[j] if lightings is not None else None)
        for j in range(n)
    ])

    init_imgs = initial_renders[:n].astype(np.uint8).reshape(n, IMAGE_SIZE, IMAGE_SIZE, 4)
    final_imgs = program_states[:n, pixel_indices].astype(np.uint8).reshape(n, IMAGE_SIZE, IMAGE_SIZE, 4)

    return init_imgs, render_imgs, physics_imgs, final_imgs


def _score_kinetic_energy(pixel_pca, physics_scaled, behavior_labels):
    """
    Logistic regression accuracy predicting KE binary label.

    Returns (render_score, physics_score, metric_label, chance_line).
    """
    log_reg = LogisticRegressionCV(cv=5, max_iter=1000, random_state=42)
    render_r2 = cross_val_score(log_reg, pixel_pca, behavior_labels, cv=5, scoring='accuracy').mean()
    physics_r2 = cross_val_score(log_reg, physics_scaled, behavior_labels, cv=5, scoring='accuracy').mean()
    return render_r2, physics_r2, "Behavior accuracy", 0.5


# ---------------------------------------------------------------------------
# Main analysis
# ---------------------------------------------------------------------------

def run_dissociation_analysis(neural_activity, scenes, neural_meta,
                               encoder,
                               objective=None,
                               *, behavioral_pca_dim=None):
    """
    Compute and plot the R² vs. behavioral sufficiency dissociation.

    Two models:
      Render model:  pixel PCA features (PIXEL_PCA_DIM dims)
      Physics model: API physics labels (15*N_OBJECTS dims)

    For each, measure:
      - Neural R² (how well it predicts neural activity)
      - Behavioral sufficiency score (determined by `objective`)

    Parameters
    ----------
    encoder : dict
        Fitted encoder from encoding analysis: {'scaler', 'pca', 'ridge', 'scaler_phys'}.
        Reused for render PCA and physics scaling to avoid redundant fitting.
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
    behavior_labels = scenes['behavior_labels']
    render_indices = scenes['metadata']['render_indices']
    pixel_indices = scenes['metadata']['pixel_indices']
    scene_configs = scenes['scene_configs']
    pillar_grays = scenes['pillar_grays']
    lightings = scenes['lightings']

    n_scenes, n_neurons = neural_activity.shape

    # --- Prepare final-frame render features (reuse encoder's scaler + PCA) ---
    print("\nPreparing render features (reusing encoder PCA)...")
    pixel_data = program_states[:, render_indices]
    pixel_pca = encoder['pca'].transform(encoder['scaler'].transform(pixel_data))

    physics_scaled = encoder['scaler_phys'].transform(physics_labels)

    # --- Neural R² (cross-validated, consistent with encoding analysis) ---
    print("Computing cross-validated neural R² for each model...")
    r2_render = ridge_r2_per_neuron(pixel_pca, neural_activity)
    r2_physics = ridge_r2_per_neuron(physics_scaled, neural_activity)

    mean_r2_render = r2_render.mean()
    mean_r2_physics = r2_physics.mean()

    print(f"  Render model mean R²:  {mean_r2_render:.4f}")
    print(f"  Physics model mean R²: {mean_r2_physics:.4f}")

    # --- RGBA-only pixel PCA for behavioral sufficiency (matches resimulate output) ---
    rgba_data = program_states[:, pixel_indices]
    scaler_rgba = StandardScaler()
    rgba_scaled = scaler_rgba.fit_transform(rgba_data)

    scaler_init_pix = StandardScaler()
    init_pix_scaled = scaler_init_pix.fit_transform(initial_renders)
    pca_init = PCA(n_components=behavioral_pca_dim, whiten=True, random_state=42)
    pixel_pca_initial = pca_init.fit_transform(init_pix_scaled)
    pca_final = PCA(n_components=behavioral_pca_dim, whiten=True, random_state=42)
    final_pixel_pca = pca_final.fit_transform(rgba_scaled)

    # --- Behavioral sufficiency ---
    print(f"Computing behavioral sufficiency ({objective})...")

    if objective == "next_frame_pixels":
        render_score, physics_score, metric_label, chance = _score_next_frame_pixels(
            pixel_pca_initial, final_pixel_pca,
            scene_configs, initial_physics_labels,
            scaler_rgba, pca_final, pillar_grays=pillar_grays,
            lightings=lightings,
        )

    elif objective == "kinetic_energy":
        render_score, physics_score, metric_label, chance = _score_kinetic_energy(
            pixel_pca, physics_scaled, behavior_labels
        )

    else:
        raise ValueError(f"Unknown objective: {objective!r}. "
                         "Use 'next_frame_pixels' or 'kinetic_energy'.")

    # --- Compute predicted frame images for visualization ---
    print("Computing predicted frame images...")
    init_imgs, render_imgs, physics_imgs, final_imgs = _compute_predicted_frames(
        pixel_pca_initial, final_pixel_pca, scaler_rgba, pca_final,
        initial_renders, program_states, pixel_indices,
        scene_configs, initial_physics_labels,
        pillar_grays=pillar_grays, lightings=lightings,
    )

    print(f"  Render  → {metric_label}: {render_score:.4f}")
    print(f"  Physics → {metric_label}: {physics_score:.4f}")

    print("\n  DISSOCIATION:")
    print(f"    Render model:  R² = {mean_r2_render:.4f}  |  {metric_label} = {render_score:.4f}")
    print(f"    Physics model: R² = {mean_r2_physics:.4f}  |  {metric_label} = {physics_score:.4f}")

    return {
        'mean_r2_render': mean_r2_render,
        'mean_r2_physics': mean_r2_physics,
        'r2_render': r2_render,
        'r2_physics': r2_physics,
        'render_behavioral_score': render_score,
        'physics_behavioral_score': physics_score,
        'metric_label': metric_label,
        'objective': objective,
        'chance': chance if chance is not None else float('nan'),
        'predicted_init_imgs': init_imgs,
        'predicted_render_imgs': render_imgs,
        'predicted_physics_imgs': physics_imgs,
        'predicted_final_imgs': final_imgs,
    }
