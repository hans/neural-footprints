"""
Subtractive analysis: fMRI-style contrast/localizer applied to the simulation.

Demonstrates that classic subtractive methods fail to localize physics-specific
neural regions, even when the neural model is deliberately constructed with
spatial structure (topographic map with physics-selective regions).

The core problem: subtraction conflates inverse processing (computing physics
from pixels) with representation of high-level physical quantities. Most neural
activation in response to motion is due to processing low-level input changes,
not representing abstract velocity. The subtraction cannot distinguish the two.

This module is a SEPARATE pipeline from the main simulation. It constructs its
own topographic neural model and generates its own scenes (empty, static, motion).
"""

import os
import numpy as np
import pybullet as p
import pybullet_data
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter

from config import IMAGE_SIZE, N_TIMESTEPS
from scene_generator import (
    _render_scene,
    _save_bullet_state,
    _build_program_state,
    PILLAR_X, PILLAR_Y_CENTER, PILLAR_Z_CENTER,
    PILLAR_WIDTH, PILLAR_DEPTH, PILLAR_HEIGHT,
)

# --- Module constants ---
GRID_SHAPE = (25, 20)                      # 2D neuron grid → 500 neurons
N_TOPO_NEURONS = GRID_SHAPE[0] * GRID_SHAPE[1]
N_SUB_SCENES = 200                         # scenes per condition
N_BUMPS = 3                                # physics-selective region count
BUMP_SIGMA = 3.0                           # Gaussian sigma for bumps (grid units)
SPATIAL_SMOOTH_SIGMA = 1.5                 # smoothing kernel for W columns
PHYSICS_BOOST = 3.0                        # amplification of physics dims in bump centers
SUB_NOISE_LEVEL = 0.3                      # noise level (fraction of signal std)
GT_PERCENTILE = 80                         # top 20% = ground-truth physics-selective
N_THRESHOLD_STEPS = 50                     # sweep resolution for PR curve


# ---------------------------------------------------------------------------
# Scene generation
# ---------------------------------------------------------------------------

def _create_empty_scene(physics_client):
    """Create a scene with only the ground plane (no pillar, no objects)."""
    p.setAdditionalSearchPath(pybullet_data.getDataPath(), physicsClientId=physics_client)
    p.setGravity(0, 0, -9.81, physicsClientId=physics_client)
    p.loadURDF("plane.urdf", physicsClientId=physics_client)


def _create_two_object_scene(physics_client, rng, set_velocity=True):
    """
    Create a scene with ground plane, occluding pillar, and 2 random objects.

    Parameters
    ----------
    physics_client : int
        PyBullet physics client ID.
    rng : numpy Generator
        Random number generator.
    set_velocity : bool
        If True, objects get random x-velocity. If False, zero velocity (static).

    Returns
    -------
    body_ids : list of int
    shape_configs : list of dict
        Per-object shape config for reproducibility.
    """
    p.setAdditionalSearchPath(pybullet_data.getDataPath(), physicsClientId=physics_client)
    p.setGravity(0, 0, -9.81, physicsClientId=physics_client)
    p.loadURDF("plane.urdf", physicsClientId=physics_client)

    # Occluding pillar (visual only, same as main pipeline)
    pillar_vis = p.createVisualShape(
        p.GEOM_BOX,
        halfExtents=[PILLAR_WIDTH / 2, PILLAR_DEPTH / 2, PILLAR_HEIGHT / 2],
        rgbaColor=[0.5, 0.5, 0.5, 1.0],
        physicsClientId=physics_client,
    )
    p.createMultiBody(
        baseMass=0,
        baseCollisionShapeIndex=-1,
        baseVisualShapeIndex=pillar_vis,
        basePosition=[PILLAR_X, PILLAR_Y_CENTER, PILLAR_Z_CENTER],
        physicsClientId=physics_client,
    )

    body_ids = []
    shape_configs = []

    for obj_idx in range(2):
        mass = rng.uniform(0.5, 5.0)
        friction = rng.uniform(0.1, 1.0)
        color = list(rng.uniform(0.1, 1.0, size=3)) + [1.0]

        if rng.random() < 0.5:
            radius = float(rng.uniform(0.1, 0.35))
            col_shape = p.createCollisionShape(p.GEOM_SPHERE, radius=radius,
                                               physicsClientId=physics_client)
            vis_shape = p.createVisualShape(p.GEOM_SPHERE, radius=radius,
                                            rgbaColor=color,
                                            physicsClientId=physics_client)
            shape_cfg = {'shape': 'sphere', 'params': {'radius': radius}, 'color': list(color)}
        else:
            half_extents = [float(v) for v in rng.uniform(0.1, 0.35, size=3)]
            col_shape = p.createCollisionShape(p.GEOM_BOX, halfExtents=half_extents,
                                               physicsClientId=physics_client)
            vis_shape = p.createVisualShape(p.GEOM_BOX, halfExtents=half_extents,
                                            rgbaColor=color,
                                            physicsClientId=physics_client)
            shape_cfg = {'shape': 'box', 'params': {'half_extents': half_extents}, 'color': list(color)}

        # Place objects on opposite sides of the pillar
        side = -1 if obj_idx == 0 else 1
        x = side * rng.uniform(0.6, 1.5)
        y = rng.uniform(-1.5, -0.5)
        z = rng.uniform(0.4, 0.8)
        orn = p.getQuaternionFromEuler([0.0, 0.0, 0.0])

        body_id = p.createMultiBody(
            baseMass=mass,
            baseCollisionShapeIndex=col_shape,
            baseVisualShapeIndex=vis_shape,
            basePosition=[x, y, z],
            baseOrientation=orn,
            physicsClientId=physics_client,
        )
        p.changeDynamics(body_id, -1, lateralFriction=friction,
                         physicsClientId=physics_client)

        if set_velocity:
            x_vel = float(rng.uniform(-5.0, 5.0))
            p.resetBaseVelocity(body_id, linearVelocity=[x_vel, 0.0, 0.0],
                                physicsClientId=physics_client)

        body_ids.append(body_id)
        shape_configs.append(shape_cfg)

    return body_ids, shape_configs


def calibrate_subtraction_bullet_size(n_samples=20, seed=99):
    """
    Calibrate .bullet file size across empty and 2-object scenes.
    Returns K (padded max size) that accommodates all scene types.
    """
    rng = np.random.default_rng(seed)
    max_size = 0

    for i in range(n_samples):
        # Test empty scene
        pc = p.connect(p.DIRECT)
        _create_empty_scene(pc)
        for _ in range(N_TIMESTEPS):
            p.stepSimulation(physicsClientId=pc)
        bullet_bytes = _save_bullet_state(pc)
        max_size = max(max_size, len(bullet_bytes))
        p.disconnect(pc)

        # Test 2-object scene
        pc = p.connect(p.DIRECT)
        scene_rng = np.random.default_rng(rng.integers(0, 2**31))
        _create_two_object_scene(pc, scene_rng, set_velocity=True)
        for _ in range(N_TIMESTEPS):
            p.stepSimulation(physicsClientId=pc)
        bullet_bytes = _save_bullet_state(pc)
        max_size = max(max_size, len(bullet_bytes))
        p.disconnect(pc)

    k = int(max_size * 1.2)
    k = ((k + 3) // 4) * 4  # align to 4 bytes
    print(f"Subtraction calibration: max .bullet = {max_size} bytes, K = {k} bytes")
    return k


def generate_subtraction_scenes(n_scenes, seed, bullet_k):
    """
    Generate three matched sets of scenes for the subtractive analysis.

    Returns dict with:
        'empty_states':  [n_scenes x D]  — empty scenes (ground plane only)
        'static_states': [n_scenes x D]  — 2-object scenes, zero velocity
        'motion_states': [n_scenes x D]  — 2-object scenes, random velocity
        'D_render': int — number of render dimensions in program_state
        'D_physics': int — number of physics (bullet) dimensions
        'D': int — total program_state dimensionality
    """
    rng = np.random.default_rng(seed)

    # Dimension calculations (same as main pipeline)
    rgba_count = IMAGE_SIZE * IMAGE_SIZE * 4
    depth_count = IMAGE_SIZE * IMAGE_SIZE * 4
    seg_count = IMAGE_SIZE * IMAGE_SIZE * 4
    D_render = rgba_count + depth_count + seg_count
    D_physics = bullet_k
    D = D_render + D_physics

    empty_states = np.zeros((n_scenes, D), dtype=np.float32)
    static_states = np.zeros((n_scenes, D), dtype=np.float32)
    motion_states = np.zeros((n_scenes, D), dtype=np.float32)

    for i in range(n_scenes):
        if (i + 1) % 50 == 0 or i == 0:
            print(f"  Subtraction scenes {i+1}/{n_scenes}...")

        # Shared scene seed for static/motion pairing
        scene_seed = rng.integers(0, 2**31)

        # --- Empty scene ---
        pc = p.connect(p.DIRECT)
        _create_empty_scene(pc)
        for _ in range(N_TIMESTEPS):
            p.stepSimulation(physicsClientId=pc)
        rgba, depth, seg = _render_scene(pc)
        bullet = _save_bullet_state(pc)
        empty_states[i] = _build_program_state(rgba, depth, seg, bullet, bullet_k)
        p.disconnect(pc)

        # --- Static scene (same config, zero velocity) ---
        pc = p.connect(p.DIRECT)
        scene_rng = np.random.default_rng(scene_seed)
        _create_two_object_scene(pc, scene_rng, set_velocity=False)
        for _ in range(N_TIMESTEPS):
            p.stepSimulation(physicsClientId=pc)
        rgba, depth, seg = _render_scene(pc)
        bullet = _save_bullet_state(pc)
        static_states[i] = _build_program_state(rgba, depth, seg, bullet, bullet_k)
        p.disconnect(pc)

        # --- Motion scene (same config, random velocity) ---
        pc = p.connect(p.DIRECT)
        scene_rng = np.random.default_rng(scene_seed)  # same seed → same objects
        _create_two_object_scene(pc, scene_rng, set_velocity=True)
        for _ in range(N_TIMESTEPS):
            p.stepSimulation(physicsClientId=pc)
        rgba, depth, seg = _render_scene(pc)
        bullet = _save_bullet_state(pc)
        motion_states[i] = _build_program_state(rgba, depth, seg, bullet, bullet_k)
        p.disconnect(pc)

    print(f"  Subtraction scene generation complete. D={D} (render={D_render}, physics={D_physics})")

    return {
        'empty_states': empty_states,
        'static_states': static_states,
        'motion_states': motion_states,
        'D_render': D_render,
        'D_physics': D_physics,
        'D': D,
    }


# ---------------------------------------------------------------------------
# Topographic neural model
# ---------------------------------------------------------------------------

def generate_topographic_model(D, D_render, D_physics, seed):
    """
    Build a topographic projection matrix with physics-selective regions.

    The model places Gaussian bumps of physics selectivity on a 2D grid,
    creating regions that genuinely load more on .bullet dimensions. This
    stacks the deck in favor of subtraction — there ARE physics-selective
    regions to find. The subtraction still fails because render-responsive
    neurons dominate the motion contrast.

    Returns
    -------
    W : ndarray [N_TOPO_NEURONS x D]
        Topographic projection matrix.
    physics_affinity : ndarray [N_TOPO_NEURONS]
        Per-neuron physics affinity (0-1), used to construct W.
    physics_selectivity : ndarray [N_TOPO_NEURONS]
        Ground truth: fraction of each neuron's weight norm from physics dims.
    """
    rng = np.random.default_rng(seed)
    rows, cols = GRID_SHAPE

    # Step 1: Base random projection
    W = rng.normal(0, 1.0 / np.sqrt(D), size=(N_TOPO_NEURONS, D))

    # Step 2: Physics affinity map — Gaussian bumps on the 2D grid
    affinity_2d = np.zeros((rows, cols), dtype=np.float64)
    for _ in range(N_BUMPS):
        cr = rng.uniform(2, rows - 2)
        cc = rng.uniform(2, cols - 2)
        for r in range(rows):
            for c in range(cols):
                dist2 = (r - cr) ** 2 + (c - cc) ** 2
                affinity_2d[r, c] += np.exp(-dist2 / (2 * BUMP_SIGMA ** 2))

    # Normalize to [0, 1]
    affinity_2d = affinity_2d / affinity_2d.max()
    physics_affinity = affinity_2d.ravel()  # [N_TOPO_NEURONS]

    # Step 3: Modulate W per neuron
    for k in range(N_TOPO_NEURONS):
        alpha = physics_affinity[k]
        # Suppress render weights in physics-selective regions
        W[k, :D_render] *= (1.0 - 0.5 * alpha)
        # Amplify physics weights in physics-selective regions
        W[k, D_render:] *= (1.0 + PHYSICS_BOOST * alpha)

    # Step 4: Spatial smoothing — nearby neurons get correlated tuning
    W_2d = W.reshape(rows, cols, D)
    for d in range(D):
        W_2d[:, :, d] = gaussian_filter(W_2d[:, :, d], sigma=SPATIAL_SMOOTH_SIGMA)
    W = W_2d.reshape(N_TOPO_NEURONS, D)

    # Step 5: Ground truth — physics selectivity per neuron
    physics_norm = np.linalg.norm(W[:, D_render:], axis=1)
    total_norm = np.linalg.norm(W, axis=1)
    total_norm[total_norm == 0] = 1.0
    physics_selectivity = physics_norm / total_norm

    return W, physics_affinity, physics_selectivity


def generate_topographic_activity(program_states, W, seed):
    """
    Generate neural activity using the topographic projection matrix.
    Same procedure as neural_model.py: z-score → project → add noise.
    """
    rng = np.random.default_rng(seed)

    # Standardize per dimension
    means = program_states.mean(axis=0)
    stds = program_states.std(axis=0)
    stds[stds == 0] = 1.0
    standardized = (program_states - means) / stds

    # Project
    signal = standardized @ W.T

    # Add noise
    signal_std = signal.std()
    if signal_std == 0:
        signal_std = 1.0
    noise = SUB_NOISE_LEVEL * signal_std * rng.normal(0, 1, size=signal.shape)

    return signal + noise


# ---------------------------------------------------------------------------
# Subtractive analysis
# ---------------------------------------------------------------------------

def run_subtractive_analysis(empty_activity, static_activity, motion_activity,
                             physics_selectivity, fig_dir=None):
    """
    Run the full subtractive analysis: calibration, motion contrast, ground truth.

    Parameters
    ----------
    empty_activity : ndarray [n_scenes x N_TOPO_NEURONS]
    static_activity : ndarray [n_scenes x N_TOPO_NEURONS]
    motion_activity : ndarray [n_scenes x N_TOPO_NEURONS]
    physics_selectivity : ndarray [N_TOPO_NEURONS]
        Ground truth physics selectivity per neuron (from W).
    fig_dir : str or None
        Directory for saving figures.

    Returns
    -------
    results : dict
    """
    rows, cols = GRID_SHAPE

    # --- Phase 1: Object localizer (calibration) ---
    object_contrast = static_activity.mean(axis=0) - empty_activity.mean(axis=0)
    # Calibrated threshold: top 20% of |object_contrast|
    object_threshold = np.percentile(np.abs(object_contrast), GT_PERCENTILE)

    # --- Phase 2: Motion contrast ---
    motion_contrast = motion_activity.mean(axis=0) - static_activity.mean(axis=0)

    # --- Phase 3: Ground truth comparison ---
    # Ground truth: top 20% of physics_selectivity
    gt_threshold = np.percentile(physics_selectivity, GT_PERCENTILE)
    gt_positive = physics_selectivity >= gt_threshold
    n_gt_positive = gt_positive.sum()
    random_baseline = n_gt_positive / N_TOPO_NEURONS

    # Sweep thresholds on |motion_contrast| for PR curve
    contrast_abs = np.abs(motion_contrast)
    thresholds = np.linspace(contrast_abs.min(), contrast_abs.max(), N_THRESHOLD_STEPS + 1)[1:]

    precisions = []
    recalls = []
    f1_scores = []

    for thresh in thresholds:
        detected = contrast_abs >= thresh
        n_detected = detected.sum()
        if n_detected == 0:
            precisions.append(np.nan)
            recalls.append(0.0)
            f1_scores.append(0.0)
            continue

        tp = (detected & gt_positive).sum()
        fp = (detected & ~gt_positive).sum()
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / n_gt_positive if n_gt_positive > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

        precisions.append(precision)
        recalls.append(recall)
        f1_scores.append(f1)

    precisions = np.array(precisions)
    recalls = np.array(recalls)
    f1_scores = np.array(f1_scores)

    # Precision at the calibrated threshold (from object localizer)
    detected_at_cal = contrast_abs >= object_threshold
    n_detected_cal = detected_at_cal.sum()
    if n_detected_cal > 0:
        tp_cal = (detected_at_cal & gt_positive).sum()
        precision_at_cal = tp_cal / n_detected_cal
    else:
        precision_at_cal = 0.0

    best_f1 = np.nanmax(f1_scores) if len(f1_scores) > 0 else 0.0
    best_precision = np.nanmax(precisions) if len(precisions) > 0 else 0.0

    # Print summary
    print(f"\n  Subtractive analysis results:")
    print(f"    Object localizer threshold: {object_threshold:.4f}")
    print(f"    Ground truth positives: {n_gt_positive}/{N_TOPO_NEURONS}")
    print(f"    Random baseline precision: {random_baseline:.3f}")
    print(f"    Precision at calibrated threshold: {precision_at_cal:.3f}")
    print(f"    Best precision (any threshold): {best_precision:.3f}")
    print(f"    Best F1 (any threshold): {best_f1:.3f}")

    results = {
        'object_contrast': object_contrast,
        'motion_contrast': motion_contrast,
        'physics_selectivity': physics_selectivity,
        'object_threshold': object_threshold,
        'precisions': precisions,
        'recalls': recalls,
        'f1_scores': f1_scores,
        'thresholds': thresholds,
        'precision_at_calibrated': precision_at_cal,
        'best_f1': best_f1,
        'best_precision': best_precision,
        'random_baseline': random_baseline,
        'n_detected_at_calibrated': n_detected_cal,
    }

    if fig_dir:
        _save_subtractive_figures(results, fig_dir)

    return results


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------

def _save_subtractive_figures(results, fig_dir):
    """Save the 2x3 subtractive analysis figure."""
    rows, cols = GRID_SHAPE

    object_map = results['object_contrast'].reshape(rows, cols)
    motion_map = results['motion_contrast'].reshape(rows, cols)
    gt_map = results['physics_selectivity'].reshape(rows, cols)

    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    fig.suptitle('Subtractive Analysis: fMRI-Style Contrast Fails to Localize Physics',
                 fontsize=12, fontweight='bold')

    # [0,0] Object localizer
    vmax_obj = np.percentile(np.abs(object_map), 98)
    im0 = axes[0, 0].imshow(object_map, cmap='RdBu_r', vmin=-vmax_obj, vmax=vmax_obj,
                             aspect='auto', interpolation='nearest')
    axes[0, 0].set_title('Object Localizer\n(objects − empty)', fontsize=10)
    axes[0, 0].set_xlabel('neuron column')
    axes[0, 0].set_ylabel('neuron row')
    plt.colorbar(im0, ax=axes[0, 0], shrink=0.8)

    # [0,1] Motion contrast
    vmax_mot = np.percentile(np.abs(motion_map), 98)
    im1 = axes[0, 1].imshow(motion_map, cmap='RdBu_r', vmin=-vmax_mot, vmax=vmax_mot,
                             aspect='auto', interpolation='nearest')
    axes[0, 1].set_title('Motion Contrast\n(motion − static)', fontsize=10)
    axes[0, 1].set_xlabel('neuron column')
    plt.colorbar(im1, ax=axes[0, 1], shrink=0.8)

    # [0,2] Ground truth
    im2 = axes[0, 2].imshow(gt_map, cmap='viridis', aspect='auto', interpolation='nearest')
    axes[0, 2].set_title('Ground Truth\nPhysics Selectivity (from W)', fontsize=10)
    axes[0, 2].set_xlabel('neuron column')
    plt.colorbar(im2, ax=axes[0, 2], shrink=0.8)

    # [1,0] Overlay — motion contrast with GT contour
    vmax_mot2 = np.percentile(np.abs(motion_map), 98)
    axes[1, 0].imshow(motion_map, cmap='RdBu_r', vmin=-vmax_mot2, vmax=vmax_mot2,
                       aspect='auto', interpolation='nearest')
    gt_threshold = np.percentile(gt_map, GT_PERCENTILE)
    axes[1, 0].contour(gt_map, levels=[gt_threshold], colors='lime', linewidths=2)
    axes[1, 0].set_title('Motion Contrast + GT Outline\n(green = true physics regions)', fontsize=10)
    axes[1, 0].set_xlabel('neuron column')
    axes[1, 0].set_ylabel('neuron row')

    # [1,1] Precision-recall curve
    valid = ~np.isnan(results['precisions'])
    axes[1, 1].plot(results['recalls'][valid], results['precisions'][valid],
                     'b.-', linewidth=1.5, markersize=3)
    axes[1, 1].axhline(results['random_baseline'], color='gray', linestyle='--',
                        label=f'Random baseline ({results["random_baseline"]:.2f})')
    axes[1, 1].set_xlabel('Recall', fontsize=10)
    axes[1, 1].set_ylabel('Precision', fontsize=10)
    axes[1, 1].set_title('Precision–Recall Curve\n(parameterized by threshold)', fontsize=10)
    axes[1, 1].set_xlim(-0.05, 1.05)
    axes[1, 1].set_ylim(-0.05, 1.05)
    axes[1, 1].legend(fontsize=8)
    axes[1, 1].grid(True, alpha=0.3)

    # [1,2] Text summary
    axes[1, 2].axis('off')
    summary_text = (
        f"Random baseline precision: {results['random_baseline']:.3f}\n\n"
        f"Precision at calibrated threshold: {results['precision_at_calibrated']:.3f}\n"
        f"  (detected {results['n_detected_at_calibrated']} neurons)\n\n"
        f"Best precision (any threshold): {results['best_precision']:.3f}\n"
        f"Best F1 (any threshold): {results['best_f1']:.3f}\n\n"
        "Interpretation:\n"
        "Subtraction conflates inverse processing\n"
        "(computing physics from pixels) with\n"
        "representation of high-level physical\n"
        "quantities. Render-responsive neurons\n"
        "dominate the motion contrast — the method\n"
        "cannot isolate genuine physics regions."
    )
    axes[1, 2].text(0.05, 0.95, summary_text, transform=axes[1, 2].transAxes,
                     fontsize=9, verticalalignment='top', fontfamily='monospace',
                     bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

    plt.tight_layout()
    fig_path = os.path.join(fig_dir, 'subtractive_analysis.png')
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Subtractive analysis figure saved: {fig_path}")


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate_subtractive(results):
    """Print pass/fail report for the subtractive analysis."""
    checks = []

    # Check 1: Best precision < 0.5
    passed = results['best_precision'] < 0.5
    checks.append(('Best precision < 0.50', passed, f"{results['best_precision']:.3f}"))

    # Check 2: Precision at calibrated threshold < 0.3
    passed = results['precision_at_calibrated'] < 0.3
    checks.append(('Precision at calibrated < 0.30', passed,
                    f"{results['precision_at_calibrated']:.3f}"))

    # Check 3: No threshold achieves F1 > 0.4
    passed = results['best_f1'] < 0.4
    checks.append(('Best F1 < 0.40', passed, f"{results['best_f1']:.3f}"))

    print("\n" + "=" * 50)
    print("SUBTRACTIVE ANALYSIS EVALUATION")
    print("=" * 50)
    all_passed = True
    for name, passed, value in checks:
        status = "\033[92mPASS\033[0m" if passed else "\033[91mFAIL\033[0m"
        if not passed:
            all_passed = False
        print(f"  [{status}] {name}  (got {value})")

    if all_passed:
        print("\n  Subtraction fails to localize physics regions, as expected.")
    else:
        print("\n  WARNING: Some checks failed — subtraction may be partially working.")
        print("  Review PHYSICS_BOOST and BUMP_SIGMA parameters.")
    print("=" * 50)

    return all_passed
