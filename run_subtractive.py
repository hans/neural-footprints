"""
Standalone runner for the subtractive analysis.

Demonstrates that fMRI-style subtractive contrasts fail to localize
physics-specific neural regions, even with a topographic neural model
that has genuine spatial structure and physics-selective regions.

Run:  python run_subtractive.py
"""

import os
import time

from analyses.subtractive import (
    calibrate_subtraction_bullet_size,
    generate_subtraction_scenes,
    generate_topographic_model,
    generate_topographic_activity,
    run_subtractive_analysis,
    evaluate_subtractive,
    N_SUB_SCENES,
)

SEED = 42


def main():
    t0 = time.time()

    fig_dir = os.path.join(os.path.dirname(__file__), "figures")
    os.makedirs(fig_dir, exist_ok=True)

    print("=" * 60)
    print("SUBTRACTIVE ANALYSIS — fMRI-Style Contrast")
    print("=" * 60)

    # Step 1: Calibrate bullet size for empty + 2-object scenes
    print("\n[1/5] Calibrating .bullet file size...")
    bullet_k = calibrate_subtraction_bullet_size()

    # Step 2: Generate scenes (empty, static, motion)
    print(f"\n[2/5] Generating {N_SUB_SCENES} scenes per condition...")
    scenes = generate_subtraction_scenes(N_SUB_SCENES, SEED, bullet_k)

    # Step 3: Build topographic neural model
    print("\n[3/5] Building topographic neural model...")
    W, physics_affinity, physics_selectivity = generate_topographic_model(
        scenes['D'], scenes['D_render'], scenes['D_physics'], seed=SEED
    )
    print(f"  Grid: {W.shape[0]} neurons, D={W.shape[1]}")
    print(f"  Physics selectivity range: [{physics_selectivity.min():.3f}, {physics_selectivity.max():.3f}]")

    # Step 4: Generate neural activations for all 3 conditions
    print("\n[4/5] Generating topographic neural activity...")
    empty_activity = generate_topographic_activity(scenes['empty_states'], W, seed=SEED + 1)
    static_activity = generate_topographic_activity(scenes['static_states'], W, seed=SEED + 2)
    motion_activity = generate_topographic_activity(scenes['motion_states'], W, seed=SEED + 3)
    print(f"  Activity shapes: {empty_activity.shape}, {static_activity.shape}, {motion_activity.shape}")

    # Step 5: Run subtractive analysis
    print("\n[5/5] Running subtractive analysis...")
    results = run_subtractive_analysis(
        empty_activity, static_activity, motion_activity,
        physics_selectivity, fig_dir=fig_dir
    )

    # Evaluation
    evaluate_subtractive(results)

    elapsed = time.time() - t0
    print(f"\nTotal time: {elapsed:.1f}s")


if __name__ == "__main__":
    main()
