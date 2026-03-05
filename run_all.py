"""
Neural Footprints Simulation — Main runner.

Demonstrates that standard neuroscience analyses (encoding models, RSA)
systematically fail to detect high-level representations even when
those representations are causally operative.
"""

import os
import sys
import time

from config import N_SCENES, RANDOM_SEED, BEHAVIORAL_OBJECTIVE
from scene_generator import calibrate_bullet_size, generate_scenes, save_sample_renders
from neural_model import generate_neural_activity, print_variance_diagnostic
from analyses.encoding import run_encoding_analysis
from analyses.rsa import run_rsa_analysis
from analyses.dissociation import run_dissociation_analysis
from analyses.dynamics import run_dynamics_analysis


def print_summary(encoding_results, rsa_results, dissociation_results):
    """Print a concise summary of all results."""
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    dr2 = encoding_results['delta_r2'].mean()
    ctrl = encoding_results['control_accuracy']
    nr = rsa_results['corr_neural_render']
    np_ = rsa_results['corr_neural_physics']
    partial = rsa_results['partial_neural_physics']
    r2_rend = dissociation_results['mean_r2_render']
    r2_phys = dissociation_results['mean_r2_physics']
    beh_rend = dissociation_results['render_behavioral_score']
    beh_phys = dissociation_results['physics_behavioral_score']
    metric = dissociation_results['metric_label']
    obj = dissociation_results['objective']

    print(f"""
Encoding Model:
  Mean ΔR² (adding physics to pixels): {dr2:.6f}  ← near zero
  Control accuracy (physics → KE label): {ctrl:.2%}  ← high

RSA:
  Neural ↔ Render:             r = {nr:.4f}  ← high
  Neural ↔ Physics:            r = {np_:.4f}  ← low
  Neural ↔ Physics | Render:   r = {partial:.4f}  ← near zero

Dissociation  (objective: {obj}):
  Render model:  R² = {r2_rend:.4f}  |  {metric} = {beh_rend:.4f}
  Physics model: R² = {r2_phys:.4f}  |  {metric} = {beh_phys:.4f}
  The model that explains the brain can't explain the world.
  The model that explains the world can't be found in the brain.
""")


def main():
    t0 = time.time()

    fig_dir = os.path.join(os.path.dirname(__file__), "figures")
    os.makedirs(fig_dir, exist_ok=True)

    print("=" * 60)
    print("NEURAL FOOTPRINTS SIMULATION")
    print("=" * 60)

    # Step 1: Calibrate bullet file size
    print("\n[1/6] Calibrating .bullet file size...")
    bullet_k = calibrate_bullet_size()

    # Step 2: Generate scenes
    print(f"\n[2/6] Generating {N_SCENES} scenes...")
    scenes = generate_scenes(N_SCENES, RANDOM_SEED, bullet_k)
    save_sample_renders(scenes, fig_dir)

    # Step 3: Generate neural activity
    print(f"\n[3/6] Generating neural activity...")
    neural, neural_meta = generate_neural_activity(scenes['program_states'], RANDOM_SEED)
    print_variance_diagnostic(scenes['metadata'], neural_meta)

    # Step 4: Encoding analysis
    print(f"\n[4/6] Running encoding analysis...")
    encoding_results = run_encoding_analysis(neural, scenes, neural_meta, fig_dir=fig_dir)

    # Step 5: RSA analysis
    print(f"\n[5/6] Running RSA analysis...")
    rsa_results = run_rsa_analysis(neural, scenes, neural_meta, fig_dir=fig_dir)

    # Step 6: Dissociation analysis
    print(f"\n[6/6] Running dissociation analysis (objective: {BEHAVIORAL_OBJECTIVE})...")
    dissociation_results = run_dissociation_analysis(
        neural, scenes, neural_meta, objective=BEHAVIORAL_OBJECTIVE, fig_dir=fig_dir
    )

    # Dynamics stub
    run_dynamics_analysis()

    # Summary
    print_summary(encoding_results, rsa_results, dissociation_results)

    elapsed = time.time() - t0
    print(f"Total time: {elapsed:.1f}s")


if __name__ == "__main__":
    main()
