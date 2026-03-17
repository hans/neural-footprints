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
from analyses.predictive_processing import run_predictive_processing_analysis


def print_summary(encoding_results, rsa_results, dissociation_results, pp_results=None):
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

    enc_inferred_line = ""
    if encoding_results.get('r2_inferred') is not None:
        r2_inf = encoding_results['r2_inferred'].mean()
        dr2_inf = encoding_results['delta_r2_inferred'].mean()
        enc_inferred_line = (
            f"  Mean R² (inferred physics only):   {r2_inf:.4f}\n"
            f"  Mean ΔR² (adding inferred to pixels): {dr2_inf:.6f}  ← near zero"
        )

    rsa_inferred_line = ""
    if rsa_results.get('corr_neural_inferred') is not None:
        ni = rsa_results['corr_neural_inferred']
        partial_ni = rsa_results['partial_neural_inferred']
        rsa_inferred_line = (
            f"  Neural ↔ Inferred physics:        r = {ni:.4f}  ← low\n"
            f"  Neural ↔ Inferred | Render:       r = {partial_ni:.4f}  ← near zero"
        )

    print(f"""
Encoding Model:
  Mean ΔR² (adding physics to pixels): {dr2:.6f}  ← near zero
  Control accuracy (physics → KE label): {ctrl:.2%}  ← high
{enc_inferred_line}

RSA:
  Neural ↔ Render:             r = {nr:.4f}  ← high
  Neural ↔ Physics:            r = {np_:.4f}  ← low
  Neural ↔ Physics | Render:   r = {partial:.4f}  ← near zero
{rsa_inferred_line}

Dissociation  (objective: {obj}):
  Render model:  R² = {r2_rend:.4f}  |  {metric} = {beh_rend:.4f}
  Physics model: R² = {r2_phys:.4f}  |  {metric} = {beh_phys:.4f}
  The model that explains the brain can't explain the world.
  The model that explains the world can't be found in the brain.
""")
    if pp_results is not None:
        print(f"""Predictive Processing:
  Prior MLP R² (phys→pix, ceiling):   {pp_results['prior_r2']:.4f}
  Oracle R² (true phys→PyBullet):     {pp_results['oracle_r2']:.4f}
  PP chain R² (inferred phys→PyBullet): {pp_results['pp_r2']:.4f}
  Render-only R² (pix→pix MLP):       {pp_results['render_r2']:.4f}
  InverseModel mean R²:               {pp_results['inverse_mean_r2']:.4f}
  Neural R² — t=0 pixel PCA:          {pp_results['neural_r2_t0']:.4f}
  Neural R² — inferred physics:       {pp_results['neural_r2_inferred_physics']:.4f}  ← should be low
  Physics intermediate is functionally operative, methodologically invisible.
""")


def main():
    t0 = time.time()

    fig_dir = os.path.join(os.path.dirname(__file__), "figures")
    os.makedirs(fig_dir, exist_ok=True)

    print("=" * 60)
    print("NEURAL FOOTPRINTS SIMULATION")
    print("=" * 60)

    # Step 1: Calibrate bullet file size
    print("\n[1/7] Calibrating .bullet file size...")
    bullet_k = calibrate_bullet_size()

    # Step 2: Generate scenes
    print(f"\n[2/7] Generating {N_SCENES} scenes...")
    scenes = generate_scenes(N_SCENES, RANDOM_SEED, bullet_k)
    save_sample_renders(scenes, fig_dir)

    # Step 3: Generate neural activity
    print(f"\n[3/7] Generating neural activity...")
    neural, neural_meta = generate_neural_activity(scenes['program_states'], RANDOM_SEED)
    print_variance_diagnostic(scenes['metadata'], neural_meta)

    # Step 4: Predictive processing analysis (must run before encoding/RSA to get inferred latents)
    print("\n[4/7] Running predictive processing analysis...")
    pp_results = run_predictive_processing_analysis(neural, scenes, neural_meta, fig_dir=fig_dir)
    inferred_physics = pp_results['inferred_physics_all']

    # Step 5: Encoding analysis
    print("\n[5/7] Running encoding analysis...")
    encoding_results = run_encoding_analysis(
        neural, scenes, neural_meta, inferred_physics=inferred_physics, fig_dir=fig_dir
    )

    # Step 6: RSA analysis
    print("\n[6/7] Running RSA analysis...")
    rsa_results = run_rsa_analysis(
        neural, scenes, neural_meta, inferred_physics=inferred_physics, fig_dir=fig_dir
    )

    # Step 7: Dissociation analysis
    print(f"\n[7/7] Running dissociation analysis (objective: {BEHAVIORAL_OBJECTIVE})...")
    dissociation_results = run_dissociation_analysis(
        neural, scenes, neural_meta, objective=BEHAVIORAL_OBJECTIVE, fig_dir=fig_dir
    )

    # Dynamics stub
    run_dynamics_analysis()

    # Summary
    print_summary(encoding_results, rsa_results, dissociation_results, pp_results=pp_results)

    elapsed = time.time() - t0
    print(f"Total time: {elapsed:.1f}s")


if __name__ == "__main__":
    main()
