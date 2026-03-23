"""
Neural Footprints Simulation — Main runner.

Demonstrates that standard neuroscience analyses (encoding models, RSA)
systematically fail to detect high-level representations even when
those representations are causally operative.
"""

import os
import time

from config import N_SCENES, RANDOM_SEED, BEHAVIORAL_OBJECTIVE
from scene_generator import calibrate_bullet_size, generate_scenes, save_sample_renders
from neural_model import generate_neural_activity
from analyses.encoding import run_encoding_analysis
from analyses.rsa import run_rsa_analysis
from analyses.dissociation import run_dissociation_analysis
from analyses.dynamics import run_dynamics_analysis
from analyses.predictive_processing import run_predictive_processing_analysis
from evaluation import evaluate


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

    # Step 4: Predictive processing analysis (must run before encoding/RSA to get inferred latents)
    print("\n[4/7] Running predictive processing analysis...")
    pp_results = run_predictive_processing_analysis(neural, scenes, fig_dir=fig_dir)
    inferred_physics = pp_results['inferred_physics_all']

    # Step 5: Encoding analysis
    print("\n[5/7] Running encoding analysis...")
    encoding_results = run_encoding_analysis(
        neural, scenes, inferred_physics=inferred_physics, fig_dir=fig_dir
    )

    # Step 6: RSA analysis
    print("\n[6/7] Running RSA analysis...")
    rsa_results = run_rsa_analysis(
        neural, scenes, inferred_physics=inferred_physics, fig_dir=fig_dir
    )

    # Step 7: Dissociation analysis
    print(f"\n[7/7] Running dissociation analysis (objective: {BEHAVIORAL_OBJECTIVE})...")
    dissociation_results = run_dissociation_analysis(
        neural, scenes, objective=BEHAVIORAL_OBJECTIVE, fig_dir=fig_dir
    )

    # Dynamics stub
    run_dynamics_analysis()

    # Evaluation
    evaluate(encoding_results, rsa_results, dissociation_results, pp_results=pp_results)

    elapsed = time.time() - t0
    print(f"Total time: {elapsed:.1f}s")


if __name__ == "__main__":
    main()
