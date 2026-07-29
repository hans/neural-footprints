"""Diagnostic: PCA decoding with raw observed renders vs. forward-model renders.

Hypothesis under test:
  The high PC1 motion_dir accuracy (0.876) is caused by the forward model baking
  inferred linvel_x (R²≈0.59) into the render block, not by raw pixel statistics.

Method:
  Replace the forward-model render block with raw observed frames
  (initial + early + late renders from scenes.npz), project through the same
  random W, run the same PCA decoding sweep, and compare PC1 motion_dir accuracy.

If raw-frame PC1 ≈ 0.876 → pixel statistics / scene diversity explains it.
If raw-frame PC1 ≈ chance → forward-model leak is the cause.
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import json
import numpy as np
from analyses.pca_analysis import run_pca_analysis

# ── load data ────────────────────────────────────────────────────────────────
print("Loading data...")
scenes_npz = np.load("data/scenes.npz", allow_pickle=True)
neural_npz = np.load("data/neural.npz")

W = neural_npz["W"]          # (n_neurons, D_total)
means = neural_npz["means"]  # (D_total,)

initial_renders = scenes_npz["initial_renders"]   # (N, 49152)
early_renders   = scenes_npz["early_renders"]     # (N, 49152)
late_renders    = scenes_npz["late_renders"]      # (N, 49152)
initial_physics = scenes_npz["initial_physics_labels"]  # (N, 16)
pillar_grays    = scenes_npz["pillar_grays"]
lightings       = json.loads(scenes_npz["lightings_json"].item())

D_render = initial_renders.shape[1] * 3  # 147456

# ── build raw-frame neural activity ──────────────────────────────────────────
print(f"Constructing raw-frame input: {D_render} dims (3 × {initial_renders.shape[1]})")
raw_input = np.concatenate([initial_renders, early_renders, late_renders], axis=1).astype(np.float32)

W_render = W[:, :D_render]                     # (n_neurons, D_render)
means_render = means[:D_render]                # (D_render,)

# Same random projection as gen_neural.py (no noise — isolates render signal)
neural_raw = (raw_input - means_render) @ W_render.T   # (N, n_neurons)
print(f"  neural_raw: shape={neural_raw.shape}, dtype={neural_raw.dtype}")

# ── reconstruct scenes dict for _build_targets ────────────────────────────
scenes = {
    "pillar_grays": pillar_grays,
    "lightings": lightings,
    "initial_physics_labels": initial_physics,
}

# ── run PCA analysis (no permutation null for speed) ─────────────────────────
print("\nRunning PCA on RAW-FRAME neural activity (no permutation null)...")
results_raw = run_pca_analysis(neural_raw, scenes, {}, compute_null=False)

# ── load existing forward-model results for comparison ───────────────────────
import json as _json
with open("outputs/pca_results.json") as f:
    results_fwd = _json.load(f)

# ── comparison table ─────────────────────────────────────────────────────────
targets = results_raw["target_names"]
pc_counts = results_raw["pc_counts"]

# PC1 index
pc1_idx = pc_counts.index(1)

print("\n" + "=" * 65)
print("PC1 DECODING ACCURACY: raw observed renders vs. forward-model renders")
print("=" * 65)
print(f"{'Target':<15} {'Raw-frame PC1':>15} {'Fwd-model PC1':>15}  {'Δ':>8}")
print("-" * 65)
for name in targets:
    raw_pc1 = results_raw["decode_accs_per_target"][name][pc1_idx]
    fwd_accs = results_fwd["decode_accs_per_target"][name]
    fwd_pc1  = fwd_accs[pc1_idx] if pc1_idx < len(fwd_accs) else float("nan")
    delta = raw_pc1 - fwd_pc1
    print(f"  {name:<13} {raw_pc1:>14.3f} {fwd_pc1:>14.3f}  {delta:>+8.3f}")

print("=" * 65)
print(f"\nFor reference, chance ≈ 0.50 (balanced binary targets, no permutation null run).")
print("\nInterpretation:")
for name in targets:
    raw_pc1 = results_raw["decode_accs_per_target"][name][pc1_idx]
    fwd_accs = results_fwd["decode_accs_per_target"][name]
    fwd_pc1  = fwd_accs[pc1_idx] if pc1_idx < len(fwd_accs) else float("nan")
    if abs(raw_pc1 - fwd_pc1) < 0.05:
        note = "similar → pixel statistics / scene diversity drives this target"
    elif raw_pc1 < fwd_pc1 - 0.10:
        note = "raw much lower → forward-model leak inflates fwd result"
    elif raw_pc1 > fwd_pc1 + 0.10:
        note = "raw much higher → (unexpected; raw renders carry more signal)"
    else:
        note = "modest difference → mixed effect"
    print(f"  {name}: {note}")
