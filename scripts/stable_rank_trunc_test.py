"""
Experiment: stable-rank truncation vs operator-norm block normalization.

Hypothesis: truncating each block to its stable rank k = round(||X||_F² / ||X||_op²)
and using the whitened score matrix U[:, :k] eliminates noise-dominated directions
so that the PC structure directly reflects stable-rank ratios across blocks, causing
motion_dir PC1 accuracy to drop toward the no-normalization baseline.

Block structure (4 blocks):
  1. raw_renders:       initial+early+late observed frames  (scenes.npz)     147456 dims
  2. fwd_renders:       forward-model render slice          (forward_renders.npz) 147456 dims
  3. hidden_acts:       physics-perception network hidden   (pp_activations.npz)  256 dims
  4. inferred_physics:  predicted physics labels            (pp_activations.npz)   16 dims
"""

import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from analyses.pca_analysis import run_pca_analysis
from neural_model import generate_neural_activity

# ── constants ─────────────────────────────────────────────────────────────────
BLOCK_NAMES = ["raw_renders", "fwd_renders", "hidden_acts", "inferred_physics"]
N_NEURONS = 500
SEED = 42

# ── load data ─────────────────────────────────────────────────────────────────
print("Loading data...")
fwd_npz = np.load("data/forward_renders.npz")
pp_npz = np.load("data/pp_activations.npz")
scenes_npz = np.load("data/scenes.npz", allow_pickle=True)

initial_renders = scenes_npz["initial_renders"].astype(np.float32)  # (2000, 49152)
early_renders = scenes_npz["early_renders"].astype(np.float32)  # (2000, 49152)
late_renders = scenes_npz["late_renders"].astype(np.float32)  # (2000, 49152)
raw_renders = np.concatenate([initial_renders, early_renders, late_renders], axis=1)
# raw_renders: (2000, 147456)

fwd_renders = fwd_npz["forward_program_states"][:, :147456].astype(np.float32)
# fwd_renders: (2000, 147456)

hidden_acts = pp_npz["hidden_acts"].astype(np.float32)  # (2000, 256)
inferred_physics = pp_npz["inferred_physics"].astype(np.float32)  # (2000, 16)

neural_input = np.concatenate(
    [raw_renders, fwd_renders, hidden_acts, inferred_physics], axis=1
)
BLOCK_SIZES = [
    raw_renders.shape[1],
    fwd_renders.shape[1],
    hidden_acts.shape[1],
    inferred_physics.shape[1],
]
print(f"Neural input shape: {neural_input.shape}  block_sizes={BLOCK_SIZES}")

initial_physics_labels = scenes_npz["initial_physics_labels"]
pillar_grays = scenes_npz["pillar_grays"]
lightings = json.loads(scenes_npz["lightings_json"].item())

scenes_dict = {
    "pillar_grays": pillar_grays,
    "lightings": lightings,
    "initial_physics_labels": initial_physics_labels,
}

# ── block stable ranks (before normalization) ──────────────────────────────────
print("\n" + "=" * 72)
print("BLOCK STATS (after centering, before normalization)")
print("=" * 72)
print(f"{'Block':<20} {'Dims':>8} {'FrobNorm':>14} {'OpNorm':>14} {'StableRank':>12}")
print("-" * 72)

means = neural_input.mean(axis=0)
centered_check = neural_input - means

start = 0
for name, size in zip(BLOCK_NAMES, BLOCK_SIZES):
    block = centered_check[:, start : start + size].astype(np.float64)
    fnorm = float(np.sqrt((block**2).sum()))
    n, d = block.shape
    gram = block @ block.T if n <= d else block.T @ block
    max_ev = float(np.linalg.eigvalsh(gram).max())
    opnorm = float(np.sqrt(max(max_ev, 0.0)))
    sr = (fnorm**2) / (opnorm**2) if opnorm > 0 else float("nan")
    print(f"  {name:<18} {size:>8} {fnorm:>14.3e} {opnorm:>14.3e} {sr:>12.1f}")
    start += size
print("=" * 72)

# ── operator-norm version ──────────────────────────────────────────────────────
print("\nRunning OPERATOR NORM version...")
neural_op, meta_op = generate_neural_activity(
    neural_input.copy(),
    seed=SEED,
    n_neurons=N_NEURONS,
    noise_level=0.0,
    block_sizes=BLOCK_SIZES,
    normalization="operator_norm",
)
print(f"  W shape: {meta_op['W'].shape}")
print(
    f"  block operator norms: {dict(zip(BLOCK_NAMES, meta_op['block_norms'].tolist()))}"
)

# ── stable-rank truncation version ────────────────────────────────────────────
print("\nRunning STABLE_RANK_TRUNC version...")
neural_trunc, meta_trunc = generate_neural_activity(
    neural_input.copy(),
    seed=SEED,
    n_neurons=N_NEURONS,
    noise_level=0.0,
    block_sizes=BLOCK_SIZES,
    normalization="stable_rank_trunc",
)
print(f"  W shape: {meta_trunc['W'].shape}  (D_proj = {meta_trunc['W'].shape[1]})")
print(
    f"  block stable ranks: {dict(zip(BLOCK_NAMES, meta_trunc['block_stable_ranks'].tolist()))}"
)
print(
    f"  block k values:     {dict(zip(BLOCK_NAMES, meta_trunc['block_k_values'].tolist()))}"
)

# ── PCA analysis ──────────────────────────────────────────────────────────────
print("\n" + "=" * 72)
print("PCA ANALYSIS — OPERATOR NORM")
print("=" * 72)
results_op = run_pca_analysis(neural_op, scenes_dict, meta_op, compute_null=False)

print("\n" + "=" * 72)
print("PCA ANALYSIS — STABLE RANK TRUNC")
print("=" * 72)
results_trunc = run_pca_analysis(neural_trunc, scenes_dict, {}, compute_null=False)

# ── comparison table ──────────────────────────────────────────────────────────
pc_counts = results_op["pc_counts"]
targets = results_op["target_names"]

show_pcs = [1, 5, 10, 50, 100, 500]
show_indices = [i for i, k in enumerate(pc_counts) if k in show_pcs]
show_counts = [pc_counts[i] for i in show_indices]

print("\n" + "=" * 78)
print("COMPARISON TABLE: Decoding accuracy by normalization scheme")
print("=" * 78)
for name in targets:
    op_accs = results_op["decode_accs_per_target"][name]
    trunc_accs = results_trunc["decode_accs_per_target"][name]
    print(f"\n  Target: {name}")
    print(
        f"  {'PCs':>6}  {'OperatorNorm':>14}  {'StableRankTrunc':>17}  {'Δ(Trunc-Op)':>13}"
    )
    print(f"  {'-'*58}")
    for idx, k in zip(show_indices, show_counts):
        op_a = op_accs[idx]
        trunc_a = trunc_accs[idx]
        delta = trunc_a - op_a
        print(f"  {k:>6}  {op_a:>14.3f}  {trunc_a:>17.3f}  {delta:>+13.3f}")

print("\n" + "=" * 78)
print("PC1 SUMMARY")
print("=" * 78)
pc1_idx = pc_counts.index(1)
print(f"  {'Target':<20} {'OpNorm PC1':>12} {'StableRankTrunc PC1':>21} {'Δ':>8}")
print(f"  {'-'*64}")
for name in targets:
    op_pc1 = results_op["decode_accs_per_target"][name][pc1_idx]
    trunc_pc1 = results_trunc["decode_accs_per_target"][name][pc1_idx]
    delta = trunc_pc1 - op_pc1
    print(f"  {name:<20} {op_pc1:>12.3f} {trunc_pc1:>21.3f} {delta:>+8.3f}")
print("=" * 78)
print("\nInterpretation: motion_dir PC1 should DROP with stable-rank truncation")
print("(physics block collapses to k~1 dim; raw_renders at k~15 dominates instead)")
