"""
Experiment: Frobenius norm vs operator norm block normalization.

Hypothesis: dividing each block by its Frobenius norm (not operator norm)
equalizes total variance across blocks proportional to stable rank, preventing
low-rank physics blocks from dominating PC1.
"""

import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from analyses.pca_analysis import run_pca_analysis
from neural_model import generate_neural_activity

BLOCK_SIZES = [147456, 256, 16]
BLOCK_NAMES = ["render", "hidden_acts", "inferred_physics"]
N_NEURONS = 500
SEED = 42

# ── load data ─────────────────────────────────────────────────────────────────
print("Loading data...")
fwd_npz = np.load("data/forward_renders.npz")
pp_npz = np.load("data/pp_activations.npz")
scenes_npz = np.load("data/scenes.npz", allow_pickle=True)

render = fwd_npz["forward_program_states"][:, : BLOCK_SIZES[0]]  # (2000, 147456)
hidden_acts = pp_npz["hidden_acts"]                              # (2000, 256)
inferred_physics = pp_npz["inferred_physics"]                    # (2000, 16)

neural_input = np.concatenate([render, hidden_acts, inferred_physics], axis=1)
print(f"Neural input shape: {neural_input.shape}")  # (2000, 147728)

initial_physics_labels = scenes_npz["initial_physics_labels"]
pillar_grays = scenes_npz["pillar_grays"]
lightings = json.loads(scenes_npz["lightings_json"].item())

scenes_dict = {
    "pillar_grays": pillar_grays,
    "lightings": lightings,
    "initial_physics_labels": initial_physics_labels,
}

# ── block stats ───────────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("BLOCK STATS (before normalization, after centering)")
print("=" * 70)
print(f"{'Block':<20} {'Dims':>8} {'Fnorm':>14} {'Opnorm':>14} {'StableRank':>12}")
print("-" * 70)

means = neural_input.mean(axis=0)
centered = neural_input - means

start = 0
block_fnorms = []
block_opnorms = []
block_stable_ranks = []

for name, size in zip(BLOCK_NAMES, BLOCK_SIZES):
    block = centered[:, start : start + size].astype(np.float64)
    fnorm = float(np.sqrt((block**2).sum()))
    n, d = block.shape
    gram = (block @ block.T if n <= d else block.T @ block)
    opnorm = float(np.sqrt(np.linalg.eigvalsh(gram).max()))
    stable_rank = (fnorm**2) / (opnorm**2) if opnorm > 0 else float("nan")
    block_fnorms.append(fnorm)
    block_opnorms.append(opnorm)
    block_stable_ranks.append(stable_rank)
    print(f"  {name:<18} {size:>8} {fnorm:>14.3e} {opnorm:>14.3e} {stable_rank:>12.1f}")
    start += size

print("=" * 70)

# ── operator norm version ─────────────────────────────────────────────────────
print("\nRunning OPERATOR NORM version...")
neural_op, meta_op = generate_neural_activity(
    neural_input.copy(),
    seed=SEED,
    n_neurons=N_NEURONS,
    noise_level=0.0,
    block_sizes=BLOCK_SIZES,
)
print(f"  block operator norms used: {dict(zip(BLOCK_NAMES, meta_op['block_norms'].tolist()))}")

# ── Frobenius norm version ────────────────────────────────────────────────────
print("\nRunning FROBENIUS NORM version...")
D = neural_input.shape[1]
centered_frob = neural_input.astype(np.float32) - neural_input.mean(axis=0).astype(np.float32)

start = 0
frob_norms_used = []
for name, size in zip(BLOCK_NAMES, BLOCK_SIZES):
    block = centered_frob[:, start : start + size]
    sigma = float(np.sqrt((block.astype(np.float64) ** 2).sum()))
    if sigma == 0.0:
        sigma = 1.0
    frob_norms_used.append(sigma)
    centered_frob[:, start : start + size] /= sigma
    start += size

print(f"  block Frobenius norms used: {dict(zip(BLOCK_NAMES, frob_norms_used))}")

# Same random projection as operator norm version (same seed, same W formula)
rng = np.random.default_rng(SEED)
W = rng.normal(0, 1.0 / np.sqrt(D), size=(N_NEURONS, D)).astype(np.float32)
neural_frob = centered_frob @ W.T  # (2000, 500)

# ── PCA analysis ──────────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("PCA ANALYSIS — OPERATOR NORM")
print("=" * 70)
results_op = run_pca_analysis(neural_op, scenes_dict, meta_op, compute_null=False)

print("\n" + "=" * 70)
print("PCA ANALYSIS — FROBENIUS NORM")
print("=" * 70)
results_frob = run_pca_analysis(neural_frob, scenes_dict, {}, compute_null=False)

# ── comparison table ──────────────────────────────────────────────────────────
pc_counts = results_op["pc_counts"]
targets = results_op["target_names"]

# Show PC1 and a few key PC counts
show_pcs = [1, 5, 10, 50, 100, 500]
show_indices = [pc_counts.index(k) for k in show_pcs if k in pc_counts]
show_counts = [pc_counts[i] for i in show_indices]

print("\n" + "=" * 75)
print("COMPARISON TABLE: Decoding accuracy by normalization scheme")
print("=" * 75)
for name in targets:
    op_accs = results_op["decode_accs_per_target"][name]
    frob_accs = results_frob["decode_accs_per_target"][name]
    print(f"\n  Target: {name}")
    print(f"  {'PCs':>6}  {'OperatorNorm':>14}  {'FrobeniusNorm':>14}  {'Δ(Frob-Op)':>12}")
    print(f"  {'-'*52}")
    for idx, k in zip(show_indices, show_counts):
        op_a = op_accs[idx]
        fr_a = frob_accs[idx]
        delta = fr_a - op_a
        print(f"  {k:>6}  {op_a:>14.3f}  {fr_a:>14.3f}  {delta:>+12.3f}")

print("\n" + "=" * 75)
print("PC1 SUMMARY")
print("=" * 75)
pc1_idx = pc_counts.index(1)
print(f"  {'Target':<20} {'OpNorm PC1':>12} {'FrobNorm PC1':>14} {'Δ':>10}")
print(f"  {'-'*60}")
for name in targets:
    op_pc1 = results_op["decode_accs_per_target"][name][pc1_idx]
    fr_pc1 = results_frob["decode_accs_per_target"][name][pc1_idx]
    delta = fr_pc1 - op_pc1
    print(f"  {name:<20} {op_pc1:>12.3f} {fr_pc1:>14.3f} {delta:>+10.3f}")
print("=" * 75)
print("\nInterpretation: motion_dir PC1 should DROP with Frobenius norm")
print("(physics block has low stable rank → contributes less to PC1)")
