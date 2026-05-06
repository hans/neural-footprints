"""
PCA Negative Analysis: variance-maximizing components miss physics.

Shows that PCA on neural activity converges on render-dominated dimensions.
Motion direction (a causally operative physics variable) is invisible in the
top PCs and barely decodable even from all PCs.
"""

import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegressionCV
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
def run_pca_analysis(neural_activity, scenes, neural_meta, n_permutations=50):
    print("\n" + "=" * 60)
    print("PCA NEGATIVE ANALYSIS: Variance ≠ Information")
    print("=" * 60)

    n_scenes, n_neurons = neural_activity.shape

    # Extract motion direction: vx at t=0 is index 7 in physics labels
    vx = scenes['initial_physics_labels'][:, 7]
    motion_dir = (vx > 0).astype(int)  # 1 = rightward, 0 = leftward
    n_right = motion_dir.sum()
    print(f"  Motion direction: {n_scenes - n_right} left, {n_right} right")

    # Full PCA on standardized neural activity
    neural_scaled = StandardScaler().fit_transform(neural_activity)
    pca = PCA(n_components=n_neurons, random_state=42)
    neural_pca = pca.fit_transform(neural_scaled)
    cumvar = np.cumsum(pca.explained_variance_ratio_)

    # Decoding accuracy as function of number of PCs
    pc_counts = [k for k in [1, 2, 5, 10, 25, 50, 100, 200, n_neurons] if k <= n_neurons]
    decode_accs = []
    for k in pc_counts:
        scores = cross_val_score(
            LogisticRegressionCV(cv=5, max_iter=1000, random_state=42),
            neural_pca[:, :k], motion_dir, cv=5, scoring='accuracy',
        )
        acc = scores.mean()
        decode_accs.append(acc)
        print(f"    Top {k:>3d} PCs: accuracy = {acc:.2%}")

    all_pc_acc = decode_accs[-1]

    # Permutation null distribution: shuffle labels, re-decode at each k
    print(f"  Computing permutation null ({n_permutations} shuffles)...")
    rng = np.random.default_rng(0)
    perm_accs = np.zeros((len(pc_counts), n_permutations))
    for p in range(n_permutations):
        shuffled = rng.permutation(motion_dir)
        for i, k in enumerate(pc_counts):
            perm_accs[i, p] = cross_val_score(
                LogisticRegressionCV(cv=5, max_iter=1000, random_state=42),
                neural_pca[:, :k], shuffled, cv=5, scoring='accuracy',
            ).mean()
    chance_lo = np.percentile(perm_accs, 5, axis=1)
    chance_hi = np.percentile(perm_accs, 95, axis=1)

    return {
        'cumulative_variance': cumvar.tolist(),
        'all_pc_decoding_accuracy': float(all_pc_acc),
        'pc_counts': pc_counts,
        'decode_accuracies': [float(a) for a in decode_accs],
        'chance_lo': chance_lo.tolist(),
        'chance_hi': chance_hi.tolist(),
        'neural_pca_2d': neural_pca[:, :2],
        'motion_dir': motion_dir,
        'n_neurons': n_neurons,
    }
