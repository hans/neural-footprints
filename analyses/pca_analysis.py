"""
PCA Negative Analysis: variance-maximizing components miss physics.

Shows that PCA on neural activity converges on render-dominated dimensions.
Motion direction (a causally operative physics variable) is invisible in the
top PCs and barely decodable even from all PCs.

A sensory positive control (mean pixel brightness) confirms that the same
early PCs *do* carry render-side structure — the failure is specific to
physics, not a flaw in PCA itself. A medium-footprint positive (pillar
gray) shows the typical mid-PC saturation.
"""

import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegressionCV
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler


def _build_targets(scenes):
    """Binary classification labels for the PCA decoding sweep.

    All targets are balanced ~50/50: median splits are exact (uniform
    samples have no ties); motion direction is symmetric by sampling.
    """
    mean_brightness = scenes['initial_renders'].mean(axis=1)
    pillar_grays = np.asarray(scenes['pillar_grays'])
    cam_height = np.array([
        float(lt['camJitter'][2]) for lt in scenes['lightings']
    ])
    vx = scenes['initial_physics_labels'][:, 7]
    return {
        'cam_height':      (cam_height > np.median(cam_height)).astype(int),
        'mean_brightness': (mean_brightness > np.median(mean_brightness)).astype(int),
        'pillar_gray':     (pillar_grays > np.median(pillar_grays)).astype(int),
        'motion_dir':      (vx > 0).astype(int),
    }


def run_pca_analysis(neural_activity, scenes, neural_meta,
                     n_permutations=50, compute_null=True):
    print("\n" + "=" * 60)
    print("PCA NEGATIVE ANALYSIS: Variance ≠ Information")
    print("=" * 60)

    n_scenes, n_neurons = neural_activity.shape

    targets = _build_targets(scenes)
    for name, y in targets.items():
        n_pos = int(y.sum())
        print(f"  {name}: {n_scenes - n_pos} class-0 / {n_pos} class-1")

    neural_scaled = StandardScaler().fit_transform(neural_activity)
    pca = PCA(n_components=n_neurons, random_state=42)
    neural_pca = pca.fit_transform(neural_scaled)
    cumvar = np.cumsum(pca.explained_variance_ratio_)

    pc_counts = sorted(set(
        list(range(1, 11)) + [15, 25, 50, 100, 200, n_neurons]
    ))
    pc_counts = [k for k in pc_counts if k <= n_neurons]

    decode_accs_per_target = {}
    chance_per_target = {}
    rng = np.random.default_rng(0)
    for name, y in targets.items():
        print(f"  Decoding '{name}':")
        accs = []
        for k in pc_counts:
            scores = cross_val_score(
                LogisticRegressionCV(cv=5, max_iter=1000, random_state=42),
                neural_pca[:, :k], y, cv=5, scoring='accuracy',
            )
            acc = scores.mean()
            accs.append(acc)
            print(f"    Top {k:>3d} PCs: accuracy = {acc:.2%}")
        decode_accs_per_target[name] = accs

        if compute_null:
            print(f"    Computing null for '{name}' ({n_permutations} shuffles)...")
            perm_accs = np.zeros((len(pc_counts), n_permutations))
            for p in range(n_permutations):
                shuffled = rng.permutation(y)
                for i, k in enumerate(pc_counts):
                    perm_accs[i, p] = cross_val_score(
                        LogisticRegressionCV(cv=5, max_iter=1000, random_state=42),
                        neural_pca[:, :k], shuffled, cv=5, scoring='accuracy',
                    ).mean()
            chance_per_target[name] = {
                'lo': np.percentile(perm_accs, 2.5, axis=1).tolist(),
                'hi': np.percentile(perm_accs, 97.5, axis=1).tolist(),
            }
        else:
            chance_per_target[name] = {
                'lo': [0.5] * len(pc_counts),
                'hi': [0.5] * len(pc_counts),
            }

    motion_dir = targets['motion_dir']

    return {
        'cumulative_variance': cumvar.tolist(),
        'pc_counts': pc_counts,
        'target_names': list(targets.keys()),
        'decode_accs_per_target': {
            name: [float(a) for a in accs]
            for name, accs in decode_accs_per_target.items()
        },
        'chance_per_target': chance_per_target,
        'all_pc_decoding_accuracy': float(decode_accs_per_target['motion_dir'][-1]),
        'neural_pca_2d': neural_pca[:, :2],
        'motion_dir': motion_dir,
        'n_neurons': n_neurons,
    }
