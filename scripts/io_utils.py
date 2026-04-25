"""Serialization helpers for pipeline intermediates."""

import json
import numpy as np


def save_scenes(scenes, path):
    """Save scenes dict to compressed .npz. Handles metadata and scene_configs as JSON."""
    arrays = {}
    for key in ['program_states', 'physics_labels', 'initial_physics_labels',
                'initial_renders', 'early_renders', 'behavior_labels', 'kinetic_energies']:
        arrays[key] = scenes[key]

    arrays['pillar_grays'] = np.array(scenes['pillar_grays'])

    # Serialize non-array fields as JSON strings
    meta = dict(scenes['metadata'])
    # slice objects aren't JSON-serializable
    pi = meta['pixel_indices']
    meta['pixel_indices'] = [pi.start, pi.stop]
    ri = meta['render_indices']
    meta['render_indices'] = [ri.start, ri.stop]
    arrays['metadata_json'] = np.array(json.dumps(meta))
    arrays['scene_configs_json'] = np.array(json.dumps(scenes['scene_configs']))
    arrays['lightings_json'] = np.array(json.dumps(scenes['lightings']))

    np.savez_compressed(path, **arrays)


def load_scenes(path):
    """Load scenes dict from .npz."""
    data = np.load(path, allow_pickle=False)
    scenes = {}
    for key in ['program_states', 'physics_labels', 'initial_physics_labels',
                'initial_renders', 'early_renders', 'behavior_labels', 'kinetic_energies']:
        scenes[key] = data[key]

    meta = json.loads(str(data['metadata_json']))
    pi = meta['pixel_indices']
    meta['pixel_indices'] = slice(pi[0], pi[1])
    ri = meta['render_indices']
    meta['render_indices'] = slice(ri[0], ri[1])
    scenes['metadata'] = meta

    scenes['scene_configs'] = json.loads(str(data['scene_configs_json']))
    scenes['pillar_grays'] = data['pillar_grays'].tolist()
    scenes['lightings'] = json.loads(str(data['lightings_json']))
    return scenes


def save_neural(neural_activity, meta, path):
    """Save neural activity and metadata to compressed .npz."""
    np.savez_compressed(
        path,
        neural_activity=neural_activity,
        W=meta['W'],
        means=meta['means'],
        stds=meta['stds'],
        signal_std=np.array(meta['signal_std']),
        var_per_dim=meta['var_per_dim'],
        total_var=np.array(meta['total_var']),
    )


def load_neural(path):
    """Load neural activity and metadata from .npz. Returns (neural_activity, meta)."""
    data = np.load(path, allow_pickle=False)
    neural_activity = data['neural_activity']
    meta = {
        'W': data['W'],
        'means': data['means'],
        'stds': data['stds'],
        'signal_std': float(data['signal_std']),
        'var_per_dim': data['var_per_dim'],
        'total_var': float(data['total_var']),
    }
    return neural_activity, meta


def _convert_numpy(obj):
    """Recursively convert numpy types to Python natives for JSON serialization."""
    if isinstance(obj, dict):
        return {k: _convert_numpy(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [_convert_numpy(v) for v in obj]
    elif hasattr(obj, 'tolist'):
        return obj.tolist()
    elif isinstance(obj, (np.floating, np.integer, np.bool_)):
        return obj.item()
    return obj


def save_encoder(encoder, path):
    """Save fitted encoder (scaler, PCA, ridge) via joblib."""
    import joblib
    joblib.dump(encoder, path)


def load_encoder(path):
    """Load fitted encoder from joblib."""
    import joblib
    return joblib.load(path)


def save_results(results, path):
    """Save analysis results dict to JSON. Converts numpy types."""
    serializable = _convert_numpy(results)
    with open(path, 'w') as f:
        json.dump(serializable, f, indent=2)


def load_results(path):
    """Load analysis results dict from JSON."""
    with open(path) as f:
        return json.load(f)
