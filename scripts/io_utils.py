"""Serialization helpers for pipeline intermediates."""

import json
import numpy as np


_SCENE_ARRAY_KEYS = [
    'program_states', 'physics_labels', 'initial_physics_labels',
    'initial_renders', 'early_renders', 'late_renders', 'target_renders',
    'behavior_labels', 'kinetic_energies',
]


def _slice_to_pair(s):
    return [s.start, s.stop]


def _pair_to_slice(p):
    return slice(p[0], p[1])


def save_scenes(scenes, path):
    """Save scenes dict to compressed .npz. Handles metadata and scene_configs as JSON."""
    arrays = {key: scenes[key] for key in _SCENE_ARRAY_KEYS}

    arrays['pillar_grays'] = np.array(scenes['pillar_grays'])

    # Serialize non-array fields as JSON strings.
    # slice objects aren't JSON-serializable — convert to [start, stop] pairs.
    meta = dict(scenes['metadata'])
    meta['pixel_indices'] = _slice_to_pair(meta['pixel_indices'])
    meta['render_indices'] = _slice_to_pair(meta['render_indices'])
    meta['target_pixel_indices'] = _slice_to_pair(meta['target_pixel_indices'])
    meta['frame_render_indices'] = {
        name: _slice_to_pair(s)
        for name, s in meta['frame_render_indices'].items()
    }
    arrays['metadata_json'] = np.array(json.dumps(meta))
    arrays['scene_configs_json'] = np.array(json.dumps(scenes['scene_configs']))
    arrays['lightings_json'] = np.array(json.dumps(scenes['lightings']))

    np.savez_compressed(path, **arrays)


def load_scenes(path):
    """Load scenes dict from .npz."""
    data = np.load(path, allow_pickle=False)
    scenes = {key: data[key] for key in _SCENE_ARRAY_KEYS}

    meta = json.loads(str(data['metadata_json']))
    meta['pixel_indices'] = _pair_to_slice(meta['pixel_indices'])
    meta['render_indices'] = _pair_to_slice(meta['render_indices'])
    meta['target_pixel_indices'] = _pair_to_slice(meta['target_pixel_indices'])
    meta['frame_render_indices'] = {
        name: _pair_to_slice(p)
        for name, p in meta['frame_render_indices'].items()
    }
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
