import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from load_config import load_config
from io_utils import load_scenes, save_neural
from neural_model import generate_neural_activity, print_variance_diagnostic

cfg = load_config()
scenes = load_scenes(snakemake.input.scenes)

scene_meta = scenes['metadata']
block_sizes = [
    scene_meta['D_render_bytes'],
    scene_meta['D_physics_labels'],
    scene_meta['D_scene_config'],
    scene_meta.get('D_scene_lighting', 0),
]

neural, neural_meta = generate_neural_activity(
    scenes['program_states'], cfg['random_seed'],
    n_neurons=cfg['n_neurons'], noise_level=cfg['noise_level'],
    block_sizes=block_sizes,
)
print_variance_diagnostic(scene_meta, neural_meta, block_sizes)

save_neural(neural, neural_meta, snakemake.output.neural)
