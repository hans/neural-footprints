import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from load_config import load_config
from io_utils import load_scenes, save_neural
from neural_model import generate_neural_activity, print_variance_diagnostic

cfg = load_config()
scenes = load_scenes(snakemake.input.scenes)

neural, neural_meta = generate_neural_activity(
    scenes['program_states'], cfg['random_seed'],
    n_neurons=cfg['n_neurons'], noise_level=cfg['noise_level'],
)
print_variance_diagnostic(scenes['metadata'], neural_meta)

save_neural(neural, neural_meta, snakemake.output.neural)
