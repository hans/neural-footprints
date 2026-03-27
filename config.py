"""Config shim — loads values from config.yaml for backward compatibility."""

from scripts.load_config import load_config

_cfg = load_config()

N_SCENES = _cfg['n_scenes']
N_OBJECTS = _cfg['n_objects']
IMAGE_SIZE = _cfg['image_size']
N_NEURONS = _cfg['n_neurons']
N_TIMESTEPS = _cfg['n_timesteps']
NOISE_LEVEL = _cfg['noise_level']
RANDOM_SEED = _cfg['random_seed']
RSA_SUBSAMPLE = _cfg['rsa_subsample']
PIXEL_PCA_DIM = _cfg['pixel_pca_dim']
BEHAVIORAL_PCA_DIM = _cfg['behavioral_pca_dim']
BEHAVIORAL_OBJECTIVE = _cfg['behavioral_objective']
