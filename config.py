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
RENDER_PCA_DIM = _cfg['render_pca_dim']
BEHAVIORAL_PCA_DIM = _cfg['behavioral_pca_dim']
BEHAVIORAL_OBJECTIVE = _cfg['behavioral_objective']

PP_HIDDEN_DIM = _cfg['pp_hidden_dim']
PP_PIXEL_PCA_DIM = _cfg['pp_pixel_pca_dim']
PP_EARLY_FRAME = _cfg['pp_early_frame']
PP_LATE_FRAME = _cfg['pp_late_frame']
PP_DROPOUT_RATE = _cfg['pp_dropout_rate']
PP_NEURAL_LAYER = _cfg.get('pp_neural_layer', 'h2')

CAMERA_FOV = _cfg['camera_fov']
LINVEL_X_MAX = _cfg.get('linvel_x_max', 8.0)
