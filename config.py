N_SCENES = 2000
N_OBJECTS = 3
IMAGE_SIZE = 64
N_NEURONS = 500
N_TIMESTEPS = 30
NOISE_LEVEL = 0.3       # fraction of signal std
RANDOM_SEED = 42
RSA_SUBSAMPLE = 500     # scenes for RDM (n^2 pairwise matrix)
BULLET_BYTES_K = None   # set automatically in scene_generator calibration pass
PIXEL_PCA_DIM = 200     # for encoding analysis only (analysis-side tractability)

# Behavioral sufficiency objective for dissociation analysis.
# "next_frame_pixels": Ridge R² predicting final-frame pixels from initial state
# "kinetic_energy":    logistic accuracy predicting KE-based binary label
BEHAVIORAL_OBJECTIVE = "next_frame_pixels"
