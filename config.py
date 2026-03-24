N_SCENES = 2000
N_OBJECTS = 1
IMAGE_SIZE = 64
N_NEURONS = 500
N_TIMESTEPS = 30
NOISE_LEVEL = 0.3       # fraction of signal std
RANDOM_SEED = 42
RSA_SUBSAMPLE = 500     # scenes for RDM (n^2 pairwise matrix)
BULLET_BYTES_K = None   # set automatically in scene_generator calibration pass
PIXEL_PCA_DIM = 500     # for encoding analysis only (analysis-side tractability)
BEHAVIORAL_PCA_DIM = 50 # for next-frame behavioral task (MLP output dim; must be < hidden layer size)

# Behavioral sufficiency objective for dissociation analysis.
# "next_frame_pixels": Ridge R² predicting final-frame pixels from initial state
# "kinetic_energy":    logistic accuracy predicting KE-based binary label
BEHAVIORAL_OBJECTIVE = "next_frame_pixels"

# Predictive Processing model hyperparameters
PP_HIDDEN_DIM = 256      # hidden layer width for InverseModel
PP_PIXEL_PCA_DIM = 50    # pixel PCA dim for two-frame input (keep small for 2000 scenes)
PP_EARLY_FRAME = 5       # simulation step at which to capture the second input frame
PP_DROPOUT_RATE = 0.05   # dropout probability (active during both training and MC inference)
