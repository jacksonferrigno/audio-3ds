from src.env.echo_env import DEFAULT_MAX_STEPS

N_TRAIN_SAMPLES = 10_000
N_VAL_SAMPLES = 500
BATCH_SIZE = 16
EPOCHS = 20
LEARNING_RATE = 1e-3
STEPS_PER_SAMPLE = DEFAULT_MAX_STEPS
MAX_OBJECTS = 8
MAX_PEOPLE = 4
DATALOADER_WORKERS = 0
DEFAULT_GEN_WORKERS = 2  # parallel RL rollouts during cache build — keep low to avoid melting the machine

DEFAULT_RL_CHECKPOINT = "checkpoints/best_model/best_model.zip"
CACHE_DIR = "data/cnn_cache"
CHECKPOINT_DIR = "checkpoints/cnn"
BEST_MODEL_PATH = f"{CHECKPOINT_DIR}/best_model.pt"
LATEST_MODEL_PATH = f"{CHECKPOINT_DIR}/latest_model.pt"
