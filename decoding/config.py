import os
import numpy as np

# paths

REPO_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
REPO_DIR = os.path.join("/Storage2", "anna", "semantic-decoding")
DATA_LM_DIR = os.path.join(REPO_DIR, "data_lm")
DATA_TRAIN_DIR = os.path.join(REPO_DIR, "data_train")
DATA_TEST_DIR = os.path.join(REPO_DIR, "data_test")
MODEL_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models"
).replace("Storage2", "home")
RESULT_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results"
).replace("Storage2", "home")
SCORE_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scores"
).replace("Storage2", "home")

# GPT encoding model parameters

TRIM = 5
STIM_DELAYS = [1, 2, 3, 4]
RESP_DELAYS = [-4, -3, -2, -1]
ALPHAS = np.logspace(1, 3, 10)
NBOOTS = 50
VOXELS = 10000
CHUNKLEN = 40
GPT_LAYER = {"original": 9, "llama3": 13, "gpt": 10, "opt": 22}
GPT_WORDS = 5
IS_PCA = True

# decoder parameters

RANKED = True
WIDTH = 200
NM_ALPHA = 2 / 3
LM_TIME = 8
LM_MASS = 0.9
LM_RATIO = 0.1
# LM_MASS = 0.98
# LM_RATIO = 0.02
EXTENSIONS = 5

# evaluation parameters

WINDOW = 20

# devices

GPT_DEVICE = "cuda"
EM_DEVICE = "cuda"
SM_DEVICE = "cuda"

gpt = "perceived"
MODELS = {
    "original": lambda gpt: os.path.join(DATA_LM_DIR, gpt, "model"),
    "llama3": "meta-llama/Meta-Llama-3-8B",
    "opt": "facebook/opt-6.7b",
    "gpt": "openai-community/openai-gpt",
}

MARK = {
    "original": " ",
    "llama3": "",
    "opt": "",
    "gpt": "",
}
