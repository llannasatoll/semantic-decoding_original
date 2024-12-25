import os
import numpy as np

# paths

# REPO_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
REPO_DIR = os.path.join("/Storage2", "anna", "semantic-decoding")
DATA_LM_DIR = os.path.join(REPO_DIR, "data_lm")
DATA_TRAIN_DIR = os.path.join(REPO_DIR, "data_train")
DATA_TEST_DIR = os.path.join(REPO_DIR, "data_test")
MODEL_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models"
)#.replace("Storage2", "home")
print("MODEL_DIR :", MODEL_DIR)
RESULT_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results"
)#.replace("Storage2", "home")
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
GPT_LAYER = {"original": 9, "llama3": 13, "gpt": 10, "opt": 22, "llama70b": 33, "falcon": 17}
GPT_LAYERS = {"original": [9],"llama3": [13],  "gpt": list(range(1,13)), "llama70b": list(range(1,81,4)), "falcon": list(range(1,61,4))}
GPT_LAYERS = {"original": [9], "llama3": [13], "gpt": list(range(1,13)), "llama70b": [25,29,33,37], "falcon": list(range(1,61,4))}
GPT_WORDS = 5
IS_PCA = True
print("IS_PCA :", IS_PCA)

# decoder parameters

RANKED = True
WIDTH = 200
NM_ALPHA = 2 / 3
LM_TIME = 8
LM_MASS = 0.9
LM_RATIO = 0.1
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
    "llama70b": "meta-llama/Llama-3.1-70B",
    "falcon": "tiiuae/falcon-40b",
}

MARK = {
    "original": " ",
    "llama3": "",
    "opt": "",
    "gpt": "",
    "llama70b": "",
}
