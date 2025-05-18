import os
import numpy as np
import json
import argparse
import logging
import sys
from scipy import stats
import h5py
import torch

import config
from GPT import GPT
from StimulusModel import LMFeatures
from utils_stim import get_stim
from utils_resp import get_resp

from utils_ridge.ridge import ridge, bootstrap_ridge
from utils_transformer.tmp import train_em, FMRISequenceDataset


np.random.seed(42)
zs = lambda v: (v - v.mean(0)) / v.std(0)  ## z-score function


def compute_correlation(x, y):
    mean_x = np.mean(x)
    mean_y = np.mean(y)

    cov_xy = np.mean((x - mean_x) * (y - mean_y))

    std_x = np.sqrt(np.mean((x - mean_x) ** 2))
    std_y = np.sqrt(np.mean((y - mean_y) ** 2))

    correlation = cov_xy / (std_x * std_y)
    return correlation


def compute_p_one_sided(corrs, n):
    t = corrs * np.sqrt(n - 2) / np.sqrt(1 - corrs**2)
    p = stats.t.sf(t, df=n - 2)
    return p


if __name__ == "__main__":

    # parser = argparse.ArgumentParser()
    # parser.add_argument("--subject", type=str, required=True)
    # parser.add_argument("--llm", type=str, required=True)
    # args = parser.parse_args()
    llm = "llama70b"
    llms = ["llama70b", "llama1b", "llama3b", "original", "gpt", "llama3.1"]
    # llms = ["original", "gpt", "llama3.1"]
    for llm in llms:
        print(llm)
        res = []
        for sub in [1, 2, 3]:
            a = np.load(
                f"/Storage2/anna/semantic-decoding_original/models/UTS0{sub}/{llm}/encoding_model_perceived.npz"
            )

            if llm in ["original", "gpt"]:
                gpt = GPT(llm, device=config.GPT_DEVICE)
                context_words = config.GPT_WORDS
            else:
                gpt = GPT(llm, device=config.GPT_DEVICE, not_load_model=True)
                context_words = -1
            features = LMFeatures(
                model=gpt,
                layer=config.GPT_LAYER[llm],
                context_words=context_words,
            )

            rstim = get_stim(["laluna"], features, tr_stats=a["tr_stats"])
            pred = rstim.dot(a["weights"])
            with h5py.File(
                os.path.join(
                    config.DATA_TEST_DIR,
                    "test_response",
                    f"UTS0{sub}",
                    "perceived_movie",
                    "laluna" + ".hf5",
                ),
                "r",
            ) as hf:
                resp = np.nan_to_num(hf["data"][:])
            corr = np.array(
                [
                    compute_correlation(resp[:, i], pred[:, i])
                    for i in range(resp.shape[-1])
                ]
            )
            print(f"Layer : {config.GPT_LAYER[llm]} , mean(fdr(corr)) : {corr.mean()}")
            res.append(corr.mean())
        print(np.mean(res))
