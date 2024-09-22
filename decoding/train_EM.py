import os
import numpy as np
import json
import argparse
import logging
import sys
from scipy import stats
import h5py

import config
from GPT import GPT
from StimulusModel import LMFeatures
from utils_stim import get_stim
from utils_resp import get_resp
from utils_ridge.ridge import ridge, bootstrap_ridge

np.random.seed(42)


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
    logger = logging.getLogger("train_EM")
    logger.setLevel(logging.WARNING)
    stdout_handler = logging.StreamHandler(stream=sys.stdout)
    stdout_handler.setLevel(logging.INFO)
    logger.addHandler(stdout_handler)

    parser = argparse.ArgumentParser()
    parser.add_argument("--subject", type=str, required=True)
    parser.add_argument("--llm", type=str, required=True)
    parser.add_argument("--gpt", type=str, default="perceived")
    parser.add_argument("--notsave", action="store_true")
    parser.add_argument(
        "--sessions",
        nargs="+",
        type=int,
        default=[2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 15, 18, 20],
    )
    args = parser.parse_args()

    # training stories
    stories = []
    with open(os.path.join(config.DATA_TRAIN_DIR, "sess_to_story.json"), "r") as f:
        sess_to_story = json.load(f)
    for sess in args.sessions:
        stories.extend(sess_to_story[str(sess)])

    layers = [config.GPT_LAYER[args.llm]]
    if args.notsave:
        # Calculate correlation using test story.
        with h5py.File(
            os.path.join(
                config.DATA_TEST_DIR,
                "test_response",
                args.subject,
                "perceived_speech",
                "wheretheressmoke" + ".hf5",
            ),
            "r",
        ) as hf:
            resp = np.nan_to_num(hf["data"][:])
        layers = (
            [1, 3, 5, 7, 9, 11]
            if args.llm in ["original", "gpt"]
            else [1, 4, 7, 10, 13, 16, 19, 22, 25, 28, 31]
        )

    gpt = GPT(llm=args.llm, device=config.GPT_DEVICE, gpt=args.gpt)
    rresp = get_resp(args.subject, stories, stack=True)

    for layer in layers:
        features = LMFeatures(
            model=gpt,
            layer=layer,
            context_words=config.GPT_WORDS,
        )

        # estimate encoding model
        rstim, tr_stats, word_stats = get_stim(stories, features)
        nchunks = int(np.ceil(rresp.shape[0] / 5 / config.CHUNKLEN))
        weights, alphas, bscorrs = bootstrap_ridge(
            rstim,
            rresp,
            use_corr=False,
            alphas=config.ALPHAS,
            nboots=config.NBOOTS,
            chunklen=config.CHUNKLEN,
            nchunks=nchunks,
            logger=logger,
        )
        bscorrs = bscorrs.mean(2).max(0)
        vox = np.sort(np.argsort(bscorrs)[-config.VOXELS :])
        # del rstim, rresp

        if args.notsave:
            # Calculate correlation using test story.
            rstim = get_stim(["wheretheressmoke"], features, tr_stats=tr_stats)
            pred = rstim.dot(weights)
            corr = np.array(
                [
                    compute_correlation(resp[:, i], pred[:, i])
                    for i in range(resp.shape[-1])
                ]
            )
            logger.warning(f"Layer : {layer} , mean(fdr(corr)) : {corr.mean()}")
    if args.notsave:
        exit()

    # estimate noise model
    stim_dict = {
        story: get_stim([story], features, tr_stats=tr_stats) for story in stories
    }
    resp_dict = get_resp(args.subject, stories, stack=False, vox=vox)
    noise_model = np.zeros([len(vox), len(vox)])
    for hstory in stories:
        tstim, hstim = (
            np.vstack([stim_dict[tstory] for tstory in stories if tstory != hstory]),
            stim_dict[hstory],
        )
        tresp, hresp = (
            np.vstack([resp_dict[tstory] for tstory in stories if tstory != hstory]),
            resp_dict[hstory],
        )
        bs_weights = ridge(tstim, tresp, alphas[vox])
        resids = hresp - hstim.dot(bs_weights)
        bs_noise_model = resids.T.dot(resids)
        noise_model += bs_noise_model / np.diag(bs_noise_model).mean() / len(stories)
    del stim_dict, resp_dict

    # save
    save_location = os.path.join(config.MODEL_DIR, args.subject)
    os.makedirs(os.path.join(save_location, args.llm), exist_ok=True)
    np.savez(
        os.path.join(save_location, args.llm, "encoding_model_%s" % args.gpt),
        weights=weights,
        noise_model=noise_model,
        alphas=alphas,
        voxels=vox,
        stories=stories,
        tr_stats=np.array(tr_stats),
        word_stats=np.array(word_stats),
        llm=args.llm,
    )
