import os
import numpy as np
import json
from sklearn.decomposition import PCA
import joblib

import config
from utils_ridge.stimulus_utils import TRFile, load_textgrids, load_simulated_trfiles
from utils_ridge.dsutils import make_word_ds
from utils_ridge.interpdata import lanczosinterp2D
from utils_ridge.util import make_delayed

pca = PCA(n_components=1000)


def get_story_wordseqs(stories):
    """loads words and word times of stimulus stories"""
    grids = load_textgrids(stories, config.DATA_TRAIN_DIR)
    with open(os.path.join(config.DATA_TRAIN_DIR, "respdict.json"), "r") as f:
        respdict = json.load(f)
    trfiles = load_simulated_trfiles(respdict)
    wordseqs = make_word_ds(grids, trfiles)
    return wordseqs


def get_stim(stories, features, tr_stats=None, use_embedding=False):
    """extract quantitative features of stimulus stories"""
    word_seqs = get_story_wordseqs(stories)
    word_vecs, wordind2tokind = {}, {}
    for story in stories:
        save_location = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "features",
            features.model.llm,
        ).replace("Storage2", config.WRITE_DIR)
        if use_embedding:
            _, wordind2tokind[story] = features.make_stim(
                word_seqs[story].data,
                story=story,
            )
            context = 5
            id_ = "embedding_" + "context" + str(context)
            word_vecs[story] = np.load(
                os.path.join(
                    config.DATA_TRAIN_DIR,
                    "train_stimulus",
                    id_,
                    "text-embedding-3-small",
                    f"{story}.npy",
                )
            )
            word_vecs[story][context - 1 :] = word_vecs[story][: -context + 1]
        elif features.context_words < 0:
            _, wordind2tokind[story] = features.make_stim(
                word_seqs[story].data,
                story=story,
            )
            word_vecs[story] = np.load(
                os.path.join(
                    save_location, story + "_layer" + str(features.layer) + ".npy"
                )
            )
        else:
            word_vecs[story], wordind2tokind[story] = features.make_stim(
                word_seqs[story].data,
                story=story,
            )
    word_mat = np.vstack([word_vecs[story] for story in stories])
    word_mean, word_std = word_mat.mean(0), word_mat.std(0)

    ds_vecs = {
        story: lanczosinterp2D(
            word_vecs[story],
            word_seqs[story].data_times[wordind2tokind[story]],
            word_seqs[story].tr_times,
        )
        for story in stories
    }
    ds_mat = np.vstack(
        [ds_vecs[story][5 + config.TRIM : -config.TRIM] for story in stories]
    )
    if tr_stats is None:
        r_mean, r_std = ds_mat.mean(0), ds_mat.std(0)
        r_std[r_std == 0] = 1
    else:
        r_mean, r_std = tr_stats
    ds_mat = np.nan_to_num(np.dot((ds_mat - r_mean), np.linalg.inv(np.diag(r_std))))
    if config.IS_PCA and features.model.llm in [
        "llama3",
        "llama3.1",
        "llama1b",
        "llama3b",
        "opt",
        "llama70b",
        "falcon",
        "falcon7b",
    ]:
        if len(stories) > 1:
            pca.fit(ds_mat)
            pca_path = (
                "/Storage2/anna/semantic-decoding_original/pca_model_%s.pkl"
                % features.model.llm
            ).replace("Storage2", config.WRITE_DIR)
            if not os.path.exists(pca_path):
                joblib.dump(pca, pca_path)
        ds_mat = pca.transform(ds_mat)
    del_mat = make_delayed(ds_mat, config.STIM_DELAYS)
    if tr_stats is None:
        return del_mat, (r_mean, r_std), (word_mean, word_std)
    else:
        return del_mat


def predict_word_rate(resp, wt, vox, mean_rate):
    """predict word rate at each acquisition time"""
    delresp = make_delayed(resp[:, vox], config.RESP_DELAYS)
    rate = ((delresp.dot(wt) + mean_rate)).reshape(-1).clip(min=0)
    return np.round(rate).astype(int)


def predict_word_times(word_rate, resp, starttime=0, tr=2):
    """predict evenly spaced word times from word rate"""
    half = tr / 2
    trf = TRFile(None, tr)
    trf.soundstarttime = starttime
    trf.simulate(resp.shape[0])
    tr_times = trf.get_reltriggertimes() + half

    word_times = []
    for mid, num in zip(tr_times, word_rate):
        if num < 1:
            continue
        word_times.extend(
            np.linspace(mid - half, mid + half, num, endpoint=False) + half / num
        )
    return np.array(word_times), tr_times
