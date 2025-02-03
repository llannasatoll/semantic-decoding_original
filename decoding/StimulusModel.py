import numpy as np
import torch
import joblib
import os

torch.set_default_tensor_type(torch.FloatTensor)

import config
from utils_ridge.interpdata import lanczosfun


def get_lanczos_mat(oldtime, newtime, window=3, cutoff_mult=1.0, rectify=False):
    """get matrix for downsampling from TR times to word times"""
    cutoff = 1 / np.mean(np.diff(newtime)) * cutoff_mult
    sincmat = np.zeros((len(newtime), len(oldtime)))
    for ndi in range(len(newtime)):
        sincmat[ndi, :] = lanczosfun(cutoff, newtime[ndi] - oldtime, window)
    return sincmat


def affected_trs(start_index, end_index, lanczos_mat, delay=True):
    """identify TRs influenced by words in the range [start_index, end_index]"""
    start_tr, end_tr = (
        np.where(lanczos_mat[:, start_index])[0][0],
        np.where(lanczos_mat[:, end_index])[0][-1],
    )
    start_tr, end_tr = start_tr + min(config.STIM_DELAYS), end_tr + max(
        config.STIM_DELAYS
    )
    start_tr, end_tr = max(start_tr, 0), min(end_tr, lanczos_mat.shape[0] - 1)
    return np.arange(start_tr, end_tr + 1)


class StimulusModel:
    """class for constructing stimulus features"""

    def __init__(self, lanczos_mat, tr_stats, word_mean, llm, device="cpu"):
        self.device = device
        self.llm = llm
        self.lanczos_mat = torch.from_numpy(lanczos_mat).float().to(self.device)
        self.tr_mean = torch.from_numpy(tr_stats[0]).float().to(device)
        self.tr_std_inv = torch.from_numpy(np.diag(1 / tr_stats[1])).float().to(device)
        self.blank = torch.from_numpy(word_mean).float().to(self.device)

    def _downsample(self, variants):
        """downsamples word embeddings to TR embeddings for each hypothesis"""
        return torch.matmul(self.lanczos_mat.unsqueeze(0), variants)

    def _normalize(self, tr_variants):
        """normalize TR embeddings for each hypothesis"""
        centered = tr_variants - self.tr_mean
        return torch.matmul(centered, self.tr_std_inv)

    def _delay(self, tr_variants, n_vars, n_feats):
        """apply finite impulse response delays to TR embeddings"""
        delays = config.STIM_DELAYS
        n_trs = tr_variants.shape[1]
        del_tr_variants = torch.zeros(n_vars, n_trs, len(delays) * n_feats)
        for c, d in enumerate(delays):
            feat_ind_start = c * n_feats
            feat_ind_end = (c + 1) * n_feats
            del_tr_variants[:, d:, feat_ind_start:feat_ind_end] = tr_variants[
                :, : n_trs - d, :
            ]
        return del_tr_variants

    def make_variants(self, sample_index, hypothesis_embs, var_embs, affected_trs):
        """create stimulus features for each hypothesis"""
        n_variants, n_feats = len(var_embs), self.blank.shape[0]
        with torch.no_grad():
            full = (
                self.blank.expand(self.lanczos_mat.shape[1], -1).clone().to("cuda:0")
            )  # word times x features
            hypothesis_embs_tensor = torch.from_numpy(np.array(hypothesis_embs)).to(
                "cuda:0"
            )
            full[:sample_index] = hypothesis_embs_tensor.reshape(-1, full.shape[1])
            # variants = full.expand(n_variants, -1, -1).to("cuda:1")
            full = full.to(self.device)
            variants = full.expand(n_variants, -1, -1).contiguous()
            var_embs_tensor = torch.from_numpy(np.array(var_embs)).to(self.device)
            variants[:, sample_index, :] = var_embs_tensor
            # full = self.blank.repeat(
            #     self.lanczos_mat.shape[1], 1
            # )  # word times x features
            # full[:sample_index] = (
            #     torch.tensor(np.array(hypothesis_embs))
            #     .float()
            #     .reshape(-1, n_feats)
            #     .to(self.device)
            # )
            # variants = full.repeat(n_variants, 1, 1)  # variants x word times x features
            # variants[:, sample_index, :] = (
            #     torch.from_numpy(var_embs).float().to(self.device)
            # )
            del var_embs_tensor, full
            torch.cuda.empty_cache()
            tr_variants = self._normalize(self._downsample(variants)).to("cpu")
            del variants
            torch.cuda.empty_cache()
            del_tr_variants = self._delay(
                tr_variants,
                n_variants,
                n_feats,
            )
        return del_tr_variants[:, affected_trs, :]


class LMFeatures:
    """class for extracting contextualized features of stimulus words"""

    def __init__(self, model, layer, context_words):
        self.model, self.layer, self.context_words = model, layer, context_words

    def extend(self, extensions, verbose=False, batch_size=10):
        """outputs array of vectors corresponding to the last words of each extension"""
        embs_list = []

        for i in range(0, len(extensions), batch_size):
            batch = extensions[i : i + batch_size]  # バッチごとに処理
            contexts = [ext[-(self.context_words + 1) :] for ext in batch]

            if verbose:
                print(contexts)

            context_array = self.model.get_context_array(contexts)
            embs = self.model.get_hidden(context_array, layer=self.layer)
            embs_list.append(embs[:, -1])  # 各バッチの結果を保存

        return np.vstack(embs_list)  # 結果を結合して返す
        return np.vstack(embs_list)
        contexts = [extension[-(self.context_words + 1) :] for extension in extensions]
        if verbose:
            print(contexts)
        context_array = self.model.get_context_array(contexts)
        embs = self.model.get_hidden(context_array, layer=self.layer)
        return embs[:, len(contexts[0]) - 1]

    def make_stim(self, words, story=None):
        """outputs matrix of features corresponding to the stimulus words"""
        path = os.path.join(
            config.DATA_TRAIN_DIR,
            "train_stimulus",
            "word_vecs_and_wordind2tokind",
            str(story) + self.model.llm + ".npz",
        ).replace("Storage2", config.WRITE_DIR)
        if False and os.path.exists(path):
            tmp = np.load(path)
            context_array, wordind2tokind = (
                tmp["context_array"],
                tmp["wordind2tokind"],
            )
            context_array = torch.tensor(context_array).long()
        else:
            context_array, wordind2tokind = self.model.get_story_array(
                words, self.context_words
            )
            if self.model.llm != "original" and (story is not None):
                np.savez(
                    path,
                    context_array=np.array(context_array),
                    wordind2tokind=np.array(wordind2tokind),
                )
        if self.context_words < 0:
            return None, wordind2tokind
        embs = self.model.get_hidden(context_array, layer=self.layer, story=story)
        torch.cuda.empty_cache()
        return embs, wordind2tokind
