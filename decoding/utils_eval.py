import os
import numpy as np
import json
import h5py

import config
from GPT import GPT
from Decoder import Decoder, Hypothesis
from LanguageModel import LanguageModel
from EncodingModel import EncodingModel
from StimulusModel import StimulusModel, get_lanczos_mat, affected_trs, LMFeatures
from utils_stim import predict_word_rate, predict_word_times

from jiwer import wer
from datasets import load_metric
from bert_score import BERTScorer

BAD_WORDS_PERCEIVED_SPEECH = frozenset(
    ["sentence_start", "sentence_end", "br", "lg", "ls", "ns", "sp"]
)
BAD_WORDS_OTHER_TASKS = frozenset(["", "sp", "uh"])

from utils_ridge.textgrid import TextGrid


def load_transcript(experiment, task):
    if experiment in ["perceived_speech", "perceived_multispeaker"]:
        skip_words = BAD_WORDS_PERCEIVED_SPEECH
    else:
        skip_words = BAD_WORDS_OTHER_TASKS
    grid_path = os.path.join(
        config.DATA_TEST_DIR,
        "test_stimulus",
        experiment,
        task.split("_")[0] + ".TextGrid",
    )
    transcript_data = {}
    with open(grid_path) as f:
        grid = TextGrid(f.read())
        if experiment == "perceived_speech":
            transcript = grid.tiers[1].make_simple_transcript()
        else:
            transcript = grid.tiers[0].make_simple_transcript()
        transcript = [
            (float(s), float(e), w.lower())
            for s, e, w in transcript
            if w.lower().strip("{}").strip() not in skip_words
        ]
    transcript_data["words"] = np.array([x[2] for x in transcript])
    transcript_data["times"] = np.array([(x[0] + x[1]) / 2 for x in transcript])
    return transcript_data


"""windows of [duration] seconds at each time point"""


def windows(start_time, end_time, duration, step=1):
    start_time, end_time = int(start_time), int(end_time)
    half = int(duration / 2)
    return [
        (center - half, center + half)
        for center in range(start_time + half, end_time - half + 1)
        if center % step == 0
    ]


"""divide [data] into list of segments defined by [cutoffs]"""


def find_split_index(segment, start, end, step, words):
    blank_count = 0
    for i in range(start, end, step):
        if " " in segment[i]:
            blank_count += 1
        if blank_count == words:
            return i
    return end  # Fallback in case the loop doesn't break


def segment_data(data, times, cutoffs):
    return [
        [x for c, x in zip(times, data) if c >= start and c < end]
        for start, end in cutoffs
    ]


# def segment_data(data, times, cutoffs):
#     segments = [
#         [x for c, x in zip(times, data) if c >= start and c < end]
#         for start, end in cutoffs
#     ]
#     result = []
#     words = 24
#     for segment in segments:
#         mid = len(segment) // 2
#         start = find_split_index(segment, mid, 0, -1, words)
#         end = find_split_index(segment, mid + 1, len(segment), 1, words)
#         result.append(segment[start:end])
#     return result


def get_em_environ(llm, subject):
    # load responses
    hf = h5py.File(
        os.path.join(
            config.DATA_TEST_DIR,
            "test_response",
            subject,
            "perceived_speech",
            "wheretheressmoke" + ".hf5",
        ),
        "r",
    )
    resp = np.nan_to_num(hf["data"][:])
    hf.close()

    load_location = os.path.join(config.MODEL_DIR, subject)
    encoding_model = np.load(
        os.path.join(load_location, llm, "encoding_model_%s.npz" % "perceived")
    )
    word_rate_model = np.load(
        os.path.join(load_location, llm, "word_rate_model_%s.npz" % "auditory"),
        allow_pickle=True,
    )
    weights = encoding_model["weights"]
    noise_model = encoding_model["noise_model"]
    tr_stats = encoding_model["tr_stats"]
    word_stats = encoding_model["word_stats"]
    em = EncodingModel(
        resp, weights, encoding_model["voxels"], noise_model, device=config.EM_DEVICE
    )
    em.set_shrinkage(config.NM_ALPHA)

    # predict word times
    word_rate = predict_word_rate(
        resp,
        word_rate_model["weights"],
        word_rate_model["voxels"],
        word_rate_model["mean_rate"],
    )
    word_times, tr_times = predict_word_times(word_rate, np.zeros(291), starttime=-10)
    lanczos_mat = get_lanczos_mat(word_times, tr_times)
    sm = StimulusModel(
        lanczos_mat,
        tr_stats,
        word_stats[0],
        llm,
        device=config.SM_DEVICE,
    )
    return sm, em, lanczos_mat


"""generate null sequences with same times as predicted sequence"""


def generate_null(pred_times, gpt_checkpoint, n, llm):
    # load language model
    gpt = GPT(
        llm=llm,
        device=config.GPT_DEVICE,
        gpt=gpt_checkpoint,
    )
    lm = LanguageModel(
        gpt, gpt.vocab, nuc_mass=config.LM_MASS, nuc_ratio=config.LM_RATIO
    )
    # generate null sequences
    null_words = []
    for _count in range(n):
        print(f"{_count}/{n}")
        decoder = Decoder(pred_times, 2 * config.EXTENSIONS)
        for sample_index in range(len(pred_times)):
            ncontext = decoder.time_window(sample_index, config.LM_TIME, floor=5)
            beam_nucs = lm.beam_propose(decoder.beam, ncontext)
            for c, (hyp, nextensions) in enumerate(decoder.get_hypotheses()):
                nuc, logprobs = beam_nucs[c]
                if len(nuc) < 1:
                    continue
                likelihoods = np.random.random(len(nuc))
                local_extensions = [
                    Hypothesis(parent=hyp, extension=x)
                    for x in zip(nuc, logprobs, [np.zeros(1) for _ in nuc])
                ]
                decoder.add_extensions(local_extensions, likelihoods, nextensions)
            decoder.extend(verbose=False)
        null_words.append(decoder.beam[0].words)
    return [gpt.decode_misencoded_text(null) for null in null_words]


"""
WER
"""


class WER(object):
    def __init__(self, use_score=True):
        self.use_score = use_score

    def score(self, ref, pred):
        scores = []
        print(ref[:2])
        print(pred[:2])
        for ref_seg, pred_seg in zip(ref, pred):
            if len(ref_seg) == 0:
                error = 1.0
            else:
                error = wer(ref_seg, pred_seg)
            if self.use_score:
                scores.append(1 - error)
            else:
                scores.append(error)
        return np.array(scores)


"""
BLEU (https://aclanthology.org/P02-1040.pdf)
"""


class BLEU(object):
    def __init__(self, n=4):
        self.metric = load_metric("bleu", keep_in_memory=True, trust_remote_code=True)
        self.n = n

    def score(self, ref, pred):
        results = []
        print(ref[:2])
        print(pred[:2])
        for r, p in zip(ref, pred):
            self.metric.add_batch(predictions=[p], references=[[r]])
            results.append(self.metric.compute(max_order=self.n)["bleu"])
        return np.array(results)


"""
METEOR (https://aclanthology.org/W05-0909.pdf)
"""


class METEOR(object):
    def __init__(self, mark):
        self.metric = load_metric("meteor", keep_in_memory=True)
        self.mark = mark

    def score(self, ref, pred):
        results = []
        ref_strings = [" ".join(x) for x in ref]
        pred_strings = [self.mark.join(x) for x in pred]
        print(ref_strings[:2])
        print(pred_strings[:2])
        for r, p in zip(ref_strings, pred_strings):
            self.metric.add_batch(predictions=[p], references=[r])
            results.append(self.metric.compute()["meteor"])
        return np.array(results)


"""
BERTScore (https://arxiv.org/abs/1904.09675)
"""


class BERTSCORE(object):
    def __init__(self, mark, idf_sents=None, rescale=True, score="f", large=False):
        self.mark = mark
        self.metric = BERTScorer(
            lang="en",
            rescale_with_baseline=rescale,
            idf=(idf_sents is not None),
            idf_sents=idf_sents,
            model_type="microsoft/deberta-xlarge-mnli" if large else "roberta-large",
        )
        if score == "precision":
            self.score_id = 0
        elif score == "recall":
            self.score_id = 1
        else:
            self.score_id = 2

    def score(self, ref, pred):
        ref_strings = [" ".join(x) for x in ref]
        pred_strings = [self.mark.join(x) for x in pred]
        print(ref_strings[:2])
        print(pred_strings[:2])
        # print([len(sent.split(" ")) for sent in ref_strings])
        # print([len(sent.split(" ")) for sent in pred_strings])
        return self.metric.score(cands=pred_strings, refs=ref_strings)[
            self.score_id
        ].numpy()
