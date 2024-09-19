import os
import numpy as np
import json
import argparse

import config
from utils_eval import (
    load_transcript,
    windows,
    segment_data,
    BERTSCORE,
)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--subject", type=str, required=True)
    parser.add_argument("--llm", type=str, required=True)
    parser.add_argument("--task", type=str, default="wheretheressmoke")
    parser.add_argument("--experiment", type=str, default="perceived_speech")
    parser.add_argument("--metrics", nargs="+", type=str, default=["BERT"])
    parser.add_argument("--references", nargs="+", type=str, default=[])
    args = parser.parse_args()

    if len(args.references) == 0:
        args.references.append(args.task)

    with open(os.path.join(config.DATA_TEST_DIR, "eval_segments.json"), "r") as f:
        eval_segments = json.load(f)

    # load language similarity metrics
    metrics = {}
    if "BERT" in args.metrics:
        metrics["BERT"] = BERTSCORE(
            mark=config.MARK[args.llm],
            idf_sents=np.load(os.path.join(config.DATA_TEST_DIR, "idf_segments.npy")),
            rescale=False,
            score="recall",
            large=True,
        )

    # load prediction transcript
    pred_path = os.path.join(
        config.RESULT_DIR,
        args.subject,
        args.experiment,
        args.task + "_" + args.llm + ".npz",
    )
    pred_data = np.load(pred_path)
    pred_words, pred_times = pred_data["words"], pred_data["times"]

    # generate null sequences
    if args.experiment in ["imagined_speech"]:
        gpt_checkpoint = "imagined"
    else:
        gpt_checkpoint = "perceived"

    save_location = os.path.join(
        config.SCORE_DIR,
        args.subject,
        args.experiment,
        args.task + "_" + args.llm + ".npz",
    )
    null_word_list = np.load(save_location)["null_word_list"].tolist()

    window_scores, window_zscores = {}, {}
    story_scores, story_zscores = {}, {}
    window_null_scores, story_null_scores = {}, {}
    for reference in args.references:

        # load reference transcript
        ref_data = load_transcript(args.experiment, reference)
        ref_words, ref_times = ref_data["words"], ref_data["times"]

        # segment prediction and reference words into windows
        window_cutoffs = windows(*eval_segments[args.task], config.WINDOW)
        ref_windows = segment_data(ref_words, ref_times, window_cutoffs)
        pred_windows = segment_data(pred_words, pred_times, window_cutoffs)
        null_window_list = [
            segment_data(null_words, pred_times, window_cutoffs)
            for null_words in null_word_list
        ]

        for mname, metric in metrics.items():
            window_null_scores[(reference, mname)] = np.array(
                [
                    metric.score(ref=ref_windows, pred=null_windows)
                    for null_windows in null_window_list
                ]
            )
            window_scores[(reference, mname)] = metric.score(
                ref=ref_windows, pred=pred_windows
            )
        story_scores[(reference, mname)] = window_scores[(reference, mname)].mean()
        story_null_scores[(reference, mname)] = window_null_scores[
            (reference, mname)
        ].mean(1)
    np.savez(
        save_location.replace(".npz", "_large.npz"),
        window_scores=window_scores,
        window_zscores=window_zscores,
        story_scores=story_scores,
        story_zscores=story_zscores,
        window_null_scores=window_null_scores,
        story_null_scores=story_null_scores,
    )
