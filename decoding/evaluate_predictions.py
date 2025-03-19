import os
import numpy as np
import json
import argparse
import inflect

import config
from utils_eval import (
    generate_null,
    load_transcript,
    windows,
    segment_data,
    WER,
    BLEU,
    METEOR,
    BERTSCORE,
)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--subject", type=str, required=True)
    parser.add_argument("--llm", type=str, required=True)
    parser.add_argument("--task", type=str, default="wheretheressmoke")
    parser.add_argument("--experiment", type=str, default="perceived_speech")
    parser.add_argument(
        "--metrics", nargs="+", type=str, default=["WER", "BLEU", "METEOR", "BERT"]
    )
    parser.add_argument("--references", nargs="+", type=str, default=[])
    parser.add_argument("--null", type=int, default=10)
    parser.add_argument("--format", action="store_true")
    parser.add_argument("--prompt", action="store_true")
    args = parser.parse_args()

    if len(args.references) == 0:
        args.references.append(args.task)

    with open(os.path.join(config.DATA_TEST_DIR, "eval_segments.json"), "r") as f:
        eval_segments = json.load(f)

    # load language similarity metrics
    metrics = {}
    if "WER" in args.metrics:
        metrics["WER"] = WER(use_score=True)
    if "BLEU" in args.metrics:
        metrics["BLEU"] = BLEU(n=1)
    if "METEOR" in args.metrics:
        metrics["METEOR"] = METEOR(mark=config.MARK[args.llm])
    if "BERT" in args.metrics:
        metrics["BERT"] = BERTSCORE(
            mark=config.MARK[args.llm],
            idf_sents=np.load(os.path.join(config.DATA_TEST_DIR, "idf_segments.npy")),
            rescale=False,
            score="recall",
        )
    suffix = "acl_prompt" if args.prompt else "acl"

    # load prediction transcript
    pred_path = os.path.join(
        config.RESULT_DIR,
        args.subject,
        args.experiment,
        args.task + "_" + args.llm + suffix + ".npz",
    )
    pred_data = np.load(pred_path)
    pred_words, pred_times = pred_data["words"], pred_data["times"]

    # generate null sequences
    if args.experiment in ["imagined_speech"]:
        gpt_checkpoint = "imagined"
    else:
        gpt_checkpoint = "perceived"
    save_location = os.path.join(config.SCORE_DIR, args.subject, args.experiment)
    if os.path.exists(
        os.path.join(save_location, args.task + "_" + args.llm)
        + suffix
        + "_format"
        + ".npz"
    ):
        print("EXIST!!!!")
        result = np.load(
            os.path.join(save_location, args.task + "_" + args.llm)
            + suffix
            + "_format"
            + ".npz"
        )
        null_word_list = result["null_word_list"].tolist()
    else:
        null_word_list = (
            generate_null(
                pred_times,
                gpt_checkpoint,
                args.null,
                args.llm,
                args.prompt,
                args.experiment,
            )
            if args.null
            else [[]]
        )
    if args.llm in ["gpt"]:
        pred_words = [w.replace("</w>", " ") for w in pred_words]
    window_scores, window_zscores = {}, {}
    story_scores, story_zscores = {}, {}
    window_null_scores, story_null_scores = {}, {}
    p = inflect.engine()
    for reference in args.references:

        # load reference transcript
        ref_data = load_transcript(args.experiment, reference)
        ref_words, ref_times = ref_data["words"], ref_data["times"]

        if args.format:
            print("FORMAT!")
            for c in [
                ".",
                '"',
                "?",
                "!",
                "”",
                "“",
                "âĢ",
                "ĺ",
                "ĵ",
                "\n",
                ":",
                "(",
                ")",
                ",",
                "#",
                "%",
                "+",
                "*",
                "_",
                "=",
                "/",
                ";",
                "[",
                "]",
            ]:
                replace = " " if c == "\n" else ""
                pred_words = [
                    (
                        p.number_to_words(int(word))
                        if word.isdecimal()
                        else word.lower().replace(c, replace)
                    )
                    for word in pred_words
                ]
                null_word_list = [
                    [
                        (
                            p.number_to_words(int(word))
                            if word.isdecimal()
                            else word.lower().replace(c, replace)
                        )
                        for word in null_words
                    ]
                    for null_words in null_word_list
                ]
        if args.format and (args.llm == "gpt"):
            rpl_lst = ["n't ", "'d ", "'ll ", "'s ", "'re ", "'m ", "'ve "]
            for i in range(len(pred_words)):
                if pred_words[i] in rpl_lst:
                    pred_words[i - 1] = pred_words[i - 1][:-1]
                for null_words in null_word_list:
                    if null_words[i] in rpl_lst:
                        null_words[i - 1] = null_words[i - 1][:-1]
        # segment prediction and reference words into windows
        window_cutoffs = windows(*eval_segments[args.task], config.WINDOW)
        ref_windows = segment_data(ref_words, ref_times, window_cutoffs)
        pred_windows = segment_data(pred_words, pred_times, window_cutoffs)
        null_window_list = [
            segment_data(null_words, pred_times, window_cutoffs)
            for null_words in null_word_list
        ]

        for mname, metric in metrics.items():
            if mname == "WER":
                window_null_scores[(reference, mname)] = np.array(
                    [
                        metric.score(
                            ref=[" ".join(ref) for ref in ref_windows],
                            pred=[
                                config.MARK[args.llm].join(null)
                                for null in null_windows
                            ],
                        )
                        for null_windows in null_window_list
                    ]
                )
                window_scores[(reference, mname)] = metric.score(
                    ref=[" ".join(ref) for ref in ref_windows],
                    pred=[config.MARK[args.llm].join(pred) for pred in pred_windows],
                )
            elif mname == "BLEU" and config.MARK[args.llm] == "":
                window_null_scores[(reference, mname)] = np.array(
                    [
                        metric.score(
                            ref=ref_windows,
                            # pred=[w for w in ["".join(null).split(" ") for null in null_windows] if w != ""],
                            pred=[
                                [w for w in "".join(null).split(" ") if w.strip()]
                                for null in null_windows
                            ],
                        )
                        for null_windows in null_window_list
                    ]
                )
                window_scores[(reference, mname)] = metric.score(
                    ref=ref_windows,
                    # pred=[w for w in ["".join(pred).split(" ") for pred in pred_windows] if w != ""],
                    pred=[
                        [w for w in "".join(pred).split(" ") if w.strip()]
                        for pred in pred_windows
                    ],
                )
            else:
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
            story_null_scores[(reference, mname)] = (
                window_null_scores[(reference, mname)].mean(1) if args.null else None
            )
    os.makedirs(save_location, exist_ok=True)
    np.savez(
        os.path.join(
            save_location,
            args.task + "_" + args.llm + suffix + ("_format" if args.format else ""),
        ),
        window_scores=window_scores,
        window_zscores=window_zscores,
        story_scores=story_scores,
        story_zscores=story_zscores,
        window_null_scores=window_null_scores,
        story_null_scores=story_null_scores,
        null_word_list=np.array(null_word_list),
    )
