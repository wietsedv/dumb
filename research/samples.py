""" Show all scores per task/model/config """

from argparse import ArgumentParser
from pathlib import Path
import json
from glob import glob
from typing import Dict, Any
import numpy as np
import pandas as pd

from constants import TASK_ORDER, MODEL_ORDER, CONFIG_KEYS, EVAL_SAMPLES, get_test_seeds
from best import load_results, find_best_results


def find_aux_runs(task, model, res, test: bool):
    results = {}
    for samples in EVAL_SAMPLES[task]:
        res_dict = {}
        for seed in get_test_seeds(task, model):
            e = res["num_train_epochs"]
            w = res["warmup_ratio"]
            b = res["train_batch_size"]
            l = res["learning_rate"]
            d = res["hidden_dropout_prob"]
            c = res["weight_decay"]
            config = f"e{e}-w{w}-b{b}-l{l}-d{d}-c{c}-m{samples}-s{seed}"
            path = Path("output") / task / model / config / "all_results.json"
            if not path.exists():
                continue
            with open(path) as f:
                aux_res = json.load(f)
            metric = f"predict_{res['metric']}" if test else f"eval_{res['metric']}"
            res_dict[seed] = aux_res[metric] * 100
        results[samples] = res_dict
    return results


def summarize_samples(tasks, models, best_results, test):
    print("| task | model | samples | runs | score | rel score |")
    print("| --- | --- | ---: | ---: | ---: | ---: |")
    missing = []
    for task in tasks:
        for model in models:
            if (task, model) not in best_results:
                continue

            res = best_results[(task, model)]
            aux_runs = find_aux_runs(task, model, res, test)

            target_score = res["test_score_mean"] if test else res["dev_score"]
            
            for samples in sorted(aux_runs.keys()):
                n = len(aux_runs[samples])
                scores = list(aux_runs[samples].values())
                mean = np.mean(scores)
                std = np.std(scores)
                rel_score = mean / target_score * 100
                print(f"| {task} | {model} | {samples} | {n} | {mean:.1f} <sub>s={std:.1f}</sub> | {rel_score:.1f}% |")


    return missing



def main():
    parser = ArgumentParser()
    parser.add_argument("-t", "--tasks", nargs="*", default=TASK_ORDER)
    parser.add_argument("-m", "--models", nargs="*", default=MODEL_ORDER)
    parser.add_argument("--test", action="store_true")
    args = parser.parse_args()

    results = load_results(args.tasks, args.models, samples=0, all=False)
    best_results = find_best_results(results)

    summarize_samples(args.tasks, args.models, best_results, args.test)


if __name__ == "__main__":
    main()
