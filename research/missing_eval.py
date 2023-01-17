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


def find_missing_aux_runs(task, model, seed, res):
    missing = []
    if task not in EVAL_SAMPLES:
        return missing
    for samples in EVAL_SAMPLES[task]:
        e = res["num_train_epochs"]
        w = res["warmup_ratio"]
        b = res["train_batch_size"]
        l = np.format_float_positional(res["learning_rate"])
        d = res["hidden_dropout_prob"]
        c = res["weight_decay"]
        config = f"e{e}-w{w}-b{b}-l{l}-d{d}-c{c}-m{samples}-s{seed}"
        path = Path("output") / task / model / config / "eval_results.json"
        if not path.exists():
            env = {"model": model, "seed": seed, **{k: res[k] for k in CONFIG_KEYS.values()}}
            env["max_train_samples"] = samples
            missing.append((task, env))
    return missing


def find_missing_runs(tasks, models, best_results, aux: bool):
    missing = []
    for task in tasks:
        for model in models:
            if (task, model) not in best_results:
                print(f"# skipping {task} {model}")
                continue

            res = best_results[(task, model)]

            test_seeds = get_test_seeds(task, model)
            for seed in test_seeds:
                if seed not in res["eval_seeds"]:
                    env = {"model": model, "seed": seed, **{k: res[k] for k in CONFIG_KEYS.values()}}
                    missing.append((task, env))
                if aux:
                    missing.extend(find_missing_aux_runs(task, model, seed, res))

    return missing



def main():
    parser = ArgumentParser()
    parser.add_argument("-t", "--tasks", nargs="*", default=TASK_ORDER)
    parser.add_argument("-m", "--models", nargs="*", default=MODEL_ORDER)
    parser.add_argument("--aux", action="store_true")
    args = parser.parse_args()

    results = load_results(args.tasks, args.models, samples=0, all=False)
    best_results = find_best_results(results)

    missing = find_missing_runs(args.tasks, args.models, best_results, args.aux)
    for task, env in missing:
        env["learning_rate"] = np.format_float_positional(env["learning_rate"])
        print(" ".join([f"{key}={val}" for key, val in env.items()]) + f" ./tasks/{task}.sh")
    print(f"\n# {len(missing)} runs")


if __name__ == "__main__":
    main()
