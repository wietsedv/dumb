""" Show all scores per task/model/config """

from argparse import ArgumentParser
from pathlib import Path
import json
from glob import glob
from typing import Dict, Any
import numpy as np
import pandas as pd

from constants import TASK_ORDER, MODEL_ORDER, get_test_seeds
from best import load_results, find_best_results


def show_summary(best_results: dict, tasks, models):
    rows = []
    for task in tasks:
        for model in models:
            if (task, model) not in best_results:
                continue
            test_seeds = get_test_seeds(task, model)
            res = {seed: score for seed, score in zip(test_seeds, best_results[(task, model)]["eval_scores"])}
            mean = np.mean(best_results[(task, model)]["eval_scores"])
            std = np.std(best_results[(task, model)]["eval_scores"])
            row = {"task": task, "model": model, **res, "mean": mean, "std": std}
            rows.append(row)
    df = pd.DataFrame(rows).round(1)
    pd.set_option('display.max_rows', None)
    print(df)


def main():
    parser = ArgumentParser()
    parser.add_argument("-t", "--tasks", nargs="*", default=TASK_ORDER)
    parser.add_argument("-m", "--models", nargs="*", default=MODEL_ORDER)
    args = parser.parse_args()

    results = load_results(args.tasks, args.models, samples=0, all=False)
    best_results = find_best_results(results)

    show_summary(best_results, args.tasks, args.models)


if __name__ == "__main__":
    main()
