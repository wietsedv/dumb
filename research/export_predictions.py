""" Show all scores per task/model/config """

from argparse import ArgumentParser
from pathlib import Path
import numpy as np
import os

from constants import TASK_ORDER, MODEL_ORDER, get_test_seeds
from best import load_results, find_best_results


def find_paths(tasks, models, best_results):
    paths = []
    for task in tasks:
        for model in models:
            if (task, model) not in best_results:
                print(f"{task} {model} is missing")
                exit(1)

            res = best_results[(task, model)]

            test_seeds = get_test_seeds(task, model)
            assert len(test_seeds) == len(res["eval_seeds"]), f"seeds missing for {task} {model}"

            paths.append((task, model, res["eval_paths"]))
    return paths



def main():
    parser = ArgumentParser()
    parser.add_argument("-t", "--tasks", nargs="*", default=TASK_ORDER)
    parser.add_argument("-m", "--models", nargs="*", default=MODEL_ORDER)
    args = parser.parse_args()

    results = load_results(args.tasks, args.models, samples=0, all=False)
    best_results = find_best_results(results)

    for task, model, src_dirs in find_paths(args.tasks, args.models, best_results):
        tgt_dir = tgt_path = Path("exports") / "predictions" / task / model
        os.makedirs(tgt_dir, exist_ok=True)

        # TODO create ensemble
        for i, src_dir in enumerate(src_dirs, start=1):
            src_path = src_dir / "predictions.txt"
            tgt_path = tgt_dir / f"{i}.txt"
            with open(src_path) as f:
                preds = f.read()
            with open(tgt_path, "w") as f:
                f.write(preds)
            print(f"{src_path} => {tgt_path}")

if __name__ == "__main__":
    main()
