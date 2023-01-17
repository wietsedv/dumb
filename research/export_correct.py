""" Show all scores per task/model/config """

from argparse import ArgumentParser
from pathlib import Path
import numpy as np
import os

from constants import TASK_ORDER, MODEL_ORDER


def check_correct(task, true, model, run):
    path = Path("exports") / "predictions" / task /  model / f"{run}.txt"

    pred = []
    with open(path) as f:
        for line in f:
            pred.append(line.rstrip().split())
    
    assert len(true) == len(pred)
    correct = [["1" if t == p else "0" for t, p in zip(T, P)] for T, P in zip(true, pred)]
    return correct


def main():
    parser = ArgumentParser()
    parser.add_argument("-t", "--tasks", nargs="*", default=TASK_ORDER)
    parser.add_argument("-m", "--models", nargs="*", default=MODEL_ORDER)
    args = parser.parse_args()

    for task in args.tasks:
        true = []
        with open(Path("exports") / "predictions" / task / "gold.txt") as f:
            for line in f:
                true.append(line.rstrip().split())

        for model in args.models:
            tgt_dir = Path("exports") / "correct" / task / model
            os.makedirs(tgt_dir, exist_ok=True)

            for run in range(1, 6):
                path = tgt_dir / f"{run}.txt"
                with open(path, "w") as f:
                    for row in check_correct(task, true, model, run):
                        f.write(" ".join(row) + "\n")
                print(path)


if __name__ == "__main__":
    main()
