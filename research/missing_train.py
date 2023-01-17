""" Show all scores per task/model/config """

from argparse import ArgumentParser
from pathlib import Path
import json
from glob import glob
from typing import Dict, Any
import numpy as np
import pandas as pd

from constants import get_train_params, TASK_ORDER, MODEL_ORDER, EVAL_SEED


def find_missing_runs(tasks, models):
    missing = []
    for task in tasks:
        params = get_train_params(task)
        for model in models:
            for e in params["e"]:
                for w in params["w"]:
                    for b in params["b"]:
                        for l in params["l"]:
                            l = np.format_float_positional(l)
                            for d in params["d"]:
                                for c in params["c"]:
                                    config = f"e{e}-w{w}-b{b}-l{l}-d{d}-c{c}-m0-s{EVAL_SEED}"
                                    path = Path("output") / task / model / config / "eval_results.json"
                                    if not path.exists():
                                        missing.append((task, {
                                            "model": model,
                                            "num_train_epochs": e,
                                            "warmup_ratio": w,
                                            "train_batch_size": b,
                                            "learning_rate": l,
                                            "hidden_dropout_prob": d,
                                            "weight_decay": c,
                                            "seed": EVAL_SEED,
                                        }))
    return missing


def main():
    parser = ArgumentParser()
    parser.add_argument("-t", "--tasks", nargs="*", default=TASK_ORDER)
    parser.add_argument("-m", "--models", nargs="*", default=MODEL_ORDER)
    args = parser.parse_args()

    missing = find_missing_runs(args.tasks, args.models)
    if len(missing) == 0:
        print("# all done!")
        exit(0)

    for task, env in missing:
        print(" ".join([f"{key}={val}" for key, val in env.items()]) + f" ./tasks/{task}.sh")
    print(f"\n# {len(missing)} runs")

    

if __name__ == "__main__":
    main()
