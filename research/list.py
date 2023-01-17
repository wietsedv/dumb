""" Show all scores per task/model/config """

from argparse import ArgumentParser
from pathlib import Path
import json
from glob import glob
import pandas as pd


CONFIG_KEYS = {
    "e": "epochs",
    "w": "warmup",
    "b": "batchsize",
    "l": "lr",
    "d": "dropout",
    "m": "samples",
    "s": "seed",
}


def main():
    parser = ArgumentParser()
    parser.add_argument("-t", "--task", default="*")
    parser.add_argument("-m", "--model", default="*")
    parser.add_argument("-c", "--config", default="*")
    parser.add_argument("-s", "--sort", nargs="*", default=["task", "model", "seed"])
    parser.add_argument("--path", action="store_true")
    parser.add_argument("--test", action="store_true")
    args = parser.parse_args()

    split = "predict" if args.test else "eval"

    rows = []
    for path in glob(f"output/{args.task}/{args.model}/{args.config}/{split}_results.json"):
        path = Path(path)
        with open(path) as f:
            res = json.load(f)
        res = {k.replace(f"{split}_", ""): v for k, v in res.items() if k != "epoch" and k.replace(split, "") not in ["_loss", "_runtime", "_samples_per_second", "_steps_per_second"]}
        cfg = {CONFIG_KEYS.get(c[0], c[0]): float(c[1:]) for c in path.parent.name.split("-")}
        cfg["batchsize"] = int(cfg["batchsize"])
        cfg["seed"] = int(cfg["seed"])
        cfg["samples"] = -1 if cfg["samples"] == 0.0 else int(cfg["samples"])
        row = {"task": path.parent.parent.parent.name, "model": path.parent.parent.name, **cfg, **res}
        if args.path:
            row["config"] = path.parent.name
        rows.append(row)

    df = pd.DataFrame(sorted(rows, key=lambda x: [x[k] for k in args.sort if k in x]))
    pd.set_option('display.max_rows', None)
    # pd.set_option('display.max_columns', 500)
    print(df)


if __name__ == "__main__":
    main()
