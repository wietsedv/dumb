""" Show all scores per task/model/config """

from argparse import ArgumentParser
from pathlib import Path
import numpy as np
import os
from datasets import load_dataset, Dataset

from constants import TASK_ORDER


def read_gold(task):
    dataset = load_dataset("hf_datasets/dumb", task, split="test", data_dir="dumb", keep_in_memory=True)
    assert isinstance(dataset, Dataset)

    label_col = "label"
    is_seq, has_labels = False, True
    if task in ["lassy-pos", "sonar-ne"]:
        label_col = "tags"
        is_seq = True
    elif task == "copanl":
        has_labels = False

    id2label = None
    if has_labels:
        if is_seq:
            id2label = dataset.features[label_col].feature.names
        else:
            id2label = dataset.features[label_col].names

    lines = []
    for row in dataset:
        label = row[label_col] # type: ignore
        if id2label is not None:
            if is_seq:
                label = " ".join([id2label[i] for i in label])
            else:
                label = id2label[label]
        lines.append(str(label))
    return lines

def main():
    parser = ArgumentParser()
    parser.add_argument("-t", "--tasks", nargs="*", default=TASK_ORDER)
    args = parser.parse_args()

    for task in args.tasks:
        path = Path("exports") / "predictions" / task / "gold.txt"
        os.makedirs(path.parent, exist_ok=True)
        with open(path, "w") as f:
            for line in read_gold(task):
                f.write(line + "\n")
        print(path)

if __name__ == "__main__":
    main()
