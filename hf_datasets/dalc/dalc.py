import os
import csv

import datasets

# TODO: Add BibTeX citation
# Find for instance the citation on arxiv or on the dataset repo/website
_CITATION = """\
@InProceedings{huggingface:dataset,
title = {A great new dataset},
author={huggingface, Inc.
},
year={2020}
}
"""

# TODO: Add description of the dataset here
# You can copy an official description
_DESCRIPTION = """\
This new dataset is designed to solve this great NLP task and is crafted with a lot of care.
"""

_HOMEPAGE = "https://github.com/tommasoc80/DALC"

_LICENSE = "gpl-3.0"

# _URLS = {
#     "train": "https://raw.githubusercontent.com/tommasoc80/DALC/master/v2.0/data/DALC-2_train.csv",
#     "dev": "https://raw.githubusercontent.com/tommasoc80/DALC/master/v2.0/data/DALC-2_dev.csv",
#     "test": "https://raw.githubusercontent.com/tommasoc80/DALC/master/v2.0/data/DALC-2_test.csv",
# }
# https://docs.google.com/forms/d/1B8kglH6TOOTzMVVyQJAz5PjPPL4NDD_-8WpZT4vvoBw/viewform?edit_requested=true
# https://drive.google.com/drive/folders/1MoghO5709wr_OzKv-ojFvBnXxmpZ0i6n?usp=sharing


class DALC(datasets.GeneratorBasedBuilder):
    VERSION = datasets.Version("2.0.0")

    def _info(self):
        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=datasets.Features({
                "id":
                datasets.Value("string"),
                "text":
                datasets.Value("string"),
                "label":
                datasets.ClassLabel(names=["not", "abusive", "offensive"]),
                "explicitness_label":
                datasets.ClassLabel(names=["not", "explicit", "implicit"]),
                "target_label":
                datasets.ClassLabel(names=["not", "group", "individual", "other"]),
                # "label":
                # datasets.ClassLabel(names=[
                #     "not",
                #     "abusive-explicit",
                #     "abusive-implicit",
                #     "offensive-explicit",
                #     "offensive-implicit",
                # ]),
            }),
            homepage=_HOMEPAGE,
            license=_LICENSE,
            citation=_CITATION,
        )

    @property
    def manual_download_instructions(self):
        return (
            "To use DALC, download the full dataset from "
            "https://docs.google.com/forms/d/1B8kglH6TOOTzMVVyQJAz5PjPPL4NDD_-8WpZT4vvoBw/viewform?edit_requested=true"
            "and load with `datasets.load_dataset('hf_datasets/dalc', data_dir='path/to/folder')`")

    def _split_generators(self, dl_manager):
        # filepaths = dl_manager.download(_URLS)
        assert dl_manager.manual_dir
        data_dir = os.path.abspath(os.path.expanduser(dl_manager.manual_dir))

        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,  # type: ignore
                gen_kwargs={
                    "filepath": os.path.join(data_dir, "DALC-2_train_full.csv"),
                },
            ),
            datasets.SplitGenerator(
                name=datasets.Split.VALIDATION,  # type: ignore
                gen_kwargs={
                    "filepath": os.path.join(data_dir, "DALC-2_dev_full.csv"),
                },
            ),
            datasets.SplitGenerator(
                name=datasets.Split.TEST,  # type: ignore
                gen_kwargs={
                    "filepath": os.path.join(data_dir, "DALC-2_test_full.csv"),
                },
            ),
        ]

    def _generate_examples(self, filepath):
        with open(filepath, "r", newline="", encoding="utf-8") as f:
            next(f)
            for data in csv.reader(f, delimiter=","):
                explicitness = data[3].lower()
                target = data[4].lower() if data[4] else "not"
                abusive_offensive = data[5].lower()
                # label = "not" if abusive_offensive == "not" else f"{abusive_offensive}-{explicitness}-{target}"
                # label = "not" if abusive_offensive == "not" else f"{abusive_offensive}-{explicitness}"
                yield data[0], {
                    "id": data[0],
                    "text": data[1],
                    "label": abusive_offensive,
                    "target_label": target,
                    "explicitness_label": explicitness,
                }
