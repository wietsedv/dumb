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

_HOMEPAGE = "https://github.com/gijswijnholds/sick_nl"

_LICENSE = "mit"

_URL = "https://raw.githubusercontent.com/gijswijnholds/sick_nl/master/data/tasks/sick_nl/SICK_NL.txt"


class SickNL(datasets.GeneratorBasedBuilder):
    VERSION = datasets.Version("1.0.0")

    def _info(self):
        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=datasets.Features(
                {
                    "pair_id": datasets.Value("string"),
                    "sentence1": datasets.Value("string"),
                    "sentence2": datasets.Value("string"),
                    "label": datasets.ClassLabel(names=["entailment", "neutral", "contradiction"]),
                    "relatedness_score": datasets.Value("float"),
                    "entailment_12": datasets.Value("string"),
                    "entailment_21": datasets.Value("string"),
                    "sentence1_original": datasets.Value("string"),
                    "sentence2_original": datasets.Value("string"),
                    "sentence1_dataset": datasets.Value("string"),
                    "sentence2_dataset": datasets.Value("string"),
                }),
            homepage=_HOMEPAGE,
            license=_LICENSE,
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager):
        filepath = dl_manager.download(_URL)

        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,  # type: ignore
                gen_kwargs={
                    "filepath": filepath,
                    "key": "TRAIN",
                },
            ),
            datasets.SplitGenerator(
                name=datasets.Split.VALIDATION,  # type: ignore
                gen_kwargs={
                    "filepath": filepath,
                    "key": "TRIAL",
                },
            ),
            datasets.SplitGenerator(
                name=datasets.Split.TEST,  # type: ignore
                gen_kwargs={
                    "filepath": filepath,
                    "key": "TEST",
                },
            ),
        ]

    def _generate_examples(self, filepath, key):
        with open(filepath, "r", encoding="utf-8") as f:
            for line in f:
                data = [s.strip() for s in line.split("\t")]
                if data[-1] == key:
                    yield data[0], {
                        "pair_id": data[0],
                        "sentence1": data[1],
                        "sentence2": data[2],
                        "label": data[3].lower(),
                        "relatedness_score": data[4],
                        "entailment_12": data[5],
                        "entailment_21": data[6],
                        "sentence1_original": data[7],
                        "sentence2_original": data[8],
                        "sentence1_dataset": data[9],
                        "sentence2_dataset": data[10],
                    }
