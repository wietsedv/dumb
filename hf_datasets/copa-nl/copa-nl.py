import os
import json

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

# TODO: Add a link to an official homepage for the dataset here
_HOMEPAGE = ""

# TODO: Add the licence for the dataset here if you can find it
_LICENSE = "bsd-2-clause"

_URL = "https://github.com/wietsedv/NLP-NL/archive/refs/tags/copa-nl-v1.0.tar.gz"


class CopaNL(datasets.GeneratorBasedBuilder):
    VERSION = datasets.Version("1.0.0")

    def _info(self):
        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=datasets.Features({
                "id": datasets.Value("int32"),
                "premise": datasets.Value("string"),
                "choice1": datasets.Value("string"),
                "choice2": datasets.Value("string"),
                "question": datasets.ClassLabel(names=["cause", "effect"]),
                "label": datasets.ClassLabel(names=["choice1", "choice2"]),
            }),
            homepage=_HOMEPAGE,
            license=_LICENSE,
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager):
        data_dir = dl_manager.download_and_extract(_URL)
        data_dir = os.path.join(data_dir, "COPA-NL")  # type: ignore

        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,  # type: ignore
                gen_kwargs={
                    "filepath": os.path.join(data_dir, "train.jsonl"),
                },
            ),
            datasets.SplitGenerator(
                name=datasets.Split.VALIDATION,  # type: ignore
                gen_kwargs={
                    "filepath": os.path.join(data_dir, "dev.jsonl"),
                },
            ),
            datasets.SplitGenerator(
                name=datasets.Split.TEST,  # type: ignore
                gen_kwargs={
                    "filepath": os.path.join(data_dir, "test.jsonl"),
                },
            ),
        ]

    def _generate_examples(self, filepath):
        with open(filepath) as f:
            for line in f:
                ex = json.loads(line)
                yield ex["id"], ex
