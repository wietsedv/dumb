from glob import glob
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

_HOMEPAGE = "https://github.com/benjaminvdb/DBRD"

_LICENSE = "mit"

_URL = "https://github.com/benjaminvdb/DBRD/releases/download/v3.0/DBRD_v3.tgz"


class DBRD(datasets.GeneratorBasedBuilder):
    VERSION = datasets.Version("3.0.0")

    def _info(self):
        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=datasets.Features(
                {
                    "doc_id": datasets.Value("string"),
                    "text": datasets.Value("string"),
                    "label": datasets.ClassLabel(names=["negative", "positive"]),
                }),
            homepage=_HOMEPAGE,
            license=_LICENSE,
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager):
        data_dir = dl_manager.download_and_extract(_URL)

        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,  # type: ignore
                gen_kwargs={
                    "data_dir": os.path.join(data_dir, "DBRD", "train"),  # type: ignore
                },
            ),
            datasets.SplitGenerator(
                name=datasets.Split.TEST,  # type: ignore
                gen_kwargs={
                    "data_dir": os.path.join(data_dir, "DBRD", "test"),  # type: ignore
                },
            ),
        ]

    def _generate_examples(self, data_dir):
        print(data_dir)
        for key, filepath in enumerate(glob(os.path.join(data_dir, "*", "*.txt"))):
            doc_id = os.path.basename(filepath).removesuffix(".txt")
            label = filepath.split("/")[-2]
            assert label in ["pos", "neg"]
            with open(filepath, encoding="utf-8") as f:
                text = f.read().strip()
            yield key, {
                "doc_id": doc_id,
                "text": text,
                "label": "positive" if label == "pos" else "negative",
            }
