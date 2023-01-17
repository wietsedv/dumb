# TODO: Currently only the filtered human annotations. Maybe add a builder for superset?
# TODO: Currently only SoNaR  and no CGN. Add CGN?

from glob import glob
import os
from xml.etree import ElementTree

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

_HOMEPAGE = "http://wordpress.let.vupr.nl/dutchsemcor/"

# TODO: Add the licence for the dataset here if you can find it
_LICENSE = "cc-by-3.0"

_URL = "http://kyoto.let.vu.nl/dutchsemcor/annotated_data/human_filtered/1.2.2.HUMAN_ANNOTATIONS_FILTERED_FOR_TRAINING.zip"


class DutchSemCor(datasets.GeneratorBasedBuilder):
    VERSION = datasets.Version("1.2.2")

    def _info(self):
        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=datasets.Features({
                # "source": datasets.ClassLabel(names=SOURCES),
                "doc_id": datasets.Value("string"),
                "sent_id": datasets.Value("string"),
                "lemma": datasets.Value("string"),
                "pos": datasets.ClassLabel(names=["A", "N", "V"]),
                "sense": datasets.Value("string"),
                "index": datasets.Value("int32"),
                "tokens": datasets.Sequence(datasets.Value("string")),
            }),
            homepage=_HOMEPAGE,
            license=_LICENSE,
            citation=_CITATION,
        )

    @property
    def manual_download_instructions(self):
        return (
            "To use SoNaR you have to download it manually. You can download the corpus at https://taalmaterialen.ivdnt.org/?__wpdmlo=1452"
            " Please extract the .tgz file in one folder and load the dataset with: "
            "`datasets.load_dataset('hf_datasets/dutch-sem-cor', data_dir='path/to/folder/SoNaR-1.2.1')`")

    def _split_generators(self, dl_manager):
        data_dir = dl_manager.download_and_extract(_URL)
        data_dir = os.path.join(data_dir, "1.2.2.HUMAN_ANNOTATIONS_FILTERED_FOR_TRAINING")  # type: ignore

        filepaths = [
            os.path.join(data_dir, "annotations.training.DSC.adjs.xml"),
            os.path.join(data_dir, "annotations.training.DSC.nouns.xml"),
            os.path.join(data_dir, "annotations.training.DSC.verbs.xml"),
        ]
        dataset = datasets.load_dataset("hf_datasets/sonar-500", "dutch-sem-cor", data_dir=dl_manager.manual_dir, split="all")

        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,  # type: ignore
                gen_kwargs={
                    "filepaths": filepaths,
                    "dataset": dataset,
                },
            ),
        ]

    def _generate_examples(self, filepaths, dataset):
        sents = {x["sent_id"]: x for x in dataset}

        key = 0
        for filepath in filepaths:
            root = ElementTree.parse(filepath).getroot()
            assert root

            examples = [token.attrib for token in root]
            for ex in examples:
                doc_id, _ = ex["token_id"].split(".", maxsplit=1)
                sent_id, _ = ex["token_id"].split(".w.")
                if sent_id not in sents:
                    continue

                token_ids = sents[sent_id]["token_ids"]
                tokens = sents[sent_id]["tokens"]

                yield key, {
                    # "source": SONAR500_COMPONENTS[doc_id[:8]],
                    "doc_id": doc_id,
                    "sent_id": sent_id,
                    "lemma": ex["lemma"],
                    "pos": ex["pos"].upper(),
                    "sense": ex["sense"],
                    "index": token_ids.index(ex["token_id"]),
                    "tokens": tokens,
                }
                key += 1
