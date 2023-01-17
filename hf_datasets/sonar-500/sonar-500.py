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

_HOMEPAGE = "https://lands.let.ru.nl/projects/SoNaR/"

_LICENSE = "other"

_DSC_URL = "http://kyoto.let.vu.nl/dutchsemcor/annotated_data/human_filtered/1.2.2.HUMAN_ANNOTATIONS_FILTERED_FOR_TRAINING.zip"


class Sonar500(datasets.GeneratorBasedBuilder):
    VERSION = datasets.Version("1.2.1")

    BUILDER_CONFIGS = [
        datasets.BuilderConfig(name="dutch-sem-cor", description="Only documents that are included in dutch-sem-cor"),
    ]

    def _info(self):
        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=datasets.Features({
                "doc_id": datasets.Value("string"),
                "sent_id": datasets.Value("string"),
                "token_ids": datasets.Sequence(datasets.Value("string")),
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

    def _dutch_sem_cor_sent_ids(self, dl_manager):
        data_dir = dl_manager.download_and_extract(_DSC_URL)
        data_dir = os.path.join(data_dir, "1.2.2.HUMAN_ANNOTATIONS_FILTERED_FOR_TRAINING")  # type: ignore
        sent_ids = set()
        for filepath in [
                os.path.join(data_dir, "annotations.training.DSC.adjs.xml"),
                os.path.join(data_dir, "annotations.training.DSC.nouns.xml"),
                os.path.join(data_dir, "annotations.training.DSC.verbs.xml")
        ]:
            root = ElementTree.parse(filepath).getroot()
            for token in root:
                sent_id, _ = token.attrib["token_id"].split(".w.", maxsplit=1)
                sent_ids.add(sent_id)
        return sent_ids

    def _split_generators(self, dl_manager):
        data_dir = os.path.abspath(os.path.expanduser(dl_manager.manual_dir))  # type: ignore

        dcoi_dir = os.path.join(data_dir, "SONAR500", "DCOI")
        if not os.path.isdir(dcoi_dir):
            raise ValueError(f"Provided SoNaR DCOI directory does not exist at {dcoi_dir}")

        if self.config.name == "dutch-sem-cor":
            sent_ids = self._dutch_sem_cor_sent_ids(dl_manager)
        else:
            sent_ids = None

        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,  # type: ignore
                gen_kwargs={"dcoi_dir": dcoi_dir, "sent_ids": sent_ids},
            ),
        ]

    def _generate_examples(self, dcoi_dir, sent_ids):
        key = 0

        if self.config.name == "dutch-sem-cor":
            doc_ids = {sent_id.split(".", maxsplit=1)[0] for sent_id in sent_ids}
        else:
            doc_ids = None

        for filepath in glob(os.path.join(dcoi_dir, "**", f"*.dcoi.xml"), recursive=True):
            doc_id = os.path.basename(filepath).removesuffix(".dcoi.xml")
            if doc_ids is not None and doc_id not in doc_ids:
                continue

            root = ElementTree.parse(filepath).getroot()
            # doc_id = root.attrib["{http://www.w3.org/XML/1998/namespace}id"]

            for s in root.iter("{http://lands.let.ru.nl/projects/d-coi/ns/1.0}s"):
                sent_id = s.attrib["{http://www.w3.org/XML/1998/namespace}id"]
                if sent_ids is not None and sent_id not in sent_ids:
                    continue

                token_ids, tokens = [], []
                for w in s.findall("{http://lands.let.ru.nl/projects/d-coi/ns/1.0}w"):
                    token_ids.append(w.attrib["{http://www.w3.org/XML/1998/namespace}id"])
                    tokens.append(w.text)

                yield key, {
                    "doc_id": doc_id,
                    "sent_id": sent_id,
                    "token_ids": token_ids,
                    "tokens": tokens,
                }
                key += 1
