"""TODO: Add a description here."""

import csv
import json
import os

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
_LICENSE = ""

_POS_TAGS = [
    "ADJ|nom|basis|met-e|mv-n", "ADJ|nom|basis|met-e|zonder-n|bijz", "ADJ|nom|basis|met-e|zonder-n|stan",
    "ADJ|nom|basis|zonder|mv-n", "ADJ|nom|basis|zonder|zonder-n", "ADJ|nom|comp|met-e|mv-n",
    "ADJ|nom|comp|met-e|zonder-n|stan", "ADJ|nom|sup|met-e|mv-n", "ADJ|nom|sup|met-e|zonder-n|stan",
    "ADJ|nom|sup|zonder|zonder-n", "ADJ|postnom|basis|met-s", "ADJ|postnom|basis|zonder", "ADJ|postnom|comp|met-s",
    "ADJ|postnom|comp|zonder", "ADJ|prenom|basis|met-e|bijz", "ADJ|prenom|basis|met-e|stan", "ADJ|prenom|basis|zonder",
    "ADJ|prenom|comp|met-e|stan", "ADJ|prenom|comp|zonder", "ADJ|prenom|sup|met-e|stan", "ADJ|prenom|sup|zonder",
    "ADJ|vrij|basis|zonder", "ADJ|vrij|comp|zonder", "ADJ|vrij|dim|zonder", "ADJ|vrij|sup|zonder", "BW", "LET",
    "LID|bep|dat|evf", "LID|bep|dat|evmo", "LID|bep|gen|evmo", "LID|bep|gen|rest3", "LID|bep|stan|evon",
    "LID|bep|stan|rest", "LID|onbep|stan|agr", "N|eigen|ev|basis|gen", "N|eigen|ev|basis|genus|stan",
    "N|eigen|ev|basis|onz|stan", "N|eigen|ev|basis|zijd|stan", "N|eigen|ev|dim|onz|stan", "N|eigen|mv|basis",
    "N|soort|ev|basis|dat", "N|soort|ev|basis|gen", "N|soort|ev|basis|genus|stan", "N|soort|ev|basis|onz|stan",
    "N|soort|ev|basis|zijd|stan", "N|soort|ev|dim|onz|stan", "N|soort|mv|basis", "N|soort|mv|dim", "SPEC|afgebr",
    "SPEC|afk", "SPEC|deeleigen", "SPEC|enof", "SPEC|meta", "SPEC|symb", "SPEC|vreemd", "TSW",
    "TW|hoofd|nom|mv-n|basis", "TW|hoofd|nom|mv-n|dim", "TW|hoofd|nom|zonder-n|basis", "TW|hoofd|nom|zonder-n|dim",
    "TW|hoofd|prenom|stan", "TW|hoofd|vrij", "TW|rang|nom|mv-n", "TW|rang|nom|zonder-n", "TW|rang|prenom|bijz",
    "TW|rang|prenom|stan", "VG|neven", "VG|onder", "VNW|aanw|adv-pron|obl|vol|3o|getal",
    "VNW|aanw|adv-pron|stan|red|3|getal", "VNW|aanw|det|dat|nom|met-e|zonder-n", "VNW|aanw|det|dat|prenom|met-e|evmo",
    "VNW|aanw|det|gen|prenom|met-e|rest3", "VNW|aanw|det|stan|nom|met-e|mv-n", "VNW|aanw|det|stan|nom|met-e|zonder-n",
    "VNW|aanw|det|stan|prenom|met-e|rest", "VNW|aanw|det|stan|prenom|zonder|agr",
    "VNW|aanw|det|stan|prenom|zonder|evon", "VNW|aanw|det|stan|prenom|zonder|rest", "VNW|aanw|det|stan|vrij|zonder",
    "VNW|aanw|pron|gen|vol|3m|ev", "VNW|aanw|pron|gen|vol|3o|ev", "VNW|aanw|pron|stan|vol|3o|ev",
    "VNW|aanw|pron|stan|vol|3|getal", "VNW|betr|det|stan|nom|zonder|zonder-n", "VNW|betr|pron|stan|vol|3|ev",
    "VNW|betr|pron|stan|vol|persoon|getal", "VNW|bez|det|dat|vol|3|ev|prenom|met-e|evf",
    "VNW|bez|det|gen|vol|1|ev|prenom|zonder|evmo", "VNW|bez|det|gen|vol|1|mv|prenom|met-e|rest3",
    "VNW|bez|det|stan|nadr|2v|mv|prenom|zonder|agr", "VNW|bez|det|stan|red|1|ev|prenom|zonder|agr",
    "VNW|bez|det|stan|red|2v|ev|prenom|zonder|agr", "VNW|bez|det|stan|red|3|ev|prenom|zonder|agr",
    "VNW|bez|det|stan|vol|1|ev|nom|met-e|zonder-n", "VNW|bez|det|stan|vol|1|ev|prenom|zonder|agr",
    "VNW|bez|det|stan|vol|1|mv|nom|met-e|zonder-n", "VNW|bez|det|stan|vol|1|mv|prenom|met-e|rest",
    "VNW|bez|det|stan|vol|1|mv|prenom|zonder|evon", "VNW|bez|det|stan|vol|2v|ev|prenom|zonder|agr",
    "VNW|bez|det|stan|vol|2|getal|nom|met-e|zonder-n", "VNW|bez|det|stan|vol|2|getal|prenom|zonder|agr",
    "VNW|bez|det|stan|vol|3m|ev|nom|met-e|mv-n", "VNW|bez|det|stan|vol|3m|ev|nom|met-e|zonder-n",
    "VNW|bez|det|stan|vol|3m|ev|prenom|met-e|rest", "VNW|bez|det|stan|vol|3p|mv|nom|met-e|mv-n",
    "VNW|bez|det|stan|vol|3p|mv|nom|met-e|zonder-n", "VNW|bez|det|stan|vol|3p|mv|prenom|met-e|rest",
    "VNW|bez|det|stan|vol|3v|ev|prenom|met-e|rest", "VNW|bez|det|stan|vol|3|ev|prenom|zonder|agr",
    "VNW|bez|det|stan|vol|3|mv|prenom|zonder|agr", "VNW|excl|pron|stan|vol|3|getal",
    "VNW|onbep|adv-pron|gen|red|3|getal", "VNW|onbep|adv-pron|obl|vol|3o|getal", "VNW|onbep|det|dat|prenom|met-e|evmo",
    "VNW|onbep|det|gen|nom|met-e|mv-n", "VNW|onbep|det|gen|prenom|met-e|mv", "VNW|onbep|det|stan|nom|met-e|mv-n",
    "VNW|onbep|det|stan|nom|met-e|zonder-n", "VNW|onbep|det|stan|nom|zonder|zonder-n",
    "VNW|onbep|det|stan|prenom|met-e|agr", "VNW|onbep|det|stan|prenom|met-e|evz", "VNW|onbep|det|stan|prenom|met-e|mv",
    "VNW|onbep|det|stan|prenom|met-e|rest", "VNW|onbep|det|stan|prenom|zonder|agr",
    "VNW|onbep|det|stan|prenom|zonder|evon", "VNW|onbep|det|stan|vrij|zonder",
    "VNW|onbep|grad|gen|nom|met-e|mv-n|basis", "VNW|onbep|grad|stan|nom|met-e|mv-n|basis",
    "VNW|onbep|grad|stan|nom|met-e|mv-n|sup", "VNW|onbep|grad|stan|nom|met-e|zonder-n|basis",
    "VNW|onbep|grad|stan|nom|met-e|zonder-n|sup", "VNW|onbep|grad|stan|nom|zonder|zonder-n|sup",
    "VNW|onbep|grad|stan|prenom|met-e|agr|basis", "VNW|onbep|grad|stan|prenom|met-e|agr|comp",
    "VNW|onbep|grad|stan|prenom|met-e|agr|sup", "VNW|onbep|grad|stan|prenom|met-e|mv|basis",
    "VNW|onbep|grad|stan|prenom|zonder|agr|basis", "VNW|onbep|grad|stan|prenom|zonder|agr|comp",
    "VNW|onbep|grad|stan|vrij|zonder|basis", "VNW|onbep|grad|stan|vrij|zonder|comp",
    "VNW|onbep|grad|stan|vrij|zonder|sup", "VNW|onbep|pron|gen|vol|3p|ev", "VNW|onbep|pron|stan|vol|3o|ev",
    "VNW|onbep|pron|stan|vol|3p|ev", "VNW|pers|pron|nomin|nadr|1|ev", "VNW|pers|pron|nomin|nadr|2b|getal",
    "VNW|pers|pron|nomin|nadr|3m|ev|masc", "VNW|pers|pron|nomin|nadr|3v|ev|fem", "VNW|pers|pron|nomin|red|1|mv",
    "VNW|pers|pron|nomin|red|2v|ev", "VNW|pers|pron|nomin|red|3p|ev|masc", "VNW|pers|pron|nomin|red|3|ev|masc",
    "VNW|pers|pron|nomin|vol|1|ev", "VNW|pers|pron|nomin|vol|1|mv", "VNW|pers|pron|nomin|vol|2b|getal",
    "VNW|pers|pron|nomin|vol|2v|ev", "VNW|pers|pron|nomin|vol|2|getal", "VNW|pers|pron|nomin|vol|3p|mv",
    "VNW|pers|pron|nomin|vol|3v|ev|fem", "VNW|pers|pron|nomin|vol|3|ev|masc", "VNW|pers|pron|obl|nadr|3m|ev|masc",
    "VNW|pers|pron|obl|nadr|3p|mv", "VNW|pers|pron|obl|nadr|3v|getal|fem", "VNW|pers|pron|obl|red|3|ev|masc",
    "VNW|pers|pron|obl|vol|2v|ev", "VNW|pers|pron|obl|vol|3p|mv", "VNW|pers|pron|obl|vol|3|ev|masc",
    "VNW|pers|pron|obl|vol|3|getal|fem", "VNW|pers|pron|stan|nadr|2v|mv", "VNW|pers|pron|stan|red|3|ev|fem",
    "VNW|pers|pron|stan|red|3|ev|onz", "VNW|pers|pron|stan|red|3|mv", "VNW|pr|pron|obl|nadr|1|ev",
    "VNW|pr|pron|obl|nadr|1|mv", "VNW|pr|pron|obl|nadr|2v|getal", "VNW|pr|pron|obl|nadr|2|getal",
    "VNW|pr|pron|obl|red|1|ev", "VNW|pr|pron|obl|red|2v|getal", "VNW|pr|pron|obl|vol|1|ev", "VNW|pr|pron|obl|vol|1|mv",
    "VNW|pr|pron|obl|vol|2|getal", "VNW|recip|pron|gen|vol|persoon|mv", "VNW|recip|pron|obl|vol|persoon|mv",
    "VNW|refl|pron|obl|nadr|3|getal", "VNW|refl|pron|obl|red|3|getal", "VNW|vb|adv-pron|obl|vol|3o|getal",
    "VNW|vb|det|stan|nom|met-e|zonder-n", "VNW|vb|det|stan|prenom|met-e|rest", "VNW|vb|det|stan|prenom|zonder|evon",
    "VNW|vb|pron|gen|vol|3m|ev", "VNW|vb|pron|gen|vol|3p|mv", "VNW|vb|pron|gen|vol|3v|ev", "VNW|vb|pron|stan|vol|3o|ev",
    "VNW|vb|pron|stan|vol|3p|getal", "VZ|fin", "VZ|init", "VZ|versm", "WW|inf|nom|zonder|zonder-n",
    "WW|inf|prenom|met-e", "WW|inf|prenom|zonder", "WW|inf|vrij|zonder", "WW|od|nom|met-e|mv-n",
    "WW|od|nom|met-e|zonder-n", "WW|od|prenom|met-e", "WW|od|prenom|zonder", "WW|od|vrij|zonder", "WW|pv|conj|ev",
    "WW|pv|tgw|ev", "WW|pv|tgw|met-t", "WW|pv|tgw|mv", "WW|pv|verl|ev", "WW|pv|verl|mv", "WW|vd|nom|met-e|mv-n",
    "WW|vd|nom|met-e|zonder-n", "WW|vd|prenom|met-e", "WW|vd|prenom|zonder", "WW|vd|vrij|zonder"
]

_NER_TAGS = [
    "O", "B-PER", "I-PER", "B-ORG", "I-ORG", "B-LOC", "I-LOC", "B-PRO", "I-PRO", "B-EVE", "I-EVE", "B-MISC", "I-MISC"
]

_DALC_LABELS = [
    "not",
    "abusive",
    "offensive",
]


class Dumb(datasets.GeneratorBasedBuilder):
    VERSION = datasets.Version("0.1.0")

    BUILDER_CONFIGS = [
        datasets.BuilderConfig(name="copanl"),
        datasets.BuilderConfig(name="dalc"),
        datasets.BuilderConfig(name="dbrd"),
        datasets.BuilderConfig(name="lassy-pos"),
        datasets.BuilderConfig(name="dpr"),
        datasets.BuilderConfig(name="sicknl-nli"),
        datasets.BuilderConfig(name="sonar-ne"),
        datasets.BuilderConfig(name="wicnl"),
    ]

    def _info(self):
        if self.config.name == "copanl":
            features = datasets.Features({
                "choices": datasets.Sequence({
                    "sentence1": datasets.Value("string"),
                    "sentence2": datasets.Value("string"),
                }),
                "label": datasets.Value("int32"),
            })
        elif self.config.name == "dalc":
            features = datasets.Features({
                "text": datasets.Value("string"),
                "label": datasets.ClassLabel(names=_DALC_LABELS),
            })
        elif self.config.name == "dbrd":
            features = datasets.Features({
                "text": datasets.Value("string"),
                "label": datasets.ClassLabel(names=["negative", "positive"]),
            })
        elif self.config.name == "lassy-pos":
            features = datasets.Features({
                "tokens": datasets.Sequence(datasets.Value("string")),
                "tags": datasets.Sequence(datasets.features.ClassLabel(names=_POS_TAGS)),
            })
        elif self.config.name == "dpr":
            features = datasets.Features({
                "tokens": datasets.Sequence(datasets.Value("string")),
                "index1": datasets.Value("int32"),
                "index2": datasets.Value("int32"),
                "label": datasets.ClassLabel(names=["different", "same"]),
            })
        elif self.config.name == "sicknl-nli":
            features = datasets.Features({
                "sentence1": datasets.Value("string"),
                "sentence2": datasets.Value("string"),
                "label": datasets.ClassLabel(names=["entailment", "neutral", "contradiction"]),
            })
        elif self.config.name == "sonar-ne":
            features = datasets.Features({
                "tokens": datasets.Sequence(datasets.Value("string")),
                "tags": datasets.Sequence(datasets.features.ClassLabel(names=_NER_TAGS)),
            })
        elif self.config.name == "wicnl":
            features = datasets.Features({
                "tokens1": datasets.Sequence(datasets.Value("string")),
                "tokens2": datasets.Sequence(datasets.Value("string")),
                "index1": datasets.Value("int32"),
                "index2": datasets.Value("int32"),
                "label": datasets.ClassLabel(names=["different", "same"]),
            })
        else:
            raise ValueError(
                f"Task {self.config.name} is not supported. Available: {list(self.builder_configs.keys())}")

        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=features,
            homepage=_HOMEPAGE,
            license=_LICENSE,
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager):
        if dl_manager.manual_dir is None:
            data_dir = os.path.abspath("dumb")
        else:
            data_dir = os.path.abspath(os.path.expanduser(dl_manager.manual_dir))

        data_dir = os.path.join(data_dir, self.config.name)
        return [
            datasets.SplitGenerator(name=split, gen_kwargs={"filepath": os.path.join(data_dir, f"{split}.jsonl")})
            for split in ["train", "validation", "test"]
        ]

    def _generate_examples(self, filepath):
        with open(filepath, encoding="utf-8") as f:
            for key, row in enumerate(f):
                item = json.loads(row)
                if self.config.name == "dalc":
                    del item["target_label"]
                    del item["explicitness_label"]
                yield key, item
