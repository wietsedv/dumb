import json
import os
# from random import Random
import conllu

import datasets

_DESCRIPTION = """\
A large corpus of written Dutch texts (1,000,000 words) has been syntactically annotated (manually corrected), based on CGN and D-COI.
"""

_HOMEPAGE = "https://www.let.rug.nl/~vannoord/Lassy/"

_LICENSE = "other"

# _RANDOM_SEED = 7892501

_UPOS_TAGS = [
    "ADJ", "ADP", "ADV", "AUX", "CCONJ", "DET", "INTJ", "NOUN", "NUM", "PART", "PRON", "PROPN", "PUNCT", "SCONJ", "SYM",
    "VERB", "X"
]

_XPOS_TAGS = [
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

_DEPRELS = [
    "root", "acl", "acl:relcl", "advcl", "advmod", "amod", "appos", "aux", "aux:pass", "case", "cc", "ccomp",
    "compound:prt", "conj", "cop", "csubj", "csubj:pass", "det", "expl", "expl:pv", "fixed", "flat", "iobj", "mark",
    "nmod", "nmod:poss", "nsubj", "nsubj:pass", "nummod", "obj", "obl", "obl:agent", "orphan", "parataxis", "punct",
    "xcomp"
]

# _NER_TAGS = [
#     "O",
#     "B-PER",  # person
#     "I-PER",
#     "B-ORG",  # organization
#     "I-ORG",
#     "B-LOC",  # location
#     "I-LOC",
#     "B-PRO",  # product
#     "I-PRO",
#     "B-EVE",  # event
#     "I-EVE",
#     "B-MISC",  # miscellaneous
#     "I-MISC",
# ]


class LassySmall(datasets.GeneratorBasedBuilder):
    """TODO: Short description of my dataset."""

    VERSION = datasets.Version("6.0.0")

    @property
    def manual_download_instructions(self):
        return (
            "To use Lassy you have to download it manually. You can download the corpus at https://taalmaterialen.ivdnt.org/?__wpdmlo=1659"
            " Please extract the .tar.gz file in one folder and load the dataset with: "
            "`datasets.load_dataset(\"hf_datasets/lassy\", data_dir=\"path/to/lassy6\")`")

    def _info(self):
        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=datasets.Features({
                "doc_id": datasets.Value("string"),
                "sent_id": datasets.Value("string"),
                "tokens": datasets.Sequence(datasets.Value("string")),
                "lemmas": datasets.Sequence(datasets.Value("string")),
                "pos_tags": datasets.Sequence(datasets.features.ClassLabel(names=_UPOS_TAGS)),
                "xpos_tags": datasets.Sequence(datasets.features.ClassLabel(names=_XPOS_TAGS)),
                "feats": datasets.Sequence(datasets.Value("string")),
                "heads": datasets.Sequence(datasets.Value("int32")),
                "dep_tags": datasets.Sequence(datasets.features.ClassLabel(names=_DEPRELS)),
                # "ner_tags": datasets.Sequence(datasets.features.ClassLabel(names=_NER_TAGS)),
                # "senses": datasets.Sequence(datasets.Value("string")),
            }),
            homepage=_HOMEPAGE,
            license=_LICENSE,
        )

    def _split_generators(self, dl_manager):
        assert dl_manager.manual_dir
        data_dir = os.path.abspath(os.path.expanduser(dl_manager.manual_dir))

        conllu_dir = os.path.join(data_dir, "CONLLU")
        doc_ids = [n[:-7] for n in os.listdir(conllu_dir) if n[-7:] == ".conllu"]

        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,  # type: ignore
                gen_kwargs={
                    "data_dir": data_dir,
                    "doc_ids": doc_ids,  # doc_ids_train
                },
            ),

        ]

    def _generate_examples(self, data_dir, doc_ids):
        key = 0
        for doc_id in doc_ids:
            conllu_filepath = os.path.join(data_dir, "CONLLU", doc_id + ".conllu")
            with open(conllu_filepath, "r", encoding="utf-8") as f:
                for sent in conllu.parse_incr(f):
                    sent_id = sent.metadata["sent_id"]

                    sent = [t for t in sent if t["form"] != "\xad"]

                    yield key, {
                        "doc_id": doc_id,
                        "sent_id": sent_id,
                        "tokens": [t["form"] for t in sent],
                        "lemmas": [t["lemma"] for t in sent],
                        "pos_tags": [t["upos"] for t in sent],
                        "xpos_tags": [t["xpos"] for t in sent],
                        "feats": [json.dumps(t["feats"]) for t in sent],
                        "heads": [t["head"] for t in sent],
                        "dep_tags": [t["deprel"] for t in sent],
                    }
                    key += 1
