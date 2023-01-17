# TODO: Currently only the filtered human annotations. Maybe add a builder for superset?
# TODO: Currently only SoNaR and no CGN. Add CGN?

import itertools
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

_HOMEPAGE = "http://stel3.ub.edu/semeval2010-coref/"

_LICENSE = "other"


class SemEval2010Task1(datasets.GeneratorBasedBuilder):
    VERSION = datasets.Version("1.0.0")

    BUILDER_CONFIGS = [
        datasets.BuilderConfig(name="ca"),
        datasets.BuilderConfig(name="es"),
        datasets.BuilderConfig(name="it"),
        datasets.BuilderConfig(name="nl"),
    ]

    def _info(self):
        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=datasets.Features({
                "tokens":
                datasets.Sequence(datasets.Value("string")),
                "pos_tags":
                datasets.Sequence(datasets.ClassLabel(names=["ADJ", "BW", "LET", "LID", "N", "SPEC", "TSW", "TW", "VNW", "VG", "VZ", "WW"])),
                "coref_spans":
                datasets.Sequence({
                    "id": datasets.Value("int32"),
                    "start": datasets.Value("int32"),
                    "end": datasets.Value("int32"),
                }),
            }),
            homepage=_HOMEPAGE,
            license=_LICENSE,
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager):
        data_dir = dl_manager.download_and_extract("https://www.cs.upc.edu/~esapena/downloads/task01.posttask.v1.0.tgz")
        data_dir = os.path.join(data_dir, "task01.posttask.v1.0", "corpora")  # type: ignore

        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,  # type: ignore
                gen_kwargs={"filename": os.path.join(data_dir, "training", f"{self.config.name}.train.txt")},
            ),
            datasets.SplitGenerator(
                name=datasets.Split.VALIDATION,  # type: ignore
                gen_kwargs={"filename": os.path.join(data_dir, "training", f"{self.config.name}.devel.txt")},
            ),
            datasets.SplitGenerator(
                name=datasets.Split.TEST,  # type: ignore
                gen_kwargs={"filename": os.path.join(data_dir, "test", f"{self.config.name}.test.txt")},
            ),
        ]

    def _generate_examples(self, filename):
        ex = None
        key = 0
        refs = {}
        with open(filename) as f:
            for line in f:
                line = line.strip()
                if not line:
                    if ex:
                        # assert len(refs) == 0, f"{refs}"
                        yield key, ex
                        key += 1
                        ex = None
                        refs = {}
                    continue
                if line[0] == "#":
                    continue

                # print(line)

                if not ex:
                    ex = {"tokens": [], "pos_tags": [], "coref_spans": []}

                parts = line.split("\t")
                # print(token_id, token, lemma, pos, feat, head, deprel, ne, pred, apreds, coref)
                # print(parts, len(parts))

                # token_id = parts[0]
                token = parts[1]
                pos_tag = parts[5]
                coref = parts[16]

                # hotfixes
                replacements = [
                    ("‚Äö√Ñ¬∞", "à"),
                    ("¬¨‚àë", "á"),
                    ("‚àö√©", "ë"),
                    ("‚àö√†", "é"),
                    ("‚àö√£", "è"),
                    ("‚àö√ß", "ê"),
                    ("‚àö√Æ", "ï"),
                    ("¬¨‚àè", "ü"),
                    ("√Ä√ú", "ö"),
                    ("‚àö√µ", "ó"),
                    ("‚àö‚àë", "Ö"),
                    ("‚àö√Ö", "ç"),
                ]
                for a, b in replacements:
                    token = token.replace(a, b)
                coref = coref.lstrip("-").lstrip("_").replace(")(", ")|(")
                if coref in ["LET()", "+"]:
                    coref = ""

                ex["tokens"].append(token)
                ex["pos_tags"].append(pos_tag)

                if coref:
                    i = len(ex["tokens"]) - 1

                    for coref_ in coref.split("|"):
                        # print(parts)
                        ref_id = int(coref_.lstrip("(").rstrip(")"))

                        is_start = coref_[0] == "("
                        is_end = coref_[-1] == ")"

                        if is_start and is_end:
                            ex["coref_spans"].append({
                                "id": ref_id,
                                "start": i,
                                "end": i + 1,
                            })
                            continue

                        if is_start:  # or (not is_end and ref_id not in refs):
                            assert ref_id not in refs, f"{ref_id} {refs}"
                            refs[ref_id] = i
                            continue

                        if is_end:  # or (not is_start and ref_id in refs):
                            ex["coref_spans"].append({
                                "id": ref_id,
                                "start": refs[ref_id],
                                "end": i + 1,
                            })
                            del refs[ref_id]
                            continue

                        print(f"unknown coref {coref} {refs}")

        if ex:
            yield key, ex