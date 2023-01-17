import itertools

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

_HOMEPAGE = ""

_LICENSE = "other"

class DPR(datasets.GeneratorBasedBuilder):
    VERSION = datasets.Version("0.1.0")

    def _info(self):
        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=datasets.Features({
                "tokens": datasets.Sequence(datasets.Value("string")),
                "span1_index": datasets.Value("int32"),
                "span2_index": datasets.Value("int32"),
                "span1_tokens": datasets.Value("string"),
                "span2_tokens": datasets.Value("string"),
                "label": datasets.ClassLabel(names=["different", "same"]),
            }),
            homepage=_HOMEPAGE,
            license=_LICENSE,
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager):
        dataset = datasets.load_dataset("hf_datasets/semeval2010-task1", "nl")
        assert isinstance(dataset, datasets.DatasetDict)
        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,  # type: ignore
                gen_kwargs={
                    "dataset": dataset["train"],
                },
            ),
            datasets.SplitGenerator(
                name=datasets.Split.VALIDATION,  # type: ignore
                gen_kwargs={
                    "dataset": dataset["validation"],
                },
            ),
            datasets.SplitGenerator(
                name=datasets.Split.TEST,  # type: ignore
                gen_kwargs={
                    "dataset": dataset["test"],
                },
            ),
        ]

    def _generate_examples(self, dataset: datasets.Dataset):
        pron_id = dataset.features["pos_tags"].feature.str2int("VNW")

        key = 0
        for sent_idx in range(len(dataset)):
            sent = dataset[sent_idx]

            coref_starts = sent["coref_spans"]["start"]
            coref_ends = sent["coref_spans"]["end"]

            for coref_idx in range(len(coref_starts)):
                start = coref_starts[coref_idx]
                end = coref_ends[coref_idx]

                if end - start != 1 or sent["pos_tags"][start] != pron_id:
                    continue

                positive_idx, negative_idx = find_coref(dataset, sent_idx, coref_idx, pron_id)
                if positive_idx is None or negative_idx is None:
                    continue

                pos_sent_idx, pos_coref_idx = positive_idx
                neg_sent_idx, neg_coref_idx = negative_idx

                tokens = list(sent["tokens"])
                offset, pos_offset, neg_offset = 0, 0, 0

                # prepend previous sentence
                if pos_sent_idx < sent_idx or neg_sent_idx < sent_idx:
                    prev_tokens = list(dataset[sent_idx - 1]["tokens"])
                    offset = len(prev_tokens)
                    if pos_sent_idx == sent_idx:
                        pos_offset = offset
                    if neg_sent_idx == sent_idx:
                        neg_offset = offset
                    tokens = prev_tokens + tokens

                # append next sentence
                if pos_sent_idx > sent_idx or neg_sent_idx > sent_idx:
                    if pos_sent_idx > sent_idx:
                        pos_offset = len(tokens)
                    if neg_sent_idx > sent_idx:
                        neg_offset = len(tokens)
                    tokens.extend(dataset[sent_idx + 1]["tokens"])

                # yield positive example
                pos_coref = dataset[pos_sent_idx]["coref_spans"]
                pos_start = pos_coref["start"][pos_coref_idx] + pos_offset
                pos_end = pos_coref["end"][pos_coref_idx] + pos_offset
                yield key, {
                    "tokens": tokens,
                    "span1_index": start + offset,
                    "span2_index": pos_start,
                    "span1_tokens": [tokens[start + offset]],
                    "span2_tokens": tokens[pos_start:pos_end],
                    "label": True,
                }
                key += 1

                # yield negative example
                neg_coref = dataset[neg_sent_idx]["coref_spans"]
                neg_start = neg_coref["start"][neg_coref_idx] + neg_offset
                neg_end = neg_coref["end"][neg_coref_idx] + neg_offset
                yield key, {
                    "tokens": tokens,
                    "span1_index": start + offset,
                    "span2_index": neg_start,
                    "span1_tokens": [tokens[start + offset]],
                    "span2_tokens": tokens[neg_start:neg_end],
                    "label": False,
                }
                key += 1



def find_coref(d, sent_idx, coref_idx, pron_id):
    coref_id = d[sent_idx]["coref_spans"]["id"][coref_idx]

    def _find_coref(other_sent_idx, positive_idx=None, negative_idx=None):
        other_sent = d[other_sent_idx]
        other_starts = other_sent["coref_spans"]["start"]
        other_ends = other_sent["coref_spans"]["end"]
        other_tags = other_sent["pos_tags"]

        for other_coref_idx, other_coref_id in enumerate(other_sent["coref_spans"]["id"]):
            if other_coref_id == coref_id and positive_idx is not None:
                continue
            if other_coref_id != coref_id and negative_idx is not None:
                continue

            if other_sent_idx == sent_idx and other_coref_idx == coref_idx:
                continue
            if other_ends[other_coref_idx] - other_starts[other_coref_idx] == 1 and other_tags[other_starts[other_coref_idx]] == pron_id:
                continue

            if other_coref_id == coref_id:
                positive_idx = (other_sent_idx, other_coref_idx)
            else:
                negative_idx = (other_sent_idx, other_coref_idx)

            if positive_idx and negative_idx:
                break
        return positive_idx, negative_idx

    # in same sentence
    positive_idx, negative_idx = _find_coref(sent_idx)
    if positive_idx and negative_idx:
        return positive_idx, negative_idx

    # in previous sentence
    if sent_idx > 0:
        positive_idx, negative_idx = _find_coref(sent_idx - 1, positive_idx, negative_idx)
        if positive_idx and negative_idx:
            return positive_idx, negative_idx

    # find in next sentence
    positive_idx, negative_idx = _find_coref(sent_idx + 1, positive_idx, negative_idx)
    return positive_idx, negative_idx