import os
from random import Random
from datasets import load_dataset, DatasetDict, Dataset, Split, Value
from argparse import ArgumentParser
from typing import List, Tuple


def split_sonar(dataset1: Dataset, dataset2: Dataset, seed: int, r_test=0.05, r_valid=0.02):
    rand = Random(seed)

    doc_ids1 = set(dataset1["doc_id"])
    doc_ids2 = set(dataset2["doc_id"])

    doc_ids_1 = sorted(doc_ids1 - doc_ids2)
    doc_ids_2 = sorted(doc_ids2 - doc_ids1)
    doc_ids12 = sorted(doc_ids1 & doc_ids2)

    rand.shuffle(doc_ids_1)
    rand.shuffle(doc_ids_2)
    rand.shuffle(doc_ids12)

    n_test_1 = int(len(doc_ids_1) * r_test)
    n_test_2 = int(len(doc_ids_2) * r_test)
    n_test12 = int(len(doc_ids12) * r_test)

    n_valid_1 = int(len(doc_ids_1) * r_valid)
    n_valid_2 = int(len(doc_ids_2) * r_valid)
    n_valid12 = int(len(doc_ids12) * r_valid)

    doc_ids_test = set(doc_ids_1[:n_test_1] + doc_ids_2[:n_test_2] + doc_ids12[:n_test12])
    doc_ids_valid = set(doc_ids_1[n_test_1:n_test_1 + n_valid_1] + doc_ids_2[n_test_2:n_test_2 + n_valid_2] +
                        doc_ids12[n_test12:n_test12 + n_valid12])
    doc_ids_train = set(doc_ids_1[n_test_1 + n_valid_1:] + doc_ids_2[n_test_2 + n_valid_2:] +
                        doc_ids12[n_test12 + n_valid12:])

    dataset1_new = DatasetDict({
        Split.TRAIN: dataset1.filter(lambda x: x["doc_id"] in doc_ids_train),
        Split.VALIDATION: dataset1.filter(lambda x: x["doc_id"] in doc_ids_valid),
        Split.TEST: dataset1.filter(lambda x: x["doc_id"] in doc_ids_test),
    })
    dataset2_new = DatasetDict({
        Split.TRAIN: dataset2.filter(lambda x: x["doc_id"] in doc_ids_train),
        Split.VALIDATION: dataset2.filter(lambda x: x["doc_id"] in doc_ids_valid),
        Split.TEST: dataset2.filter(lambda x: x["doc_id"] in doc_ids_test),
    })
    return dataset1_new, dataset2_new


def split_dbrd(dataset: DatasetDict, seed: int):
    split_dataset = dataset["train"].train_test_split(test_size=500, seed=seed)
    return DatasetDict({
        Split.TRAIN: split_dataset[Split.TRAIN],
        Split.VALIDATION: split_dataset[Split.TEST],
        Split.TEST: dataset[Split.TEST],
    })


def scale_sicknl_score(x):
    x["score"] = x["score"] / 5
    return x


def prepare_copa(x):
    c1s1, c1s2 = (x["premise"], x["choice1"]) if x["question"] == 1 else (x["choice1"], x["premise"])
    c2s1, c2s2 = (x["premise"], x["choice2"]) if x["question"] == 1 else (x["choice2"], x["premise"])
    return {
        "choices": [
            {
                "sentence1": c1s1,
                "sentence2": c1s2
            },
            {
                "sentence1": c2s1,
                "sentence2": c2s2
            },
        ],
        "label": x["label"],
    }


def prepare_tasks(args):
    # lassy* / sonar*
    # lassy_dataset = load_dataset("hf_datasets/lassy-small", data_dir=args.lassy,
    #                              split="all").remove_columns(["sent_id", "pos_tags", "feats", "lemmas"])
    # assert isinstance(lassy_dataset, Dataset)
    # sonar_ner_dataset = load_dataset("hf_datasets/sonar-1", "ne", data_dir=args.sonar,
    #                                  split="all").remove_columns(["sent_nr",
    #                                                               "xner_tags"]).rename_column("ner_tags", "tags")
    # assert isinstance(sonar_ner_dataset, Dataset)
    # lassy_dataset, sonar_ner_dataset = split_sonar(lassy_dataset, sonar_ner_dataset, args.seed)

    # # lassy-pos
    # lassy_pos_dataset = lassy_dataset.remove_columns(["doc_id", "heads", "dep_tags"]).rename_column("xpos_tags", "tags")
    # lassy_pos_id2label = lassy_pos_dataset["train"].features["tags"].feature.names
    # lassy_pos_dataset = lassy_pos_dataset.map(lambda x: {
    #     "tags_str": [lassy_pos_id2label[t] for t in x["tags"]]
    # }).remove_columns(["tags"]).rename_column("tags_str", "tags")

    # # sonar-ne
    # sonar_ner_dataset = sonar_ner_dataset.remove_columns(["doc_id"])
    # sonar_ner_id2label = sonar_ner_dataset["train"].features["tags"].feature.names
    # sonar_ner_dataset = sonar_ner_dataset.map(lambda x: {
    #     "tags_str": [sonar_ner_id2label[t] for t in x["tags"]]
    # }).remove_columns(["tags"]).rename_column("tags_str", "tags")

    # # sicknl*
    # sicknl_dataset = load_dataset("hf_datasets/sick-nl").remove_columns([
    #     "pair_id", "entailment_12", "entailment_21", "sentence1_original", "sentence2_original", "sentence1_dataset",
    #     "sentence2_dataset"
    # ])

    # # sicknl-nli
    # sicknl_nli_dataset = sicknl_dataset.remove_columns(["relatedness_score"])
    # assert isinstance(sicknl_nli_dataset, DatasetDict)
    # sicknl_nli_id2label = sicknl_nli_dataset["train"].features["label"].names
    # sicknl_nli_dataset = sicknl_nli_dataset.map(lambda x: {
    #     "label_str": sicknl_nli_id2label[x["label"]]
    # }).remove_columns(["label"]).rename_column("label_str", "label")

    # # dbrd
    # dbrd_dataset = load_dataset("hf_datasets/dbrd").remove_columns(["doc_id"])
    # assert isinstance(dbrd_dataset, DatasetDict)
    # dbrd_dataset = split_dbrd(dbrd_dataset, args.seed)
    # dbrd_id2label = dbrd_dataset["train"].features["label"].names
    # dbrd_dataset = dbrd_dataset.map(lambda x: {
    #     "label_str": dbrd_id2label[x["label"]]
    # }).remove_columns(["label"]).rename_column("label_str", "label")

    # # dalc
    # dalc_dataset = load_dataset("hf_datasets/dalc", data_dir=args.dalc).remove_columns(["id"])
    # assert isinstance(dalc_dataset, DatasetDict)
    # dalc_id2label = dalc_dataset["train"].features["label"].names
    # dalc_id2tgtlabel = dalc_dataset["train"].features["target_label"].names
    # dalc_id2explabel = dalc_dataset["train"].features["explicitness_label"].names
    # dalc_dataset = dalc_dataset.map(
    #     lambda x: {
    #         "label_str": dalc_id2label[x["label"]],
    #         "target_label_str": dalc_id2tgtlabel[x["target_label"]],
    #         "explicitness_label_str": dalc_id2explabel[x["explicitness_label"]],
    #     }).remove_columns(["label", "target_label", "explicitness_label"]).rename_columns({
    #         "label_str":
    #         "label",
    #         "target_label_str":
    #         "target_label",
    #         "explicitness_label_str":
    #         "explicitness_label",
    #     })

    # # copanl
    # copanl_dataset = load_dataset("hf_datasets/copa-nl").map(prepare_copa).remove_columns(
    #     ["id", "premise", "choice1", "choice2", "question"])

    # # wicnl
    # wicnl_dataset = load_dataset("hf_datasets/wic-nl", data_dir=args.sonar).remove_columns(["lemma"])
    # assert isinstance(wicnl_dataset, DatasetDict)
    # wicnl_id2label = wicnl_dataset["train"].features["label"].names
    # wicnl_dataset = wicnl_dataset.map(lambda x: {
    #     "label_str": wicnl_id2label[x["label"]]
    # }).remove_columns(["label"]).rename_column("label_str", "label")

    # # dpr
    # dpr_dataset = load_dataset("hf_datasets/dpr").remove_columns(["span1_tokens", "span2_tokens"]).rename_column(
    #     "span1_index", "index1").rename_column("span2_index", "index2")
    # assert isinstance(dpr_dataset, DatasetDict)
    # dpr_id2label = dpr_dataset["train"].features["label"].names
    # dpr_dataset = dpr_dataset.map(lambda x: {
    #     "label_str": dpr_id2label[x["label"]]
    # }).remove_columns(["label"]).rename_column("label_str", "label")

    # squadnl
    squadnl_dataset = load_dataset("hf_datasets/squad-nl", "nl-v2.0")
    assert isinstance(squadnl_dataset, DatasetDict)

    tasks = [
        # ("copanl", copanl_dataset),  # causal reasoning
        # ("dalc", dalc_dataset),  # abusive language detection
        # ("dbrd", dbrd_dataset),  # sentiment analysis
        # ("lassy-pos", lassy_pos_dataset),  # part of speech tagging
        # ("sicknl-nli", sicknl_nli_dataset),  # natural language inference
        # ("sonar-ne", sonar_ner_dataset),  # named entity recognition
        ("squadnl", squadnl_dataset),  # question answering
        # ("wicnl", wicnl_dataset),  # word sense disambiguation
        # ("dpr", dpr_dataset),  # coreference resolution
    ]
    return tasks


def main():
    parser = ArgumentParser()
    parser.add_argument("--sonar", default="/Volumes/Data/SoNaR-v1.2")
    parser.add_argument("--lassy", default="/Volumes/Data/LassySmall-v6.0")
    parser.add_argument("--dalc", default="/Volumes/Data/DALC")
    parser.add_argument("--seed", type=int, default=872691)
    args = parser.parse_args()

    tasks: List[Tuple[str, DatasetDict]] = prepare_tasks(args)

    for config_name, dataset_dict in tasks:
        data_dir = os.path.join("dumb", config_name)
        os.makedirs(data_dir, exist_ok=True)
        for split in dataset_dict:
            path = os.path.join(data_dir, f"{split}.jsonl")
            if os.path.exists(path):
                continue
            dataset_dict[split].to_json(path, num_proc=2)
        del dataset_dict


if __name__ == "__main__":
    main()
