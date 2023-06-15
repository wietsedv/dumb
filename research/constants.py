BASELINE_MODEL = "bertje"

MODEL_ORDER = [
    # dutch
    "bertje",
    "robbert-v1",
    "robbert-v2",
    "robbert-2022",

    # multilingual
    "mbert",
    "xlmr-base",
    "mdeberta",
    "xlmr-large",

    # english
    "bert-base",
    "roberta-base",
    "deberta-v3-base",
    "bert-large",
    "roberta-large",
    "deberta-v3-large",

    # distilled
    # "mbert-distil",
    # "robbertje",
    # "robbertje-shuffled",
    # "robbertje-merged",
    # "robbertje-bort",
]

MODEL_GROUPS = {
    "baseline": ["bertje"],
    "dutch-base": ["robbert-v1", "robbert-v2", "robbert-2022"],
    "multi-base": ["mbert", "xlmr-base", "mdeberta"],
    "multi-large": ["xlmr-large"],
    "english-base": ["bert-base", "roberta-base", "deberta-v3-base"],
    "english-large": ["bert-large", "roberta-large", "deberta-v3-large"],
    # "distilled": ["mbert-distil", "robbertje", "robbertje-shuffled", "robbertje-merged", "robbertje-bort"],
}

MODEL_IDS = {
    "bertje": "GroNLP/bert-base-dutch-cased",
    "robbert-v1": "pdelobelle/robBERT-base",
    "robbert-v2": "pdelobelle/robbert-v2-dutch-base",
    "robbert-2022": "DTAI-KULeuven/robbert-2022-dutch-base",
    "bert-base": "bert-base-cased",
    "roberta-base": "roberta-base",
    "deberta-base": "microsoft/mdeberta-base",
    "deberta-v3-base": "microsoft/mdeberta-v3-base",
    "bert-large": "bert-large-cased",
    "roberta-large": "roberta-large",
    "deberta-large": "microsoft/mdeberta-large",
    "deberta-v3-large": "microsoft/mdeberta-v3-large",
    "mbert": "bert-base-multilingual-cased",
    # "mbert-uncased": "bert-base-multilingual-uncased",
    "xlmr-base": "xlm-roberta-base",
    "xlmr-large": "xlm-roberta-large",
    "mdeberta": "microsoft/mdeberta-v3-base",
    "robbertje": "DTAI-KULeuven/robbertje-1-gb-non-shuffled",
    "robbertje-shuffled": "DTAI-KULeuven/robbertje-1-gb-shuffled",
    "robbertje-merged": "DTAI-KULeuven/robbertje-1-gb-merged",
    "robbertje-bort": "DTAI-KULeuven/robbertje-1-gb-bort",
    "mbert-distil": "distilbert-base-multilingual-cased",
}

MODEL_PRETTY = {
    "bertje": "BERTje",
    "robbert-v1": "RobBERT\\textsubscript{V1}",
    "robbert-v2": "RobBERT\\textsubscript{V2}",
    "robbert-2022": "RobBERT\\textsubscript{2022}",
    "bert-base": "BERT\\textsubscript{base}",
    "roberta-base": "RoBERTa\\textsubscript{base}",
    "deberta-base": "DeBERTa\\textsubscript{base}",
    "deberta-v3-base": "DeBERTaV3\\textsubscript{base}",
    "bert-large": "BERT\\textsubscript{large}",
    "roberta-large": "RoBERTa\\textsubscript{large}",
    "deberta-large": "DeBERTa\\textsubscript{large}",
    "deberta-v3-large": "DeBERTaV3\\textsubscript{large}",
    "mbert": "mBERT\\textsubscript{cased}",
    # "mbert-uncased": "mBERT\\textsubscript{uncased}",
    "xlmr-base": "XLM-R\\textsubscript{base}",
    "xlmr-large": "XLM-R\\textsubscript{large}",
    "mdeberta": "mDeBERTaV3\\textsubscript{base}",
    "robbertje": "RobBERTje",
    "robbertje-shuffled": "RobBERTje\\textsubscript{shuf}",
    "robbertje-merged": "RobBERTje\\textsubscript{merged}",
    "robbertje-bort": "RobBERTje\\textsubscript{bort}",
    "mbert-distil": "DistilBERT\\textsubscript{mBERT}",
}

# MODEL_HTML = {
#     "bertje": "🇳🇱&nbsp;BERTje",
#     "robbert-v1": "🇳🇱&nbsp;RobBERT<sub>V1</sub>",
#     "robbert-v2": "🇳🇱&nbsp;RobBERT<sub>V2</sub>",
#     "robbert-2022": "🇳🇱&nbsp;RobBERT<sub>2022</sub>",

#     "bert-base": "🇺🇸&nbsp;BERT<sub>base</sub>",
#     "roberta-base": "🇺🇸&nbsp;RoBERTa<sub>base</sub>",
#     "deberta-v3-base": "🇺🇸&nbsp;DeBERTaV3<sub>base</sub>",

#     "bert-large": "🇺🇸&nbsp;BERT<sub>large</sub>",
#     "roberta-large": "🇺🇸&nbsp;RoBERTa<sub>large</sub>",
#     "deberta-v3-large": "🇺🇸&nbsp;DeBERTaV3<sub>large</sub>",

#     "mbert": "🌍&nbsp;mBERT<sub>cased</sub>",
#     # "mbert-uncased": "🌍&nbsp;mBERT<sub>uncased</sub>",
#     "xlmr-base": "🌍&nbsp;XLM-R<sub>base</sub>",
#     "xlmr-large": "🌍&nbsp;XLM-R<sub>large</sub>",
#     "mdeberta": "🌍&nbsp;mDeBERTa<sub>base</sub>",

#     "robbertje": "🇳🇱&nbsp;RobBERTje",
#     "mbert-distil": "🌍&nbsp;DistilBERT<sub>mBERT</sub>",
# }

MODEL_INFO = {
    "bertje": {
        "name": "BERTje",
        "lang": "dutch",
        "type": "bert",
        "size": "base"
    },
    "robbert-v1": {
        "name": "RobBERT<sub>V1</sub>",
        "lang": "dutch",
        "type": "roberta",
        "size": "base"
    },
    "robbert-v2": {
        "name": "RobBERT<sub>V2</sub>",
        "lang": "dutch",
        "type": "roberta",
        "size": "base"
    },
    "robbert-2022": {
        "name": "RobBERT<sub>2022</sub>",
        "lang": "dutch",
        "type": "roberta",
        "size": "base"
    },
    "bert-base": {
        "name": "BERT<sub>base</sub>",
        "lang": "english",
        "type": "bert",
        "size": "base"
    },
    "roberta-base": {
        "name": "RoBERTa<sub>base</sub>",
        "lang": "english",
        "type": "roberta",
        "size": "base"
    },
    "deberta-v3-base": {
        "name": "DeBERTaV3<sub>base</sub>",
        "lang": "english",
        "type": "debertav3",
        "size": "base"
    },
    "bert-large": {
        "name": "BERT<sub>large</sub>",
        "lang": "english",
        "type": "bert",
        "size": "large"
    },
    "roberta-large": {
        "name": "RoBERTa<sub>large</sub>",
        "lang": "english",
        "type": "roberta",
        "size": "large"
    },
    "deberta-large": {
        "name": "DeBERTaV3<sub>large</sub>",
        "lang": "english",
        "type": "deberta",
        "size": "large"
    },
    "deberta-v3-large": {
        "name": "DeBERTaV3<sub>large</sub>",
        "lang": "english",
        "type": "debertav3",
        "size": "large"
    },
    "mbert": {
        "name": "mBERT<sub>cased</sub>",
        "lang": "multilingual",
        "type": "bert",
        "size": "base"
    },
    "mbert-uncased": {
        "name": "mBERT<sub>uncased</sub>",
        "lang": "multilingual",
        "type": "bert",
        "size": "base"
    },
    "xlmr-base": {
        "name": "XLM-R<sub>base</sub>",
        "lang": "multilingual",
        "type": "roberta",
        "size": "base"
    },
    "xlmr-large": {
        "name": "XLM-R<sub>large</sub>",
        "lang": "multilingual",
        "type": "roberta",
        "size": "large"
    },
    "mdeberta": {
        "name": "mDeBERTaV3<sub>base</sub>",
        "lang": "multilingual",
        "type": "debertav3",
        "size": "base"
    },
    "mbert-distil": {
        "name": "DistilBERT<sub>mBERT</sub>",
        "lang": "multilingual",
        "type": "bert",
        "size": "base-distil"
    },
    "robbertje": {
        "name": "RobBERTje<sub>unshuffled</sub>",
        "lang": "dutch",
        "type": "roberta",
        "size": "base-distil"
    },
    "robbertje-shuffled": {
        "name": "RobBERTje<sub>shuffled</sub>",
        "lang": "dutch",
        "type": "roberta",
        "size": "base-distil"
    },
    "robbertje-merged": {
        "name": "RobBERTje<sub>merged</sub>",
        "lang": "dutch",
        "type": "roberta",
        "size": "base-distil"
    },
    "robbertje-bort": {
        "name": "RobBERTje<sub>bort</sub>",
        "lang": "dutch",
        "type": "roberta",
        "size": "base-distil"
    },
}

MODEL_EMOJI = {
    "bert-base": "english",
    "roberta-base": "english",
    "deberta-v3-base": "english",
    "bert-large": "english",
    "roberta-large": "english",
    "deberta-large": "english",
    "deberta-v3-large": "english",
    "bertje": "dutch",
    "robbert-v1": "dutch",
    "robbert-v2": "dutch",
    "robbertje": "dutch",
    "robbertje-shuffled": "dutch",
    "robbertje-merged": "dutch",
    "robbertje-bort": "dutch",
    "robbert-2022": "dutch",
    "mbert": "multi",
    "mbert-uncased": "multi",
    "mbert-distil": "multi",
    "xlmr-base": "multi",
    "xlmr-large": "multi",
    "mdeberta": "multi",
}

TASK_ORDER = [
    "lassy-pos",
    "sonar-ne",
    "wicnl",
    "dpr",
    "copanl",
    "sicknl-nli",
    "dbrd",
    "dalc",
    "squadnl",
]

TASK_GROUPS = {
    "Word": ["lassy-pos", "sonar-ne"],
    "Word Pair": ["wicnl", "dpr"],
    "Sentence Pair": [
        "copanl",
        "sicknl-nli"
    ],
    "Document": ["dbrd", "dalc", "squadnl"],
}

TASK_PRETTY = {
    "lassy-pos": "POS",
    "sonar-ne": "NER",
    "wicnl": "WSD",
    "dpr": "PR",
    "copanl": "CR",
    "sicknl-nli": "NLI",
    "dbrd": "SA",
    "dalc": "ALD",
    "squadnl": "QA",
}

TASK_METRICS_PRETTY = {
    "lassy-pos": "Acc.",
    "sonar-ne": "F1",
    "wicnl": "Acc.",
    "dpr": "Acc.",
    "copanl": "Acc.",
    "sicknl-nli": "Acc.",
    "dbrd": "Acc.",
    "dalc": "F1",
    "squadnl": "F1",
}

CONFIG_KEYS = {
    "e": "num_train_epochs",
    "w": "warmup_ratio",
    "b": "train_batch_size",
    "l": "learning_rate",
    "d": "hidden_dropout_prob",
    "c": "weight_decay",
}

EVAL_SEED = 639808

def get_train_params(task):
    WHITELIST = {
        "e": [1, 3, 5],
        "b": [32],
        "w": [0.0, 0.3],
        "l": [0.00001, 0.00003, 0.00005, 0.0001],
        "d": [0.0, 0.1],
        "c": [0.0],
    }

    TASK_WHITELIST = {
        "lassy-pos": {
            "l": [0.00003, 0.00005, 0.0001],
        },
        "sonar-ne": {
            "l": [0.00001, 0.00003, 0.00005],  # , 0.0001
        },

        # token-classification
        "wicnl": {
            "e": [1, 3, 5, 10],
        },
        "dpr": {
            "e": [1, 3, 5, 10, 20],
            # "c": [0.0, 0.01, 0.1],
        },

        # multiple-choice
        "copanl": {
            "e": [1, 3, 5, 10, 20],
            "c": [0.0, 0.1],
        },

        # sequence-classification
        "sicknl-nli": {
            "e": [1, 3, 5, 10],
        },
        "dalc": {
            "e": [1, 3, 5, 10],
            "l": [0.00003, 0.00005, 0.0001],
        },
        "dbrd": {
            "e": [1, 2, 3],
            "l": [0.00003, 0.00005, 0.0001],
        },

        # question answering
        "squadnl": {
            "e": [2],
            "l": [0.00003, 0.00005, 0.0001],  # 0.00001
        },
    }
    if task in TASK_WHITELIST:
        return {k: TASK_WHITELIST[task].get(k, WHITELIST[k]) for k in WHITELIST}
    return WHITELIST


def get_test_seeds(task, model):
    # # TEST_SEEDS = [639808, 107584, 251499, 370845, 890307] + [422136, 903324, 196100, 474114, 627276]

    # TASK_MODEL_SEEDS = {
    #     ("wicnl", "bertje"): [639808, 107584, 370845, 890307, 422136],
    #     ("wicnl", "robbert-v2"): [639808, 107584, 251499, 370845, 422136],
    #     ("dpr", "bertje"): [639808, 251499, 370845, 890307, 422136],
    #     ("dpr", "xlmr-base"): [639808, 107584, 251499, 370845, 196100],
    #     ("copanl", "mbert"): [639808, 370845, 890307, 903324, 627276],
    #     ("copanl", "xlmr-base"): [639808, 107584, 370845, 890307, 422136],
    #     ("sicknl-nli", "roberta-base"): [639808, 107584, 251499, 890307, 422136]

    #     # ("", ""): [639808, 107584, 251499, 370845, 890307],
    # }

    # seeds = TASK_MODEL_SEEDS.get((task, model), [639808, 107584, 251499, 370845, 890307])
    # return seeds
    return [639808, 107584, 251499, 370845, 890307]
