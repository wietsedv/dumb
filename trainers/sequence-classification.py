#!/usr/bin/env python3
# coding=utf-8
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
""" Finetuning the library models for sequence classification."""
# You can also adapt this script on your own text classification task. Pointers for this are left as comments.

import os
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

import evaluate
import transformers
from transformers import (
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    EvalPrediction,
    HfArgumentParser,
    PreTrainedModel,
    PreTrainedTokenizerFast,
    Trainer,
    TrainingArguments,
    default_data_collator,
)
from torch import nn

from utils import setup

@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.

    Using `HfArgumentParser` we can turn this class
    into argparse arguments to be able to specify them on
    the command line.
    """

    task_name: str = field(
        metadata={"help": "The name of the task."},
    )
    dataset_name: str = field(metadata={"help": "The name of the dataset to use (via the datasets library)."}
    )
    dataset_config_name: Optional[str] = field(
        default=None, metadata={"help": "The configuration name of the dataset to use (via the datasets library)."}
    )
    max_seq_length: int = field(
        default=512,
        metadata={
            "help": (
                "The maximum total input sequence length after tokenization. Sequences longer "
                "than this will be truncated, sequences shorter will be padded."
            )
        },
    )
    overwrite_cache: bool = field(
        default=False, metadata={"help": "Overwrite the cached preprocessed datasets or not."}
    )
    pad_to_max_length: bool = field(
        default=False,
        metadata={
            "help": (
                "Whether to pad all samples to `max_seq_length`. "
                "If False, will pad the samples dynamically when batching to the maximum length in the batch."
            )
        },
    )
    max_train_samples: int = field(
        default=0,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of training examples to this "
                "value if set."
            )
        },
    )
    add_tokens: Optional[str] = field(default=None, metadata={"help": "Add special tokens to the vocabulary."})
    metric: str = field(default="default", metadata={"help": "Metric for evaluation."})

@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """

    model_id: str = field(
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Where do you want to store the pretrained models downloaded from huggingface.co"},
    )
    use_fast_tokenizer: bool = field(
        default=True,
        metadata={"help": "Whether to use one of the fast tokenizer (backed by the tokenizers library) or not."},
    )
    model_revision: str = field(
        default="main",
        metadata={"help": "The specific model version to use (can be a branch name, tag name or commit id)."},
    )
    use_auth_token: bool = field(
        default=False,
        metadata={
            "help": (
                "Will use the token generated when running `huggingface-cli login` (necessary to use this script "
                "with private models)."
            )
        },
    )
    hidden_dropout_prob: float = field(
        default=0.1,
        metadata={
        "help": "The dropout probability for all fully connected layers in the embeddings, encoder, and pooler."
    })
    save_model: bool = field(
        default=False,
        metadata={
            "help": ("Whether to save the model after training.")
        },
    )
    # reset_weights: bool = field(default=False, metadata={"help": "Whether to reset all model weights before training."})

def parse_columns(raw_datasets):
    column_names = raw_datasets["train"].column_names

    # Labels
    is_regression = "score" in column_names
    if is_regression:
        label_column_name = "score"
        label_list = []
        label2id = None
        id2label = None
    else:
        label_column_name = "label"
        label_list = raw_datasets["train"].features[label_column_name].names  # type: ignore
        label2id = {l: i for i, l in enumerate(label_list)}
        id2label = {i: l for i, l in enumerate(label_list)}
    
    # Preprocessing the raw_datasets
    if "text" in column_names:
        text1_column_name, text2_column_name = "text", None
    # elif "tokens" in column_names:
    #     sentence1_key, sentence2_key = "tokens", None
    elif "sentence1" in column_names:
        text1_column_name, text2_column_name = "sentence1", "sentence2"
    # elif "text1" in column_names:
    #     sentence1_key, sentence2_key = "text1", "text2"
    else:
        raise ValueError("cannot find columns")

    return text1_column_name, text2_column_name, label_column_name, label2id, id2label



def main():
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    logger, last_checkpoint, raw_datasets = setup(model_args, data_args, training_args)
    text1_column_name, text2_column_name, label_column_name, label2id, id2label = parse_columns(raw_datasets)

    is_regression = label_column_name == "score"

    config = AutoConfig.from_pretrained(
        model_args.config_name if model_args.config_name else model_args.model_id,
        **({"num_labels": 1} if is_regression else {"label2id": label2id, "id2label": id2label}),
        finetuning_task=data_args.task_name,
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        hidden_dropout_prob=model_args.hidden_dropout_prob,
        use_auth_token=True if model_args.use_auth_token else None,
        output_hidden_states=False,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.tokenizer_name if model_args.tokenizer_name else model_args.model_id,
        cache_dir=model_args.cache_dir,
        use_fast=model_args.use_fast_tokenizer,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
    )
    model = AutoModelForSequenceClassification.from_pretrained(
        model_args.model_id,
        from_tf=bool(".ckpt" in model_args.model_id),
        config=config,
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
    )
    assert isinstance(tokenizer, PreTrainedTokenizerFast), "This model does not have a Fast Tokenizer"
    assert isinstance(model, PreTrainedModel)

    # if model_args.reset_weights:
    #     def _reset_weights(m):
    #         nn.init.kaiming_uniform_(m.weight)
    #     model.apply(_reset_weights)

    padding = "max_length" if data_args.pad_to_max_length else False
    max_seq_length = tokenizer.model_max_length if data_args.max_seq_length is None else min(data_args.max_seq_length, tokenizer.model_max_length)

    if data_args.add_tokens:
        logger.info(f"Old vocabulary size: {len(tokenizer)}")
        n = tokenizer.add_tokens(data_args.add_tokens.split())
        model.resize_token_embeddings(len(tokenizer))
        logger.info(str(data_args.add_tokens))
        logger.info(f"Added {n} tokens. New vocabulary size: {len(tokenizer)}")

    def preprocess_function(examples):
        # Tokenize the texts
        args = (
            (examples[text1_column_name],) if text2_column_name is None else (examples[text1_column_name], examples[text2_column_name])
        )
        tokenized_inputs = tokenizer(*args, padding=padding, max_length=max_seq_length, truncation=True)
        if label_column_name == "score":
            tokenized_inputs["labels"] = examples["score"]
        return tokenized_inputs

    def prepare_dataset(split):
        if split not in raw_datasets:
            raise ValueError("requires a train dataset")
        split_dataset = raw_datasets[split]
        if split == "train" and data_args.max_train_samples > 0:
            max_train_samples = min(len(split_dataset), data_args.max_train_samples)
            split_dataset = split_dataset.select(range(max_train_samples))
        with training_args.main_process_first(desc=f"{split} dataset map pre-processing"):
            split_dataset = split_dataset.map(
                preprocess_function,
                batched=True,
                load_from_cache_file=not data_args.overwrite_cache,
                desc=f"Running tokenizer on {split} dataset",
            )
        return split_dataset

    train_dataset = prepare_dataset("train") if training_args.do_train else None
    eval_dataset = prepare_dataset("validation") if training_args.do_eval else None
    predict_dataset = prepare_dataset("test") if training_args.do_predict else None

    # Get the metric function
    if data_args.metric != "default":
        metric_path = data_args.metric
    elif is_regression:
        metric_path = "pearsonr"
    # elif num_labels > 2:
    #     metric_path = "f1"
    else:
        metric_path = "accuracy"
    metric = evaluate.load(metric_path)

    # You can define your custom compute_metrics function. It takes an `EvalPrediction` object (a namedtuple with a
    # predictions and label_ids field) and has to return a dictionary string to float.
    def compute_metrics(p: EvalPrediction):
        preds = p.predictions[0] if isinstance(p.predictions, tuple) else p.predictions
        preds = np.squeeze(preds) if is_regression else np.argmax(preds, axis=1)
        if metric_path == "f1":
            return metric.compute(predictions=preds, references=p.label_ids, average="macro")
        return metric.compute(predictions=preds, references=p.label_ids)

    # Data collator will default to DataCollatorWithPadding when the tokenizer is passed to Trainer, so we change it if
    # we already did the padding.
    if data_args.pad_to_max_length:
        data_collator = default_data_collator
    elif training_args.fp16:
        data_collator = DataCollatorWithPadding(tokenizer, pad_to_multiple_of=8)
    else:
        data_collator = None

    # Initialize our Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,  # type: ignore
        eval_dataset=eval_dataset,  # type: ignore
        compute_metrics=compute_metrics,  # type: ignore
        tokenizer=tokenizer,
        data_collator=data_collator,
    )

    # Training
    if training_args.do_train:
        checkpoint = None
        if training_args.resume_from_checkpoint is not None:
            checkpoint = training_args.resume_from_checkpoint
        elif last_checkpoint is not None:
            checkpoint = last_checkpoint
        train_result = trainer.train(resume_from_checkpoint=checkpoint)
        metrics = train_result.metrics
        max_train_samples = (
            data_args.max_train_samples if data_args.max_train_samples > 0 else len(train_dataset)  # type: ignore
        )
        metrics["train_samples"] = min(max_train_samples, len(train_dataset))  # type: ignore

        if model_args.save_model:
            trainer.save_model()  # Saves the tokenizer too for easy upload
        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()

    # Evaluation
    if training_args.do_eval:
        logger.info("*** Evaluate ***")

        metrics = trainer.evaluate(eval_dataset=eval_dataset)  # type: ignore
        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)

    if training_args.do_predict:
        logger.info("*** Predict ***")

        predictions, _, metrics = trainer.predict(predict_dataset, metric_key_prefix="predict")  # type: ignore
        predictions = np.squeeze(predictions) if is_regression else np.argmax(predictions, axis=1)
        trainer.log_metrics("predict", metrics)
        trainer.save_metrics("predict", metrics)

        # Save predictions
        output_predict_file = os.path.join(training_args.output_dir, f"predictions.txt")
        if trainer.is_world_process_zero():
            with open(output_predict_file, "w") as f:
                for prediction in predictions:
                    if is_regression:
                        f.write(f"{prediction:3.3f}\n")
                    elif id2label is not None:
                        f.write(f"{id2label[prediction]}\n")
                    else:
                        f.write(f"{prediction}\n")

    kwargs = {"finetuned_from": model_args.model_id, "tasks": "text-classification"}
    if data_args.dataset_name is not None:
        kwargs["dataset_tags"] = data_args.dataset_name
        if data_args.dataset_config_name is not None:
            kwargs["dataset_args"] = data_args.dataset_config_name
            kwargs["dataset"] = f"{data_args.dataset_name} {data_args.dataset_config_name}"
        else:
            kwargs["dataset"] = data_args.dataset_name

    if training_args.push_to_hub:
        trainer.push_to_hub(**kwargs)
    else:
        trainer.create_model_card(**kwargs)


def _mp_fn(index):
    # For xla_spawn (TPUs)
    main()


if __name__ == "__main__":
    main()
