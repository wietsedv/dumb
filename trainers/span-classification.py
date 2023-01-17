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
"""
Fine-tuning the library models for token classification.
"""
# You can also adapt this script on your own token classification task and datasets. Pointers for this are left as
# comments.

import os
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

import evaluate
from transformers import (
    AutoConfig,
    AutoTokenizer,
    DataCollatorWithPadding,
    HfArgumentParser,
    PreTrainedTokenizerFast,
    Trainer,
    TrainingArguments,
    EvalPrediction,
    default_data_collator,
)

from utils import setup
from models.modeling_auto import AutoModelForSpanClassification


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """

    model_id: str = field(
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"})
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"})
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"})
    cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Where do you want to store the pretrained models downloaded from huggingface.co"},
    )
    model_revision: str = field(
        default="main",
        metadata={"help": "The specific model version to use (can be a branch name, tag name or commit id)."},
    )
    use_auth_token: bool = field(
        default=False,
        metadata={
            "help": ("Will use the token generated when running `huggingface-cli login` (necessary to use this script "
                     "with private models).")
        },
    )
    hidden_dropout_prob: float = field(
        default=0.1,
        metadata={
            "help": "The dropout probability for all fully connected layers in the embeddings, encoder, and pooler."
        })
    save_model: bool = field(
        default=False,
        metadata={"help": ("Whether to save the model after training.")},
    )


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """

    task_name: str = field(metadata={"help": "The name of the task (ner, pos...)."})
    dataset_name: str = field(metadata={"help": "The name of the dataset to use (via the datasets library)."})
    dataset_config_name: Optional[str] = field(
        default=None, metadata={"help": "The configuration name of the dataset to use (via the datasets library)."})
    overwrite_cache: bool = field(default=False, metadata={"help": "Overwrite the cached training and evaluation sets"})
    preprocessing_num_workers: Optional[int] = field(
        default=None,
        metadata={"help": "The number of processes to use for the preprocessing."},
    )
    max_seq_length: int = field(
        default=512,
        metadata={
            "help": ("The maximum total input sequence length after tokenization. If set, sequences longer "
                     "than this will be truncated, sequences shorter will be padded.")
        },
    )
    pad_to_max_length: bool = field(
        default=False,
        metadata={
            "help":
            ("Whether to pad all samples to model maximum sentence length. "
             "If False, will pad the samples dynamically when batching to the maximum length in the batch. More "
             "efficient on GPU but very bad for TPU.")
        },
    )
    max_train_samples: int = field(
        default=0,
        metadata={
            "help": ("For debugging purposes or quicker training, truncate the number of training examples to this "
                     "value if set.")
        },
    )
    # aggregate_all_tokens: bool = field(
    #     default=False,
    #     metadata={
    #         "help": ("Whether to pool all the tokens in the span instead of just taking the first.")
    #     },
    # )
    add_tokens: Optional[str] = field(default=None, metadata={"help": "Add special tokens to the vocabulary."})


def parse_columns(raw_datasets):
    column_names = raw_datasets["train"].column_names

    if "tokens1" in column_names:
        text1_column_name, text2_column_name = "tokens1", "tokens2"
    else:
        text1_column_name, text2_column_name = "tokens", None

    assert "index1" in column_names
    index1_column_name, index2_column_name = "index1", "index2"

    assert "label" in column_names
    label_column_name = "label"

    label_list = raw_datasets["train"].features[label_column_name].names
    label2id = {l: i for i, l in enumerate(label_list)}
    id2label = {i: l for i, l in enumerate(label_list)}

    return text1_column_name, text2_column_name, index1_column_name, index2_column_name, label_column_name, label2id, id2label


def main():
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    logger, last_checkpoint, raw_datasets = setup(model_args, data_args, training_args)
    text1_column_name, text2_column_name, index1_column_name, index2_column_name, label_column_name, label2id, id2label = parse_columns(
        raw_datasets)

    config = AutoConfig.from_pretrained(
        model_args.config_name if model_args.config_name else model_args.model_id,
        id2label=id2label,
        label2id=label2id,
        finetuning_task=data_args.task_name,
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        hidden_dropout_prob=model_args.hidden_dropout_prob,
        output_hidden_states=False,
        use_auth_token=True if model_args.use_auth_token else None,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.tokenizer_name if model_args.tokenizer_name else model_args.model_id,
        cache_dir=model_args.cache_dir,
        use_fast=True,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
        **({
            "add_prefix_space": True
        } if config.model_type in {"bloom", "gpt2", "roberta", "deberta"} else {}))
    model = AutoModelForSpanClassification.from_pretrained(
        model_args.model_id,
        from_tf=bool(".ckpt" in model_args.model_id),
        config=config,
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
    )
    assert isinstance(tokenizer, PreTrainedTokenizerFast), "This model does not have a Fast Tokenizer"

    if data_args.add_tokens:
        logger.info(f"Old vocabulary size: {len(tokenizer)}")
        n = tokenizer.add_tokens(data_args.add_tokens.split())
        model.resize_token_embeddings(len(tokenizer))
        logger.info(str(data_args.add_tokens))
        logger.info(f"Added {n} tokens. New vocabulary size: {len(tokenizer)}")

    padding = "max_length" if data_args.pad_to_max_length else False
    max_seq_length = tokenizer.model_max_length if data_args.max_seq_length is None else min(data_args.max_seq_length, tokenizer.model_max_length)

    # assert data_args.aggregate_all_tokens is False, "not implemented"

    # Tokenize all texts and align the labels with them.
    def prepare_for_model(examples):
        args = (
            (examples[text1_column_name],) if text2_column_name is None else (examples[text1_column_name], examples[text2_column_name])
        )
        tokenized_inputs = tokenizer(
            *args,
            padding=padding,
            truncation=True,
            max_length=max_seq_length,
            is_split_into_words=True,
        )
        # classifier_mask = []
        # for i, input_ids in enumerate(tokenized_inputs["input_ids"]):  # type: ignore
        #     span1 = tokenized_inputs.word_to_tokens(i, examples[index1_column_name][i], 0)
        #     span2 = tokenized_inputs.word_to_tokens(i, examples[index2_column_name][i], 1 if len(args) > 1 else 0)
        #     assert span1 is not None and span2 is not None, f"{span1} {span2}"
        #     start1 = span1.start
        #     start2 = span2.start
        #     mask = [1 if j in (start1, start2) else 0 for j in range(len(input_ids))]
        #     classifier_mask.append(mask)
        # tokenized_inputs["classifier_mask"] = classifier_mask

        return tokenized_inputs

    def prepare_dataset(split):
        if split not in raw_datasets:
            raise ValueError(f"requires a {split} dataset")
        split_dataset = raw_datasets[split]
        if split == "train" and data_args.max_train_samples > 0:
            max_train_samples = min(len(split_dataset), data_args.max_train_samples)
            split_dataset = split_dataset.select(range(max_train_samples))
        with training_args.main_process_first(desc=f"{split} dataset map pre-processing"):
            split_dataset = split_dataset.map(
                prepare_for_model,
                batched=True,
                num_proc=data_args.preprocessing_num_workers,
                load_from_cache_file=not data_args.overwrite_cache,
                desc=f"Running tokenizer on {split} dataset",
            )
        return split_dataset

    train_dataset = prepare_dataset("train") if training_args.do_train else None
    eval_dataset = prepare_dataset("validation") if training_args.do_eval else None
    predict_dataset = prepare_dataset("test") if training_args.do_predict else None

    # Data collator
    if data_args.pad_to_max_length:
        data_collator = default_data_collator
    elif training_args.fp16:
        data_collator = DataCollatorWithPadding(tokenizer, pad_to_multiple_of=8)
    else:
        data_collator = None

    metric = evaluate.load("accuracy")
    def compute_metrics(p: EvalPrediction):
        preds = p.predictions[0] if isinstance(p.predictions, tuple) else p.predictions
        preds = np.argmax(preds, axis=1)
        return metric.compute(predictions=preds, references=p.label_ids)

    # Initialize our Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,  # type: ignore
        eval_dataset=eval_dataset,  # type: ignore
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,  # type: ignore
    )

    # Training
    if training_args.do_train:
        assert train_dataset is not None

        checkpoint = None
        if training_args.resume_from_checkpoint is not None:
            checkpoint = training_args.resume_from_checkpoint
        elif last_checkpoint is not None:
            checkpoint = last_checkpoint
        train_result = trainer.train(resume_from_checkpoint=checkpoint)
        metrics = train_result.metrics
        if model_args.save_model:
            trainer.save_model()

        max_train_samples = (data_args.max_train_samples if data_args.max_train_samples > 0 else len(train_dataset))
        metrics["train_samples"] = min(max_train_samples, len(train_dataset))

        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()

    # Evaluation
    if training_args.do_eval:
        logger.info("*** Evaluate ***")
        metrics = trainer.evaluate()
        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)

    # Predict
    if training_args.do_predict:
        logger.info("*** Predict ***")

        predictions, _, metrics = trainer.predict(predict_dataset, metric_key_prefix="predict")  # type: ignore
        predictions = np.argmax(predictions, axis=1)
        trainer.log_metrics("predict", metrics)
        trainer.save_metrics("predict", metrics)

        # Save predictions
        output_predict_file = os.path.join(training_args.output_dir, f"predictions.txt")
        if trainer.is_world_process_zero():
            with open(output_predict_file, "w") as f:
                for prediction in predictions:
                    if id2label is not None:
                        f.write(f"{id2label[prediction]}\n")
                    else:
                        f.write(f"{prediction}\n")

    kwargs = {"finetuned_from": model_args.model_id, "tasks": "token-classification"}
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
