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
Fine-tuning the library models for multiple choice.
"""
# You can also adapt this script on your own multiple choice task. Pointers for this are left as comments.

import os
from dataclasses import dataclass, field
from itertools import chain
from typing import Optional, Union

import numpy as np
import torch

from transformers import (
    AutoConfig,
    AutoTokenizer,
    AutoModelForMultipleChoice,
    HfArgumentParser,
    Trainer,
    TrainingArguments,
)
from transformers.tokenization_utils_base import PreTrainedTokenizerBase
from transformers.utils import PaddingStrategy

from utils import setup


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
        metadata={
            "help": ("Whether to save the model after training.")
        },
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
            "help": ("The maximum total input sequence length after tokenization. If passed, sequences longer "
                     "than this will be truncated, sequences shorter will be padded.")
        },
    )
    pad_to_max_length: bool = field(
        default=False,
        metadata={
            "help":
            ("Whether to pad all samples to the maximum sentence length. "
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
    add_tokens: Optional[str] = field(default=None, metadata={"help": "Add special tokens to the vocabulary."})


@dataclass
class DataCollatorForMultipleChoice:
    """
    Data collator that will dynamically pad the inputs for multiple choice received.

    Args:
        tokenizer ([`PreTrainedTokenizer`] or [`PreTrainedTokenizerFast`]):
            The tokenizer used for encoding the data.
        padding (`bool`, `str` or [`~utils.PaddingStrategy`], *optional*, defaults to `True`):
            Select a strategy to pad the returned sequences (according to the model's padding side and padding index)
            among:

            - `True` or `'longest'`: Pad to the longest sequence in the batch (or no padding if only a single sequence
              if provided).
            - `'max_length'`: Pad to a maximum length specified with the argument `max_length` or to the maximum
              acceptable input length for the model if that argument is not provided.
            - `False` or `'do_not_pad'` (default): No padding (i.e., can output a batch with sequences of different
              lengths).
        max_length (`int`, *optional*):
            Maximum length of the returned list and optionally padding length (see above).
        pad_to_multiple_of (`int`, *optional*):
            If set will pad the sequence to a multiple of the provided value.

            This is especially useful to enable the use of Tensor Cores on NVIDIA hardware with compute capability >=
            7.5 (Volta).
    """

    tokenizer: PreTrainedTokenizerBase
    padding: Union[bool, str, PaddingStrategy] = True
    max_length: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None

    def __call__(self, features):
        label_name = "label" if "label" in features[0].keys() else "labels"
        labels = [feature.pop(label_name) for feature in features]
        batch_size = len(features)
        num_choices = len(features[0]["input_ids"])
        flattened_features = [[{k: v[i]
                                for k, v in feature.items()} for i in range(num_choices)] for feature in features]
        flattened_features = list(chain(*flattened_features))

        batch = self.tokenizer.pad(
            flattened_features,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors="pt",
        )

        # Un-flatten
        batch = {k: v.view(batch_size, num_choices, -1) for k, v in batch.items()}
        # Add back labels
        batch["labels"] = torch.tensor(labels, dtype=torch.int64)
        return batch


def parse_columns(raw_datasets):
    column_names = raw_datasets["train"].column_names
    if "choices" not in column_names:
        raise ValueError("no choices column")
    choices_column_name = "choices"
    text1_column_name = "sentence1"
    text2_column_name = "sentence2"
    label_column_name = "label"
    return choices_column_name, text1_column_name, text2_column_name, label_column_name


def main():
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    logger, last_checkpoint, raw_datasets = setup(model_args, data_args, training_args)
    choices_column_name, text1_column_name, text2_column_name, label_column_name = parse_columns(raw_datasets)

    config = AutoConfig.from_pretrained(
        model_args.config_name if model_args.config_name else model_args.model_id,
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
        use_fast=model_args.use_fast_tokenizer,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
    )

    if model_args.model_id in ["microsoft/deberta-base", "microsoft/deberta-large"]:
        from models.modeling_deberta import DebertaForMultipleChoice
        automodel = DebertaForMultipleChoice
    else:
        automodel = AutoModelForMultipleChoice
    model = automodel.from_pretrained(
        model_args.model_id,
        from_tf=bool(".ckpt" in model_args.model_id),
        config=config,
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        # output_hidden_states=False,
        use_auth_token=True if model_args.use_auth_token else None,
    )

    max_seq_length = tokenizer.model_max_length if data_args.max_seq_length is None else min(data_args.max_seq_length, tokenizer.model_max_length)

    if data_args.add_tokens:
        logger.info(f"Old vocabulary size: {len(tokenizer)}")
        n = tokenizer.add_tokens(data_args.add_tokens.split())
        model.resize_token_embeddings(len(tokenizer))
        logger.info(str(data_args.add_tokens))
        logger.info(f"Added {n} tokens. New vocabulary size: {len(tokenizer)}")

    # Preprocessing the datasets.
    def preprocess_function(examples):
        num_choices = len(examples[choices_column_name][0][text1_column_name])
        sentences1 = [ex[text1_column_name] for ex in examples[choices_column_name]]
        sentences2 = [ex[text2_column_name] for ex in examples[choices_column_name]]
        # Flatten out
        sentences1 = list(chain(*sentences1))
        sentences2 = list(chain(*sentences2))
        # Tokenize
        tokenized_examples = tokenizer(
            sentences1,
            sentences2,
            truncation=True,
            max_length=max_seq_length,
            padding="max_length" if data_args.pad_to_max_length else False,
        )
        # Un-flatten
        return {k: [v[i:i + num_choices] for i in range(0, len(v), num_choices)] for k, v in tokenized_examples.items()}

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

    # Data collator
    data_collator = (DataCollatorForMultipleChoice(tokenizer=tokenizer,
                                                   pad_to_multiple_of=8 if training_args.fp16 else None))

    # Metric
    def compute_metrics(eval_predictions):
        predictions, label_ids = eval_predictions
        preds = np.argmax(predictions, axis=1)
        return {"accuracy": (preds == label_ids).astype(np.float32).mean().item()}

    # Initialize our Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,  # type: ignore
        eval_dataset=eval_dataset,  # type: ignore
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    # Training
    if training_args.do_train:
        checkpoint = None
        if training_args.resume_from_checkpoint is not None:
            checkpoint = training_args.resume_from_checkpoint
        elif last_checkpoint is not None:
            checkpoint = last_checkpoint
        train_result = trainer.train(resume_from_checkpoint=checkpoint)
        if model_args.save_model:
            trainer.save_model()  # Saves the tokenizer too for easy upload
        metrics = train_result.metrics

        max_train_samples = (
            data_args.max_train_samples
            if data_args.max_train_samples > 0 else len(train_dataset)  # type: ignore
        )
        metrics["train_samples"] = min(max_train_samples, len(train_dataset))  # type: ignore

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
        predictions = np.squeeze(predictions)
        trainer.log_metrics("predict", metrics)
        trainer.save_metrics("predict", metrics)

        # Save predictions
        output_predictions_file = os.path.join(training_args.output_dir, "predictions.txt")
        if trainer.is_world_process_zero():
            with open(output_predictions_file, "w") as writer:
                for prediction in predictions:
                    writer.write(f"{prediction.argmax()}\n")

    kwargs = {"finetuned_from": model_args.model_id, "tasks": "multiple-choice"}
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
