#!/usr/bin/env bash

task=dalc
trainer=sequence-classification
add_tokens="@USER URL"
metric=f1

eval_steps=250

. trainers/run.sh
