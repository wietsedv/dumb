#!/usr/bin/env bash

task=sonar-ne
trainer=token-classification

eval_steps=500

. trainers/run.sh
