#!/bin/sh

# task / trainer / model
task="${task:-UNDEFINED}"
trainer="${trainer:-UNDEFINED}"
model="${model:-UNDEFINED}"

# generic arguments
eval_steps=${eval_steps:-500}
add_tokens=${add_tokens:-}
metric=${metric:-default}

# training arguments
num_train_epochs=${num_train_epochs:-3}
warmup_ratio=${warmup_ratio:-0.0}
train_batch_size=${train_batch_size:-32}
learning_rate=${learning_rate:-0.00005}
hidden_dropout_prob=${hidden_dropout_prob:-0.1}
weight_decay=${weight_decay:-0.0}
max_train_samples=${max_train_samples:-0}
seed=${seed:-639808}

# output
export WANDB_PROJECT="dumb-${task}"
run_name="e${num_train_epochs}-w${warmup_ratio}-b${train_batch_size}-l${learning_rate}-d${hidden_dropout_prob}-c${weight_decay}-m${max_train_samples}-s${seed}"
output_dir="output/${task}/${model}/${run_name}"

cmd="python3"
args=""
gradient_accumulation_steps=1

# support large models
if [ $model = "deberta-large" ] && [ $train_batch_size = "32" ]; then
    if [ $task = "dalc" ]; then
        train_batch_size=4
        gradient_accumulation_steps=8
    elif [ $task = "lassy-pos" ] || [ $task = "sonar-ne" ] || [ $task = "wicnl" ] || [ $task = "dbrd" ]; then
        train_batch_size=8
        gradient_accumulation_steps=4
    fi
elif [ $model = "deberta-v3-large" ] && [ $train_batch_size = "32" ] && [ $task = "dalc" ]; then
    train_batch_size=8
    gradient_accumulation_steps=4
elif [ $model = "deberta-v3-large" ] && [ $train_batch_size = "32" ] && [ $task = "dbrd" ]; then
    train_batch_size=8
    gradient_accumulation_steps=4
elif [[ $model == *-large* ]] && [ $train_batch_size = "32" ]; then
    if [ $task = "lassy-pos" ] || [ $task = "sonar-ne" ] || [ $task = "wicnl" ] || [ $task = "dalc" ] || [ $task = "dbrd" ] || [ $task = "squadnl" ]; then
        train_batch_size=16
        gradient_accumulation_steps=2
    fi
fi

# model path
if [ ${model} = "bertje" ]; then model_path="GroNLP/bert-base-dutch-cased"

elif [ ${model} = "bert-base" ]; then model_path="bert-base-cased"
elif [ ${model} = "roberta-base" ]; then model_path="roberta-base"
elif [ ${model} = "deberta-base" ]; then model_path="microsoft/deberta-base"
elif [ ${model} = "deberta-v3-base" ]; then model_path="microsoft/deberta-v3-base"

elif [ ${model} = "bert-large" ]; then model_path="bert-large-cased"
elif [ ${model} = "roberta-large" ]; then model_path="roberta-large"
elif [ ${model} = "deberta-large" ]; then model_path="microsoft/deberta-large"
elif [ ${model} = "deberta-v3-large" ]; then model_path="microsoft/deberta-v3-large"

elif [ ${model} = "robbert-v1" ]; then model_path="pdelobelle/robBERT-base"
elif [ ${model} = "robbert-v2" ]; then model_path="pdelobelle/robbert-v2-dutch-base"
elif [ ${model} = "robbert-2022" ]; then model_path="DTAI-KULeuven/robbert-2022-dutch-base"

elif [ ${model} = "robbert-2023-base" ]; then model_path="FremyCompany/olm-bert-oscar-nl-step4"
elif [ ${model} = "robbert-2023-large" ]; then model_path="FremyCompany/rl-bert-oscar-nl-step4"
elif [ ${model} = "robbert-2023-large-v2" ]; then model_path="FremyCompany/roberta-large-nl-oscar23"
elif [ ${model} = "robbert-2023-large-v2-sts" ]; then model_path="FremyCompany/stsb_ossts_roberta-large-nl-oscar23"

elif [ ${model} = "mbert" ]; then model_path="bert-base-multilingual-cased"
elif [ ${model} = "mbert-uncased" ]; then model_path="bert-base-multilingual-uncased"
elif [ ${model} = "xlmr-base" ]; then model_path="xlm-roberta-base"
elif [ ${model} = "xlmr-large" ]; then model_path="xlm-roberta-large"
elif [ ${model} = "mdeberta" ]; then model_path="microsoft/mdeberta-v3-base"

elif [ ${model} = "mbert-distil" ]; then model_path="distilbert-base-multilingual-cased"
elif [ ${model} = "robbertje" ]; then model_path="DTAI-KULeuven/robbertje-1-gb-non-shuffled"
elif [ ${model} = "robbertje-shuffled" ]; then model_path="DTAI-KULeuven/robbertje-1-gb-shuffled"
elif [ ${model} = "robbertje-merged" ]; then model_path="DTAI-KULeuven/robbertje-1-gb-merged"
elif [ ${model} = "robbertje-bort" ]; then model_path="DTAI-KULeuven/robbertje-1-gb-bort"

else model_path="${model:-UNDEFINED}"
fi

"${cmd}" -u trainers/${trainer}.py $args \
    --task_name "${task}" \
    --do_train \
    --do_eval \
    --do_predict \
    --evaluation_strategy no \
    --eval_steps "${eval_steps}" \
    --save_strategy no \
    --per_device_train_batch_size "${train_batch_size}" \
    --gradient_accumulation_steps "${gradient_accumulation_steps}" \
    --per_device_eval_batch_size 32 \
    --dataset_name "hf_datasets/dumb" \
    --dataset_config_name "${task}" \
    --model_id "${model_path}" \
    --output_dir "${output_dir}" \
    --run_name "${model}/${run_name}" \
    --num_train_epochs "${num_train_epochs}" \
    --warmup_ratio "${warmup_ratio}" \
    --learning_rate "${learning_rate}" \
    --hidden_dropout_prob "${hidden_dropout_prob}" \
    --weight_decay "${weight_decay}" \
    --max_train_samples "${max_train_samples}" \
    --seed "${seed}" \
    --add_tokens "${add_tokens}" \
    --metric "${metric}" \
