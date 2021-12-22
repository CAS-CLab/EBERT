#!/bin/bash 
export WANDB_PROJECT="EBERT-lab"
export WANDB_DISABLED="true"

TASK_NAME=QNLI
RUN_NAME=roberta_qnli_base

CUDA_VISIBLE_DEVICES=0 python run_glue.py \
    --model_name_or_path  ./data/roberta-base \
    --task_name $TASK_NAME \
    --data_dir ./data/GLUE/$TASK_NAME \
    --do_train \
    --do_eval \
    --evaluation_strategy epoch \
    --max_seq_length 128 \
    --per_device_train_batch_size 32 \
    --per_device_eval_batch_size 32 \
    --learning_rate 3e-5 \
    --num_train_epochs 3.0 \
    --save_steps 2000 \
    --save_total_limit 1 \
    --output_dir ./logs/$TASK_NAME/$RUN_NAME \
    --logging_dir ./logs/$TASK_NAME/$RUN_NAME \
    --logging_steps 50 \
    --run_name $RUN_NAME \
    --disable_tqdm
