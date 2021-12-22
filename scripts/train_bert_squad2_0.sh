#!/bin/bash
export WANDB_PROJECT="EBERT-lab"
export WANDB_DISABLED="true"

RUN_NAME=bert_squad2.0_base

CUDA_VISIBLE_DEVICES=0 python run_squad.py \
    --model_type bert \
    --model_name_or_path ./data/bert-base-uncased \
    --data_dir ./data/SQuAD2.0 \
    --do_train \
    --do_eval \
    --do_lower_case \
    --train_file train-v2.0.json \
    --predict_file dev-v2.0.json \
    --per_gpu_train_batch_size 12 \
    --per_gpu_eval_batch_size 8 \
    --learning_rate 3e-5 \
    --num_train_epochs 2.0 \
    --max_seq_length 384 \
    --doc_stride 128 \
    --logging_steps 100 \
    --save_steps 0 \
    --output_dir logs/SQuAD2.0/$RUN_NAME \
    --version_2_with_negative \
    --finetuning_task SQuAD2.0 \
    --run_name $RUN_NAME
  