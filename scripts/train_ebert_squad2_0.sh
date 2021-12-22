#!/bin/bash
export WANDB_PROJECT="EBERT-lab"
export WANDB_DISABLED="true"

RUN_NAME=bert_squad2.0_s0.8

CUDA_VISIBLE_DEVICES=0 python run_dy_squad.py \
    --model_type bert \
    --model_name_or_path ./logs/SQuAD2.0/bert_squad2.0_base \
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
    --target_flops_ratio 0.8 \
    --predictor_lr 0.02 \
    --loss_lambda 4 \
    --head_mask_mode gumbel \
    --ffn_mask_mode gumbel \
    --version_2_with_negative \
    --output_dir logs/SQuAD2.0/$RUN_NAME \
    --finetuning_task SQuAD2.0 \
    --run_name $RUN_NAME 
    