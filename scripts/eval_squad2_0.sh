#!/bin/bash
export WANDB_DISABLED="true"

RUN_NAME=test
MODEL_PATH=

CUDA_VISIBLE_DEVICES=0 python run_dy_squad.py \
  --model_type bert \
  --model_name_or_path  $MODEL_PATH \
  --data_dir ./data/SQuAD2.0 \
  --do_eval \
  --do_lower_case \
  --train_file train-v2.0.json \
  --predict_file dev-v2.0.json \
  --per_gpu_eval_batch_size 8 \
  --max_seq_length 384 \
  --doc_stride 128 \
  --output_dir logs/test/SQuAD2.0/$RUN_NAME \
  --run_name $RUN_NAME \
  --version_2_with_negative