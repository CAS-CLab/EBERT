#!/bin/bash 
export WANDB_DISABLED="true"

TASK_NAME=SST-2
MODEL_PATH=

CUDA_VISIBLE_DEVICES=0 python run_dy_glue.py \
  --model_name_or_path $MODEL_PATH \
  --task_name $TASK_NAME \
  --data_dir ./data/GLUE/$TASK_NAME \
  --max_seq_length 64 \
  --per_device_eval_batch_size 64 \
  --output_dir ./logs/test/$TASK_NAME/test \
  --do_eval