#!/bin/bash 
export WANDB_PROJECT="EBERT-lab"
export WANDB_DISABLED="true"

TASK_NAME=SST-2
RUN_NAME=bert_sst_2_s0.8

CUDA_VISIBLE_DEVICES=0 python run_dy_glue.py \
    --model_name_or_path ./logs/SST-2/bert_sst_2_base \
    --task_name $TASK_NAME \
    --data_dir ./data/GLUE/$TASK_NAME \
    --do_train \
    --do_eval \
    --evaluation_strategy epoch \
    --max_seq_length 64 \
    --per_device_train_batch_size 32 \
    --per_device_eval_batch_size 32 \
    --learning_rate 3e-5 \
    --num_train_epochs 3.0 \
    --target_flops_ratio 0.8 \
    --predictor_lr 0.02 \
    --loss_lambda 4 \
    --head_mask_mode gumbel \
    --ffn_mask_mode gumbel \
    --fill_mode zero \
    --load_best_model_at_end \
    --output_dir ./logs/$TASK_NAME/$RUN_NAME \
    --logging_dir ./logs/$TASK_NAME/$RUN_NAME \
    --logging_steps 50 \
    --run_name $RUN_NAME \
    --disable_tqdm
