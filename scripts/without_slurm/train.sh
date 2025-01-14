#!/usr/bin/env bash

export SLURM_JOB_ID=$(date +%Y%m%d_%H%M%S)

BASE_PATH="arxiv2025-inherent-limits-plms/src/"

python3 ${BASE_PATH}/train.py --slurm_job_id $SLURM_JOB_ID --model_path_idx 1 --model_size_idx 1 --num_epochs 1 --prompts_type 8 --num_train 20000 --batch_size 2
