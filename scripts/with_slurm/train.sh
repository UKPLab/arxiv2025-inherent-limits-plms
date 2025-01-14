#!/usr/bin/env bash

#SBATCH --job-name=<JOB_NAME>
#SBATCH --mail-user=<USER_EMAIL>
#SBATCH --output=<OUT_DIR>
#SBATCH --mail-type=ALL

BASE_PATH="arxiv2025-inherent-limits-plms/src/"

python3 ${BASE_PATH}/train.py --slurm_job_id $SLURM_JOB_ID --model_path_idx 1 --model_size_idx 1 --num_epochs 1 --prompts_type 8 --num_train 20000 --batch_size 2
