#!/usr/bin/env bash

#SBATCH --job-name=<JOB_NAME>
#SBATCH --mail-user=<USER_EMAIL>
#SBATCH --output=<OUT_DIR>
#SBATCH --mail-type=ALL

BASE_PATH="arxiv2025-inherent-limits-plms/src/"

python3 ${BASE_PATH}/create_prompts.py --num_datapoints 100 --create_custom_prompts "True" --dataset_type "train+val"
