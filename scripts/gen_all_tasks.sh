#!/usr/bin/env bash

#SBATCH --job-name=<JOB_NAME>
#SBATCH --mail-user=<USER_EMAIL>
#SBATCH --output=<OUT_DIR>
#SBATCH --mail-type=ALL

BASE_PATH="arxiv2025-inherent-limits-plms/src/"

CUDA_LAUNCH_BLOCKING=1 python3 ${BASE_PATH}/create_prompts.py
