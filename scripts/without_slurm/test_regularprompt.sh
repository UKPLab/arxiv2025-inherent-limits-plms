#!/usr/bin/env bash

export SLURM_JOB_ID=$(date +%Y%m%d_%H%M%S)

BASE_PATH="arxiv2025-inherent-limits-plms/src/"

python3 ${BASE_PATH}/test.py --slurm_job_id $SLURM_JOB_ID --model_path_idx 1 --model_size_idx 1 --test_task 0 --run_name '20241022_233143' --num_test 2500 --batch_size 10 --test_prompt_format "mix_prompts" --bb_test_prompt_format "closed" --unified_prompts "False" --use_exemplar_gen "False" --use_vllm "True"
