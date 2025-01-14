#!/usr/bin/env bash

#SBATCH --job-name=<JOB_NAME>
#SBATCH --mail-user=<USER_EMAIL>
#SBATCH --output=<OUT_DIR>
#SBATCH --mail-type=ALL

BASE_PATH="arxiv2025-inherent-limits-plms/src/"

python3 ${BASE_PATH}/test.py --slurm_job_id $SLURM_JOB_ID --model_path_idx 1 --model_size_idx 1 --test_task 29 --run_name '20241022_233143' --num_test 2500 --batch_size 10 --test_prompt_format "mix_prompts" --bb_test_prompt_format "closed" --unified_prompts "False" --use_exemplar_gen "False" --use_vllm "True"
python3 ${BASE_PATH}/test.py --slurm_job_id $SLURM_JOB_ID --model_path_idx 1 --model_size_idx 1 --test_task 29 --run_name 'base' --samplegen_model "20241022_233143" --batch_size 10 --num_test 2500 --sample_source "train_data" --test_prompt_format "flan_prompts" --bb_test_prompt_format "closed" --unified_prompts "False" --use_exemplar_gen "True" --use_vllm "True"
python3 ${BASE_PATH}/test.py --slurm_job_id $SLURM_JOB_ID --model_path_idx 1 --model_size_idx 1 --test_task 29 --run_name 'base' --samplegen_model "20241022_233143" --batch_size 10 --num_test 2500 --sample_source "model" --test_prompt_format "flan_prompts" --bb_test_prompt_format "closed" --unified_prompts "False" --use_exemplar_gen "True" --use_vllm "True"
