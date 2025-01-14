#!/usr/bin/env bash

BASE_PATH="arxiv2025-inherent-limits-plms/src/"

python3 ${BASE_PATH}/create_prompts.py --num_datapoints 100 --create_custom_prompts "True" --dataset_type "train+val"
