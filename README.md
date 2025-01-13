# The Inherent Limits of Pretrained LLMs: The Unexpected Convergence of Instruction Tuning and In-Context Learning Capabilities
[![Arxiv](https://img.shields.io/badge/Arxiv-YYMM.NNNNN-red?style=flat-square&logo=arxiv&logoColor=white)](https://put-here-your-paper.com)
[![License](https://img.shields.io/github/license/UKPLab/arxiv2025-inherent-limits-plms)](https://opensource.org/licenses/Apache-2-0)
[![Python Versions](https://img.shields.io/badge/Python-3.9-blue.svg?style=flat&logo=python&logoColor=white)](https://www.python.org/)
[![CI](https://github.com/UKPLab/arxiv2025-inherent-limits-plms/actions/workflows/main.yml/badge.svg)](https://github.com/UKPLab/arxiv2025-inherent-limits-plms/actions/workflows/main.yml)

This is the accompanying code repository for the paper: [The Inherent Limits of Pretrained LLMs: The Unexpected Convergence of Instruction Tuning and In-Context Learning Capabilities](https://github.com/rochacbruno/python-project-template/).

> **Abstract:** Large Language Models (LLMs), trained on extensive web-scale corpora, have demonstrated remarkable abilities across diverse tasks, especially as they are scaled up. Nevertheless, even state-of-the-art models struggle in certain cases, sometimes failing at problems solvable by young children, indicating that traditional notions of task complexity are insufficient for explaining LLM capabilities. However, exploring LLM capabilities is complicated by the fact that most widely-used models are also `instruction-tuned' to respond appropriately to prompts. With the goal of disentangling the factors influencing LLM performance, we investigate whether instruction-tuned models possess fundamentally different capabilities from base models that are prompted using in-context examples. Through extensive experiments across various model families, scales and task types, which included instruction tuning 90 different LLMs, we demonstrate that the performance of instruction-tuned models is significantly correlated with the in-context performance of their base counterparts. By clarifying what instruction-tuning contributes, we extend prior research into in-context learning, which suggests that base models use priors from pretraining data to solve tasks. Specifically, we extend this understanding to instruction-tuned models, suggesting that their pretraining data similarly sets a limiting boundary on the tasks they can solve, with the added influence of the instruction-tuning dataset.

Contact person: [Irina Bigoulaeva](mailto:irina.bigoulaeva@gmail.com) 

[UKP Lab](https://www.ukp.tu-darmstadt.de/) | [TU Darmstadt](https://www.tu-darmstadt.de/
)


## Getting Started

Prepare a virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate
pip install .
pip install -r requirements.txt
```
We recommend using a separate virtual environment, since we use [vLLM](https://docs.vllm.ai/en/stable/) for speeding up inference. As stated in their documentation, it is recommended to install vLLM in a fresh environment, since it compiles specific CUDA versions that may be incompatible with preexisting environments.

The current version of the code is designed to run as individual modules called by scripts. Sample scripts can be found in the `scripts` folder. Model training and inference is designed to run on a SLURM scheduling system. In particular, a `$SLURM_JOB_ID` is needed in all scripts, and is additionally invoked by `train.py` to record run info, as well as by `test.py` and `bertscore.py` to retrieve relevant evaluation files.

## Reproducing the Experiments

Our experiments can be reproduced in the following steps:

1. Dataset Creation
2. Model Training

## Dataset Creation

We reproduce the [FLAN dataset](https://arxiv.org/pdf/2109.01652) based on the [code of the authors](https://github.com/google-research/FLAN/tree/main/flan), which was distributed under the Apache 2.0 License. Our implementation is based on [HuggingFace Datasets](https://huggingface.co/docs/hub/en/datasets). For each task in the original FLAN, we found the equivalent in HuggingFace Datasets and reimplemented the preprocesing in the original code to the best of our ability. However, due to the differing data sources, we note that the contents of our version may differ slightly. We designate all areas that were modified from the original.

The data loading is handled by `data_utils.py`. This loads all datasets mentioned in [Wei et al., 2022](https://arxiv.org/pdf/2109.01652), although we use only a subset of these for our experiments. Please see our paper for more details.

The preprocessing and prompt formatting is handled by `create_prompts.py`. This calls `data_utils.py` to load the data, preprocesses it, and finally formats it into one of two prompt types: *Regular Prompt* or *SampleGen Prompt*. 
  * *Regular Prompt* corresponds to the prompt formats of the original FLAN, which are included in `prompts/flan_orig.py`.
  * *SampleGen Prompt* corresponds to the prompt format that we develop for training our SampleGen models. 

The resulting dataset is saved to disk.

Since we instruction-tune our models with many prompt variations, we must create separate datasets or each variation. In all cases, the procedure is the same:

1. Generate individual task datasets
2. Create a training mixture from the individual tasks

### Step 1: Generate Individual Task Datasets

First, we create and save datasets for each of our tasks, as listed in `config.py`. To generate the datasets, call the `gen_all_tasks` function from the terminal and set the desired parameters. These parameters will specify the prompt format and the amount of data to include from each task. (Note: This does **not** correspond to the amount of data in the final training mixture. This will be specified later.)

#### Parameter Description
* `--num_data`: The number of samples desired in the training split of the dataset. The size of the validation split will be set accordingly. (This is relevant for creating SampleGen Prompt, since the procedure involves partitioning the training set into two parts.)
* `--create_custom_prompts`: Set to `True` for SampleGen Prompt, `False` for Regular Prompt.
* `--dataset_type`: Either `train+val` or `test`.
* `--making_adv`: Set to `True` if the in-context samples for SampleGen should come from a different task than the target.
* `--for_samplegen_pipeline`: Set to `True` if generating a gold pipeline dataset. This ensures that the samples will be formatted correctly.

For our experiments, the following dataset types must be created:

1. Regular Prompt
2. SampleGen Prompt
3. Gold Pipeline Test

#### 1: Regular Prompt
This is our baseline dataset, following the original FLAN dataset in prompt formatting. To ensure that the dataset matches the original FLAN as closely as possible, use all the tasks in `config.py`.

```py    
gen_all_tasks(num_data=10,
              create_custom_prompts=False,
              dataset_type="train+val",
              convert_letter_choices=True,
              )
```

The output will be a folder titled `flan_prompts`.


#### 2: SampleGen Prompt
To create our SampleGen Prompt, set the following arguments. Importantly, we set `create_custom_prompts=True`, which ensures that the samples will be 

```py    
gen_all_tasks(num_data=10,
              create_custom_prompts=True,
              dataset_type="train+val",
              convert_letter_choices=True,
              )
```
The output will be a folder titled `samplegen_prompts`.

#### 3: Gold Pipeline Test Set

Our Gold Pipeline experiments require the test set to be of a specific format. This must be generated and saved along with the other datasets. To ensure the correct formatting, both `create_custom_prompts` and `for_samplegen_pipeline` must be set to `True`.

```py    
gen_all_tasks(num_data=10,
              create_custom_prompts=True,
              dataset_type="train+val",
              for_samplegen_pipeline=True,
              convert_letter_choices=True,
              )
```
The output will be a folder titled `gold_pipeline_test`.


### Step 2: Training Mixture Generation
Once the individual task datasets are created, we combine these datasets into a *training mixture*. This is done by `make_mixtures.py`. 

```py
  mix_tasks(prompt_format="mix_prompts",
            do_mixed_gold_pipeline=True,
            setup_num="4"
            )
```

## Model Training
Once the dataset(s) have been created, call `train.sh` with the desired parameters, e.g.

```
--slurm_job_id $SLURM_JOB_ID --model_path_idx 1 --model_size_idx 1 --num_epochs 1 --prompts_type 0 --num_train 20000 --batch_size 2
```
The model will be saved to the `saved_models` folder and will have a unique name generated according to the timestamp at the start of training.

#### Important Parameters
* `--prompts_type`: The type of prompt to train on, as listed in `train.py`.

## Model Testing
For model testing, call one of the two testing scripts: `test_regularprompt.sh` or `test_samplegen+pipeline.sh`.

#### Important Parameters
* `--run_name`: The unique timestamp of the model in `saved_models`.


## Cite

Please use the following citation:

```
@InProceedings{smith:20xx:CONFERENCE_TITLE,
  author    = {Smith, John},
  title     = {My Paper Title},
  booktitle = {Proceedings of the 20XX Conference on XXXX},
  month     = mmm,
  year      = {20xx},
  address   = {Gotham City, USA},
  publisher = {Association for XXX},
  pages     = {XXXX--XXXX},
  url       = {http://xxxx.xxx}
}
```

## Disclaimer

> This repository contains experimental software and is published for the sole purpose of giving additional background details on the respective publication. 
