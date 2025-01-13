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

Our experiments can be reproduced in the following steps:

* Dataset Creation
  * Task Dataset Creation
  * Training Mixture Generation
* Model Training 


## Dataset Creation

We reproduce the [FLAN dataset](https://arxiv.org/pdf/2109.01652) based on the [code of the authors](https://github.com/google-research/FLAN/tree/main/flan), which was distributed under the Apache 2.0 License. Our implementation is based on [HuggingFace Datasets](https://huggingface.co/docs/hub/en/datasets). For each task in the original FLAN, we found the equivalent in HuggingFace Datasets and reimplemented the preprocesing in the original code to the best of our ability. However, due to the differing data sources, we note that the contents of our version may differ slightly. We designate all areas that were modified from the original.

The data loading is handled by `data_utils.py`. This loads all datasets mentioned in [Wei et al., 2022](https://arxiv.org/pdf/2109.01652), although we use only a subset of these for our experiments.

The preprocessing and prompt formatting is handled by `create_prompts.py`. This calls `data_utils.py` to load the data, preprocesses it, and finally formats it into one of two prompt types: *Regular Prompt* and *SampleGen Prompt*. 
  * *Regular Prompt* corresponds to the prompt formats of the original FLAN, which are included in `prompts/flan_orig.py`.
  * *SampleGen Prompt* corresponds to the prompt format we develop in the paper for training our SampleGen models. 

The resulting dataset is saved to disk.

### Creating and Saving a Dataset

Since we instruction-tune our models with many prompt variations, we must create individual training datasets for each variation. Once the datasets are 

many prompt variations and modifications, which are all saved as individual datasets `create_prompts.py` generates 
To create and save a dataset, call the `gen_all_tasks`function from the terminal.

#### Parameter description

* `Dataset Size, --num_data`: The number of samples desired in the training split of the dataset. The size of the validation split will be set accordingly. (This is relevant for creating SampleGen Prompt, since the procedure involves partitioning the training set into two parts.)
* `SampleGen Prompt, --create_custom_prompts`: Set to `True` for SampleGen Prompt, `False` for Regular Prompt.
* `Dataset Type, --dataset_type`: Either `train+val` or `test`. Note that both must usually be created.
* `Make Adversarial, --making_adv`: Set to `True` if the in-context samples for SampleGen should come from a different task than the target.
* `Gold Pipeline Samples, --for_samplegen_pipeline`: Set to `True` if generating a gold pipeline dataset. This ensures that the samples will be formatted correctly.

#### Example: Original FLAN Dataset
To generate the FLAN dataset as close to the original as possible, set the following arguments. Note that we use Regular Prompt here.

```py    
gen_all_tasks(num_data=10,
              create_custom_prompts=False,
              dataset_type="train+val",
              making_adv=False,
              for_samplegen_pipeline=False,
              convert_letter_choices=True,
              )
```

#### Example: The SampleGen Prompt dataset used in our experiments

```py    
gen_all_tasks(num_data=10,
              create_custom_prompts=True,
              dataset_type="train+val",
              making_adv=False,
              for_samplegen_pipeline=False,
              convert_letter_choices=True,
              )
```


```py
from .base import BaseClass # Notice how I omit the package name

BaseClass().something()
```

To import classes/methods from outside the package (e.g. when you want to use the package in some other project) you can instead refer to the package name:

```py
from arxiv2025_inherent_limits_plms import BaseClass # Notice how I omit the file name
from arxiv2025_inherent_limits_plms.subpackage import SubPackageClass # Here it's necessary because it's a subpackage

BaseClass().something()
SubPackageClass().something()
```

### Using scripts

This is how you can use `arxiv2025_inherent_limits_plms` from command line:

```bash
$ python -m arxiv2025_inherent_limits_plms
```

### Expected results

After running the experiments, you should expect the following results:

(Feel free to describe your expected results here...)

### Parameter description

* `x, --xxxx`: This parameter does something nice

* ...

* `z, --zzzz`: This parameter does something even nicer

## Development

Read the FAQs in [ABOUT_THIS_TEMPLATE.md](ABOUT_THIS_TEMPLATE.md) to learn more about how this template works and where you should put your classes & methods. Make sure you've correctly installed `requirements-dev.txt` dependencies

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
