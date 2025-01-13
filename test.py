import bertscore
import config
from create_prompts import *
from collections import defaultdict
import datasets
from functools import partial
from gen_response import get_response
from get_api_response import get_api_response
from model_init import *
import numpy as np
import os
import pandas as pd
from prompts.flan_orig import *
import random
import re
import string
import time
import torch
from torch.utils.data import Dataset, DataLoader
import glue_utils
import t5_metrics as metrics
import gm_metrics as gm_metrics


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
USE_VLLM = config.use_vllm
random.seed(config.seed)
np.random.seed(config.seed)

'''
Metrics used by FLAN:
Reference for task-specific metrics: https://github.com/google-research/FLAN/blob/main/flan/tasks.py
Glue/SuperGLUE metric names: https://github.com/google-research/text-to-text-transfer-transformer/blob/main/t5/data/glue_utils.py
Metric implementations: https://github.com/google-research/text-to-text-transfer-transformer/blob/main/t5/evaluation/metrics.py
'''

class LocalDataset(Dataset):
    def __init__(self, hf_data, adversarial=False, translated=False):
        self.hf_data = hf_data
        self.adversarial = adversarial
        self.translated = translated
        self.inputs = self.hf_data["text"]
        self.targets = self.hf_data["labels"]
        self.orig_options = self.get_orig_options()
        
    def get_orig_options(self):
        if "gold_options" in self.hf_data.column_names:
            return self.hf_data["gold_options"]
        elif "multiple_choice_targets" in self.hf_data.column_names:
            return self.hf_data["multiple_choice_targets"]
        else:
            return ""

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, index):
        if self.orig_options != "":
            return [self.inputs[index], self.targets[index], "###".join(self.orig_options[index])]
        else:
            return [self.inputs[index], self.targets[index], ""]


class SampleGenDataset(Dataset):
    def __init__(self, samplegen_data):
        self.samplegen_data = samplegen_data
        self.inputs = self.get_inputs()
        self.targets = self.samplegen_data["target"]
        self.orig_options = self.get_orig_options()

    def get_inputs(self):
        if self.samplegen_data["response"] is None:
            return ""
        else:
            return self.samplegen_data["response"]
        
    def get_orig_options(self):
        if "gold_options" in self.samplegen_data:
            if type(self.samplegen_data["gold_options"]) == list:
                if self.samplegen_data["gold_options"][0] is None:
                    return ""
                else:
                    return self.samplegen_data["gold_options"]
            else:
                return ""
        else:
            return ""

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, index):
        return [self.inputs[index], self.targets[index], self.orig_options]

def translate_test_data(text, translation_type):
    prompt_prefix = "Translate the following text to Scottish Gaelic:\n\n"
    if translation_type == "full":
        text = get_api_response(prompt_prefix + text)
    elif translation_type == "instr_only":
        pass
    print("TRANSLATED TEXT:", text)
    return text

def extract_inputs(sample):
    # Find first occurrence of phrase "Answer Solution:"
    # Slice at that pos + 16 chars in the phrase
    raw_inputs = sample["response"]
    raw_inputs = raw_inputs.lstrip("\n")
    stop_pos = raw_inputs.find("Answer Solution:")
    if stop_pos != -1:
        extracted_inputs = raw_inputs[0:stop_pos+16]
        sample["response"] = extracted_inputs
    return sample

def convert_tensor_targets(sample):
    sample["target"] = sample["target"].item()
    return sample

def remove_instructions(sample):
    sample["text"] = re.sub(r'<s>\[INST\].*\n<</SYS>>\n', '', sample["text"], flags=re.DOTALL)
    sample["text"] = re.sub(r'\n\[/INST\]', '', sample["text"])
    sample["text"] = sample["text"].replace("[/INST]", "")
    return sample

def format_ic_bigbench(sample, inner_template, task_name, task_dataset, ic_example_num):
    formatted_sample = format_sample(inner_template, 
                                     template_targ_field="labels", 
                                     data_item=sample,
                                     task_name=task_name,
                                     dataset=task_dataset,
                                     main_sample=False, 
                                     ic_example_num=ic_example_num,
                                     )
    sample["text"] = formatted_sample
    return sample

def filter_bad_samples(example):
    val = True
    if "Applied for full membership" in example["text"]:
        val = False
    elif "Louis van Gaal: W61 D22 L17" in example["text"]:
        val = False
    elif "Final: Saturday 20 August (18:00 BST)" in example["text"]:
        val = False
    elif "Communications Bill: James Brokenshire and Shami Chakrabarti" in example["text"]:
        val = False
    elif "Age: 22 Sport: Gymnastics" in example["text"]:
        val = False
    return val

def add_train_ic(sample, 
                 train_data=None, 
                 task_source=None, 
                 task_name=None, 
                 unified_prompts=False, 
                 translation_type=None, 
                 do_gp_ablation=False,
                 num_samples_ablation=None
                 ):
    # Randomly pick samples from the train data as a source of IC exemplars
    if task_source == "bigbench+gsm8k":
        # If it's a BB task, we need to pick num_inner samples.
        num_samples = config.num_inner
        sample_idcs = random.sample(list(range(len(train_data))), num_samples)
        exemplars = []
        count = 1
        for sample_idx in sample_idcs:
            train_sample = train_data[sample_idx]["text"]
            # Get rid of some artifacts present for some tasks
            train_sample = re.sub(r'Q: ', '', train_sample)
            train_sample = re.sub(r' choice:.*', '', train_sample, flags=re.DOTALL)
            train_sample = re.sub(r'Context: ', '', train_sample)
            train_sample = train_sample.replace("\n \n", "\n").replace("\n\n", "\n")
            train_sample = train_sample.replace("Answer Solution:", "Answer " + str(count) + ":")
            train_sample = train_sample.replace("Question:", "Question {}:".format(count))
            train_sample = train_sample.replace("Options:", "Options {}:".format(count))
            train_sample = "Example " + str(count) + ":\n" + train_sample
            exemplars.append(train_sample)
            count += 1
        exemplars = "\n\n".join(exemplars)
        if task_name in ["strange_stories"]:
            matches = re.findall(r'\n(\".+\")+', exemplars)
        else:
            matches = re.findall(r'(\".+\")+', exemplars)
        count = 1
        for match in matches:
            exemplars = exemplars.replace(match, "Options {}: ".format(count) + match)
            count += 1
        exemplars = exemplars.replace(" Options", "Options")

        # Format the main sample in the same way as the IC samples
        main_sample = sample["text"].replace("\n \n", "\n").replace("\n\n", "\n")
        main_sample = main_sample.replace("Question", "Question Solution")
        match = re.findall(r'(\".+\")+', main_sample)[0]
        main_sample = main_sample.replace(match, "Options Solution: " + match + "\nAnswer Solution:")
        sample["text"] = exemplars + "\n\nSolution:\n" + main_sample

        # Modify prefix names if doing unified prompts
        if unified_prompts:
            sample["text"] = sample["text"].replace("Question", "Input 1")
            sample["text"] = sample["text"].replace("Options", "Input 2")

        # Hard-code template translations
        if translation_type != "":
            sample["text"] = sample["text"].replace("Options Solution", "Fuasgladh Roghainnean")
            sample["text"] = sample["text"].replace("Options", "Roghainnean")
            sample["text"] = sample["text"].replace("Answer Solution", "Fuasgladh Freagairt")
            
    elif task_source == "flan":
        # If it's a FLAN task, a single sample will already have num_inner IC samples and we just need to extract them.
        if do_gp_ablation:
            num_samples = num_samples_ablation
        else:
            num_samples = 1
        sample_idcs = random.sample(range(len(train_data)), num_samples)   # pick without replacement
        exemplar = ""
        ic_sample_count = 2   # Start with 2
        for sample_idx in sample_idcs:
            sample_text = train_data[sample_idx]["text"]
            # Remove the system prompt
            sample_text = re.sub(r'<s>\[INST\].*\[/INST\]\n\n', '', sample_text, flags=re.DOTALL)
            # Pick out the IC samples
            match = re.findall(r'([A-Za-z0-9\s]+:.*\n)+\n', sample_text, flags=re.DOTALL)   # There might be a runtime bottleneck if some samples have too many matches; these samples must be removed.
            try:
                ic_samples = match[0].replace("END_EXAMPLES", "")
            except IndexError:
                print("ERROR: add_train_ic")
            if do_gp_ablation:
                one_ic_sample = ic_samples.split("Example 2:")[0]
                if exemplar != "":
                    one_ic_sample = one_ic_sample.replace("Example 1:", "Example " + str(ic_sample_count) + ":")
                    ic_sample_count += 1
                exemplar = exemplar + one_ic_sample
            else:
                exemplar = exemplar + ic_samples
        exemplar = exemplar.rstrip("\n\n") + "\n"                                             
        sample["text"] = exemplar + "\nSolution:\n" + sample["text"].lstrip("\n")
        sample["text"] = re.sub(r'<s>\[INST\].*\n<</SYS>>\n', '', sample["text"], flags=re.DOTALL)
        sample["text"] = re.sub(r'\n\[/INST\]', '', sample["text"])
        sample["text"] = sample["text"].rstrip(" ")
    return sample
    
def normalize(text):
    text = text.lower()
    text = text.replace('"', '')
    text = text.rstrip(".")
    return text
    
def simple_accuracy(predictions, targets):
    if predictions == targets:
        return {"simple acc": 1}
    else:
        return {"simple acc": 0}
    
def convert_from_letter(letter_resp, options):
    # Convert a letter response to the option of corresponding alphabetic index
    letter_resp = re.sub(r'[\*\.:`)(\n]*', '', letter_resp)
    letter_labels_upper = [l for l in string.ascii_uppercase]
    letter_labels_lower = [l for l in string.ascii_lowercase]
    try:
        model_ans_idx = letter_labels_upper.index(letter_resp)
        converted_answer = options[model_ans_idx]
    except:
        try:
            model_ans_idx = letter_labels_lower.index(letter_resp)
            converted_answer = options[model_ans_idx]
        except:
            converted_answer = None
    return converted_answer

def convert_from_number(number_resp, options):
    # Convert a number response to the option of corresponding numeric index
    number_resp = re.sub(r'[\*\.:`)(\n]*', '', number_resp)
    try:
        model_ans_idx = int(number_resp) - 1       # The model starts at index 1 rather than index 0
        converted_answer = options[model_ans_idx]
    except ValueError:
        converted_answer = None
    except IndexError:
        converted_answer = None
    return converted_answer


def get_letter_match(model_answer, indiv_options, model_type):
    letter_ans_match = None
    try:
        # For letter choices enclosed in brackets, possibly with a space preceding them
        letter_ans_match = re.search(pattern=r'[ ]+\([a-zA-Z]\)\n', string=model_answer).group()
        letter_ans_match = convert_from_letter(letter_ans_match, indiv_options)       
    except AttributeError:        
        try:
            if model_type == "samplegen":
                # For answers formatted as A, B, C. Only for samplegen models, otherwise it will overmatch:
                letter_ans_match = re.search(pattern=r'[A-Z][\.:`)]*', string=model_answer).group()
                letter_ans_match = letter_ans_match.rstrip(") ").lstrip("(")
                letter_ans_match = convert_from_letter(letter_ans_match, indiv_options)
            else:
                # For answers formatted as A), B), etc. on a line by themselves
                letter_ans_match = re.search(pattern=r'[A-Z][\.:`)]+', string=model_answer).group()
                letter_ans_match = letter_ans_match.rstrip(") ").lstrip("(")
                letter_ans_match = convert_from_letter(letter_ans_match, indiv_options)
        except AttributeError:
            # For answers formatted as A), B), etc. but with other text following
            try:
                letter_ans_match = re.search(pattern=r'[A-Z][\.:`)]+.*\n', string=model_answer).group()
                letter_ans_match = convert_from_letter(letter_ans_match, indiv_options)
            except AttributeError:
                pass
    if model_type == "base":
        try:
            letter_ans_match = re.search(pattern=r'[A-Z]', string=model_answer).group()
            letter_ans_match = convert_from_letter(letter_ans_match, indiv_options)
        except AttributeError:
            pass
    return letter_ans_match


def get_number_match(model_answer, indiv_options):
    num_ans_match = None
    try:
        num_ans_match = re.search(pattern=r'[0-9][\.:`)]*\n', string=model_answer).group()
        num_ans_match = convert_from_number(num_ans_match, indiv_options)
    except AttributeError:
        pass
    return num_ans_match


def get_verbatim_match(model_answer, indiv_options, target, do_targ_search=False):
    # Check if the model explicitly printed one of the gold options
    normalized_response = normalize(model_answer)
    verbatim_match = None
    if "" not in indiv_options:
        for option in indiv_options:
            normalized_opt = normalize(option)
            pattern = re.compile("^(\")*{}(\")*\n".format(option))
            match = re.search(pattern, normalized_response)
            if match is not None:
                verbatim_match = option
    # Search for the target in the response only for base models on tasks with no options
    elif "" in indiv_options and do_targ_search:
        normalized_targ = normalize(target)
        if normalized_targ in normalized_response:
            verbatim_match = target
    return verbatim_match


def format_main_sample_for_pipeline(sample, task_name, task_dataset):
    formatted_sample = format_sample(inner_template, 
                                     template_targ_field="labels", 
                                     data_item=sample,
                                     task_name=task_name,
                                     dataset=task_dataset,
                                     main_sample=True,
                                     )
    sample["text"] = formatted_sample
    return sample


def eval(model_responses, targets, prompt_type, task, batch_start, orig_options=None, doing_from_samplegen=False, sample_source=None):
    '''
    The model's answer is taken to be the text
    coming after the first "Answer Solution:".
    Targets and responses are processed as batches.
    '''
    targets = list(targets)
    proc_responses = []
    
    if config.args.run_name == "base":
        model_type = "base"
    else:
        model_type = "samplegen"

    for idx in range(len(model_responses)):
        response = model_responses[idx]
        model_answer = ""

        # Reformat tensor targets (tensors not compatible with eval metrics)
        if doing_from_samplegen and task in ["gsm8k", "bool_q"] and sample_source == "model":
            if type(targets[idx]) is not str:
                targets[idx] = targets[idx].item()

        # Get the options by splitting with the sep string
        # The orig_options format differs between the dataset classes
        if doing_from_samplegen:
            if sample_source == "model" and "" not in orig_options:
                indiv_options = orig_options[batch_start+idx]
                if len(indiv_options) > 0:
                    indiv_options = indiv_options[0].split("###")
            elif sample_source == "model" and "" in orig_options:
                indiv_options = ""
            elif sample_source == "train_data":
                indiv_options = orig_options[idx].split("###")
        elif not doing_from_samplegen:
            indiv_options = orig_options[idx].split("###")

        # Define where within the raw response we search for the model's answer (answer_form_match).
        # First, check if the model printed something like "Answer/Answer Solution". The answer is expected to be here.
        # We always expect an answer_form match in direct SampleGen testing, but it might also occur in base/base+pipeline.
        if config.args.run_name != "base":
            # We are testing a SampleGen model directly
            if prompt_type == "mix_prompts":
                # Take the first occurrence of "Answer Solution:"
                answer_form_match = re.search(pattern=r'Answer Solution:.*', string=response)
            elif prompt_type == "flan_prompts":
                # We're testing a Regular Prompt model
                answer_form_match = re.search(pattern=r'(.(?!</s>))*.(?=</s>)', string=response)
                if answer_form_match is None:
                    # Try a more restrictive answer form match search, because Mistral-FLAN is smarter than Llama-FLAN
                    answer_form_match = re.search(pattern=r'Answer Solution:.*', string=response)

        elif config.args.run_name == "base":
            # We are testing a base model through the SampleGen pipeline.
            # Search the first line for an occurrence of "Answer Solution", etc. If first line is empty, the response is empty.
            response = response.lstrip("\n")
            try:
                first_line = [i.lstrip().rstrip() for i in response.split('\n') if i.lstrip().rstrip() != ''][0]
            except IndexError:
                first_line = ""
            # Locate the substring containing the answer within the first line
            answer_form_match = re.search(pattern=r'[aA]nswer( Solution)*:.*', string=first_line)

        # If answer_form_match exists, we take the answer from there
        if answer_form_match is not None:
            answer_form_match = answer_form_match.group()
            answer_form_match = re.sub(r"[aA]nswer( Solution):[ \*]*", "", answer_form_match)
            answer_form_match = answer_form_match.replace("Answer: ", "").lstrip(" ")
            model_answer = answer_form_match
            if config.bb_test_prompt_format in ["closed-adv", "adv"]:
                letter_match = get_letter_match(answer_form_match, indiv_options, model_type)
                if letter_match is not None:
                    model_answer = letter_match
            elif config.bb_test_prompt_format in ["closed"] or "bad-prompt" in config.bb_test_prompt_format:
                if task in ["logical_deduction"]:
                # Try searching for a verbatim match of an option within the answer form match.
                # If not found, return the original answer form match.
                    verbatim_match = get_verbatim_match(answer_form_match, indiv_options, targets[idx])
                    if verbatim_match is not None:
                        model_answer = verbatim_match
                    else:
                        model_answer = answer_form_match.lstrip(" ").strip("\"")
            else:
                model_answer = answer_form_match.strip(" ").strip("\"")

        # If no answer_form_match exists, search the first line. This usually happens with base models.
        elif answer_form_match is None:
            if config.args.run_name == "base":
                # Try verbatim match on first response line only. Exclude GSM8K to prevent within-number matching, as well as generative tasks.
                # If no verbatim match found, preserve the whole first line.
                try:
                    first_line = [i.lstrip().rstrip() for i in response.split('\n') if i.lstrip().rstrip() != ''][0]
                    model_answer = first_line
                    if task not in ["gsm8k", "common_gen"] and "wmt" not in task:
                        verbatim_match = get_verbatim_match(first_line, indiv_options, targets[idx], do_targ_search=False)
                        if verbatim_match is not None:
                            model_answer = verbatim_match
                except IndexError:
                    # First line does not exist (e.g. blank response)
                    model_answer = ""
            elif config.args.run_name != "base":
            # If there is no answer_form_match with a SampleGen model, this is a hallucination
            # We do not want to be lenient here, so mark answer as incorrect
            # Otherwise, certain metrics might trivially match a generated IC exemplar, etc.
                model_answer = ""

        # Some tasks require format postprocessing
        if task in ["stsb"]:
            if model_answer != "":
                try:
                    model_answer = float(model_answer)
                except:
                    model_answer = 9.0
            targets[idx] = float(targets[idx])
        elif task in ["gsm8k"]:
            model_answer = model_answer.replace("$", "").replace("%", "").replace("\"", "")
            targets[idx] = str(targets[idx])
        
        proc_responses.append(model_answer)

    targets = tuple(targets)   # Convert back if list

    # Calculate task-specific metrics
    if task in [#"dart"
                ]:
        result = gm_metrics.rouge_fn(
                                targets=targets,
                                predictions=proc_responses,
                                )
    elif task in ["dart",
                "aeslc",
                "cnn_dailymail",
                "gigaword",
                "newsroom",
                "samsum",
                "xsum",
                "common_gen"
                ]:
        result = metrics.rouge(
                                targets=targets,
                                predictions=proc_responses,
                                )
    elif task in ["ag_news",
                  "trec", 
                  "math_dataset",
                  "story_cloze", 
                  "wsc273",
                  "yelp_polarity_reviews",
                  "gsm8k",
                  "winogrande",
                  "snli"
                  ] or task in config.bigbench_tasks:
        result = metrics.accuracy(
                                targets=targets,
                                predictions=proc_responses,
                                )
    # SuperGLUE Tasks
    elif task in ["bool_q",
                  "cb",
                  "copa",
                  "multirc",
                  "record",
                  "rte",
                  "wic"
                ]:
        metric_list = glue_utils.get_super_glue_metric(task)
        result = {}
        for metric in metric_list:
            metric_result = metric(
                        targets=targets,
                        predictions=proc_responses
                        )
            result.update(metric_result)
     # GLUE Tasks
    elif task in ["cola",
                  "sst2",
                  "mrpc",
                  "stsb",
                  "qqp",
                  "mnli",
                  "mnli_matched",
                  "mnli_mismatched",
                  "qnli",
                  "rte",
                  "wnli"
                ]:
        metric_list = glue_utils.get_glue_metric(task)
        result = {}
        for metric in metric_list:
            metric_result = metric(
                        targets=targets,
                        predictions=proc_responses
                        )
            result.update(metric_result)
    elif "wmt16" in task:
        result = metrics.bleu(
                                targets=targets,
                                predictions=proc_responses,
                                )
    elif task == "fix_punct":
        result = metrics.edit_distance(
                                targets=targets,
                                predictions=proc_responses,
                                )
    elif task in ["squad",
                  "squad_v2"]:
        result = metrics.squad(
                                targets=targets,
                                predictions=proc_responses,
                                )
    elif "trivia_qa" in task:
        result = metrics.trivia_qa(
                                targets=targets,
                                predictions=proc_responses,
                                )
    else:
        result = simple_accuracy(
                                predictions=proc_responses, 
                                targets=targets
                                )
    return result, proc_responses, indiv_options

def get_samplegen_eval_name():
    config.args.samplegen_eval_file
    return

def run_test(model_name,
             prompt_type, 
             from_samplegen, 
             task, 
             sample_source=None,
             samplegen_model=None,
             samplegen_eval_file=None,
             unified_prompts=False,
             translation_type=None,
             gp_ablation_setup=None
             ):
    eval_name = time.strftime('%Y%m%d_%H%M%S', time.localtime())
    print("TESTING TASK:", task)

    # Prepare output file for model responses
    responses = {"test_example": [],
                 "response": [],
                 "target": [],
                 "proc_model_response": [],
                 "gold_options": []
                 }

    # Load a trained or base model
    saved_model = "/storage/ukp/work/bigoulaeva/CoT_Recovery/src/saved_models/" + config.args.run_name
    if config.args.run_name == "base":
        saved_model = None
        if not from_samplegen:
            out_file = "/storage/ukp/work/bigoulaeva/CoT_Recovery/src/saved_models/base_model_evals/" + eval_name + ".csv"
        elif from_samplegen:
            target_folder = config.path + "saved_models/" + config.samplegen_model + "/"
            samplegen_eval_path = "samplegen_pipeline_evals"
            out_file = "{}{}/{}.csv".format(target_folder,
                                             samplegen_eval_path,
                                             eval_name
                                            )
            if samplegen_eval_path not in os.listdir(target_folder):
                os.makedirs(target_folder + samplegen_eval_path)    
    else:
        new_eval_path = saved_model + "/evals"
        if "evals" not in os.listdir(saved_model):
            os.makedirs(new_eval_path, exist_ok=True)
        out_file = new_eval_path + "/" + eval_name + ".csv"

    print("LOADING ADAPTER:", saved_model)
    model, tokenizer, _, lora_request = init_model(
                                       model_id=config.init_id, 
                                       do_train=False, 
                                       saved_adapter=saved_model, 
                                       use_vllm=USE_VLLM                  
                                       )
    # Load test data
    if not from_samplegen or (from_samplegen and sample_source == "train_data"):
        print("SAMPLE SOURCE:", sample_source)
        if task in config.test_tasks:
            adv = False
            if from_samplegen and sample_source == "train_data":
                # Load the gold-pipeline-formatted test data generated by create_prompts
                if gp_ablation_setup:
                    train_path = config.path + "data/ablation_{}_train/train".format(config.gp_ablation_setup)
                    gold_pipeline_load_path = config.path + "data/gold_pipeline_test/" + test_task + "/test_data"
                    do_gp_ablation = True
                elif unified_prompts:
                    train_path = config.path + "data/mix_prompts_train_unified/" + task + "/30000/train"
                    gold_pipeline_load_path = config.path + "data/gold_pipeline_test_unified/" + test_task + "/test_data"
                    do_gp_ablation = False
                else:
                    train_path = config.path + "data/mix_prompts_train/" + task + "/30000/train"
                    gold_pipeline_load_path = config.path + "data/gold_pipeline_test/" + test_task + "/test_data"
                    do_gp_ablation = False
            
                raw_test_data = datasets.load_from_disk(gold_pipeline_load_path)
                # Select num_samples here
                sample_amount = min(len(raw_test_data), config.num_test)
                indices = random.sample(range(len(raw_test_data)), k=sample_amount)
                raw_test_data = raw_test_data.select(indices)
                train_data = datasets.load_from_disk(train_path).shuffle(seed=config.seed)
                train_data_filtered = train_data.filter(filter_bad_samples)
                raw_test_data = raw_test_data.map(partial(add_train_ic, 
                                                          train_data=train_data_filtered,
                                                          task_source="flan",
                                                          do_gp_ablation=do_gp_ablation,
                                                          num_samples_ablation=config.num_samples_ablation,
                                                        ),
                                                        batched=False)
            elif not from_samplegen:
                # Load the regular test sets generated by create_prompts
                raw_test_data = datasets.load_from_disk(config.test_data_load_path.format(prompt_format=prompt_type))
                # Select num_samples here
                sample_amount = min(len(raw_test_data), config.num_test)
                indices = random.sample(range(len(raw_test_data)), k=sample_amount)
                raw_test_data = raw_test_data.select(indices)
            # If running a base model, remove the sys prompt + instructions
            if config.args.run_name == "base":
                raw_test_data = raw_test_data.map(remove_instructions)
        elif task in config.bigbench_tasks or task in config.other_tasks or (task in config.translated_tasks and not from_samplegen):
            # Load the original HF data and format prompts on-the-fly
            tmpl_path = config.path + "prompts/templates.json"
            with open(tmpl_path, "r") as main_tempfile:
                main_templates = json.load(main_tempfile)
                if config.args.run_name != "base":
                    if prompt_type == "mix_prompts":
                        sys_prompt = main_templates["sys-prompt"]
                    elif prompt_type == "flan_prompts":
                        sys_prompt = main_templates["sys-prompt-bigbench-flan"]
                    else:
                        sys_prompt = main_templates["sys-prompt"]
                    template = main_templates["bigbench-" + config.bb_test_prompt_format]
                elif config.args.run_name == "base":
                    sys_prompt = None
                    template = main_templates["bigbench-" + config.bb_test_prompt_format]

                if translation_type != "":
                    template = main_templates["bigbench-" + config.bb_test_prompt_format + "-translated"]
                    message = " & translate"
                elif task == "social_iqa_translated":
                    template = main_templates["bigbench-" + config.bb_test_prompt_format + "-translated"]
                    message = ""
                else:
                    message = ""
            # Load test data and select num_samples
            # These datasets are for test only, so we load the "train" split (not the "validation" split on HF)
            if task == "social_iqa_translated":
                load_path = config.path + "data/closed-translated/social_iqa/test_data/2500"
                raw_test_data = datasets.load_from_disk(load_path)
            else:
                raw_test_data = load_hf_dataset(task, split="train")
            sample_amount = min(len(raw_test_data), config.num_test)
            indices = random.sample(range(len(raw_test_data)), k=sample_amount)
            raw_test_data = raw_test_data.select(indices)
            raw_test_data = raw_test_data.map(partial(create_bigbench_task_prompts, 
                                                  template=template, 
                                                  template_name=config.bb_test_prompt_format,
                                                  sys_prompt=sys_prompt,
                                                  task_name=task,
                                                  translation_type=translation_type,
                                                  ), 
                                            desc="Format BigBench test data" + message,
                                            batched=False
                                            )
            translated_save_path_test = config.test_data_save_path.format(prompt_format=config.bb_test_prompt_format + "-translated",
                                                                 task="social_iqa",
                                                                 num_data=config.num_test
                                                                 )
            raw_test_data.save_to_disk(translated_save_path_test)

            if from_samplegen and sample_source == "train_data":
                # Load the "validation" split to be the source of the gold samples
                train_data = load_hf_dataset(task, split="validation").shuffle(seed=config.seed)
                # Select n samples, where n is the number of test samples * number of IC samples defined in config
                sample_amount = min(len(train_data), config.num_test*config.num_inner)
                indices = random.sample(range(len(train_data)), k=sample_amount)
                train_data = train_data.select(indices)
                # Get the template
                if translation_type == "":
                    template2 = main_templates["bigbench-" + config.bb_test_prompt_format + "-samplegen"]
                else:
                    template2 = main_templates["bigbench-" + config.bb_test_prompt_format + "-samplegen-translated"]
                # Apply the prompt format and optional translation to the train data
                train_data = train_data.map(partial(create_bigbench_task_prompts, 
                                                  template=template2, 
                                                  template_name="bigbench-" + config.bb_test_prompt_format + "-samplegen",
                                                  sys_prompt=sys_prompt,
                                                  task_name=task,
                                                  add_answer=True,
                                                  translation_type=translation_type,
                                                  ), 
                                            desc="Format gold IC data" + message,
                                            batched=False
                                            )
                # Add the formatted IC samples to the test data
                raw_test_data = raw_test_data.map(partial(add_train_ic, 
                                                  train_data=train_data,
                                                  task_source="bigbench+gsm8k",
                                                  task_name=task,
                                                  unified_prompts=unified_prompts,
                                                  translation_type=translation_type
                                                  ),
                                                batched=False,
                                                desc="Add formatted train IC to test data"
                                                )
            # Mark the dataset as adversarial if using an adv prompt format
            if config.bb_test_prompt_format in ["closed-adv", "adv"]:
                adv = True
            else:
                adv = False

            # If translating, save the translated dataset to file
            if translation_type != "":
                translated_save_path_with_ic = config.test_data_save_path.format(prompt_format=config.bb_test_prompt_format + "-translated+goldic",
                                                                 task="social_iqa",
                                                                 num_data=config.num_test
                                                                 )
                raw_test_data.save_to_disk(translated_save_path_with_ic)
        elif task in config.translated_tasks and from_samplegen:
            adv = False
            # The dataset is already formatted, simply load it
            if not from_samplegen:
                load_path = config.path + "data/closed-translated/social_iqa/test_data/2500"
            elif sample_source == "train_data":
                load_path = config.path + "data/closed-translated+goldic/social_iqa/test_data/2500"
            raw_test_data = datasets.load_from_disk(load_path)
            # Select num samples
            sample_amount = min(len(raw_test_data), config.num_test)
            indices = random.sample(range(len(raw_test_data)), k=sample_amount)
            raw_test_data = raw_test_data.select(indices)

        test_data = LocalDataset(raw_test_data, adversarial=adv)

    elif from_samplegen and sample_source == "model":
        # Test samples already correspond to the needed format; no need for further postprocessing (e.g. translation)
        # First search for samplegen eval file name; the job id is the same across the 3 runs in test_13B_full.sh
        with open(config.eval_log, "r") as eval_logfile:
            info = pd.read_csv(eval_logfile, sep=",")
            for i in range(len(info)):
                if info.loc[i]["job_id"] == "job-" + str(config.slurm_job_id):
                    if info.loc[i]["eval_mode"] == "direct_eval":
                        samplegen_eval_file = info.loc[i]["eval_name"]
        path = "{}saved_models/{}/evals/{}.csv".format(config.path, 
                                                        samplegen_model, 
                                                        samplegen_eval_file,
                                                        )

        # Read num_test samples from the file
        with open(path, "r") as samplegen_file:
            data = pd.read_csv(samplegen_file, sep="\t")
            data = datasets.Dataset.from_pandas(data).shuffle(seed=config.seed)
            data = data.map(extract_inputs)
            data = data[0:config.num_test]
        test_data = SampleGenDataset(data)
    
    test_dataloader = DataLoader(test_data, 
                                batch_size=config.batch_size,
                                shuffle=False
                                )

    # Begin inference
    num_samples = len(test_data)
    per_batch_scores = defaultdict(list)
    print("Begin testing model:", model)
    print("Num test samples:", num_samples)
    batch_start = 0
    for batch in test_dataloader:
        inputs, targets, orig_options = batch[0], batch[1], batch[2]

        if len(inputs) == 0 or inputs is None:
            inputs = ""

        raw_responses = get_response(model, 
                                tokenizer, 
                                inputs,
                                use_vllm=USE_VLLM, 
                                lora_request=lora_request
                                )
        if USE_VLLM:
            responses_list = [r.outputs[0].text for r in raw_responses]
        else:
            responses_list = raw_responses
            responses["response"].append(raw_responses)
        score_dict, proc_responses, gold_options = eval(responses_list, 
                          targets, 
                          prompt_type, 
                          task,
                          batch_start=batch_start,
                          orig_options=orig_options,
                          doing_from_samplegen=from_samplegen,
                          sample_source=sample_source
                          )
        for key, value in score_dict.items():
            per_batch_scores[key].append(value)
        for idx in range(len(inputs)):
            responses["test_example"].append(inputs[idx])
            responses["response"].append(responses_list[idx])
            responses["target"].append(targets[idx])
            responses["proc_model_response"].append(proc_responses[idx])
            if not from_samplegen:
                responses["gold_options"].append(orig_options[idx])
            elif from_samplegen:
                if config.sample_source == "model":
                    if orig_options[0] != "":   # Otherwise it breaks for tasks without options (batch_start+idx too high by 1).....
                        responses["gold_options"].append(orig_options[batch_start+idx][0])
                    else:
                        responses["gold_options"].append("")
                elif config.sample_source == "train_data":
                    responses["gold_options"].append(orig_options[idx])
        batch_start += config.batch_size

    # Get average score across all batches
    avg_scores = {key: np.mean(per_batch_scores[key]) for key in per_batch_scores}
    if config.bb_test_prompt_format is not None:
        test_prompt_type = config.bb_test_prompt_format
    else:
        test_prompt_type = prompt_type

    # Write raw responses to output file
    eval_info = {"job_id": "job-" + str(config.slurm_job_id),
                "seed": config.seed,
                "run_name": saved_model if saved_model else config.init_id,
                "test_task": config.test_task,
                "test_prompt_type": test_prompt_type,
                "model_type": config.test_prompt_format,
                "num_test": num_samples,
                "eval_mode": "samplegen_pipeline" if config.use_exemplar_gen else "direct_eval",
                "eval_name": eval_name,
                "samplegen_model_name": config.samplegen_model,
                "samplegen_file_name": samplegen_eval_file,
                "samplegen_sample_source": config.sample_source,
                "ablation_setup": config.gp_ablation_setup,
                "num_samples_ablation": config.num_samples_ablation
                }
    for score in avg_scores:
        eval_info[score] = avg_scores[score]
    eval_df = pd.DataFrame(eval_info, columns=eval_info.keys(), index=[0])
    if config.eval_log_name in os.listdir(config.path):
        header = False
    else:
        header = True
    eval_df.to_csv(config.eval_log, sep=",", mode='a', header=header, index=False)
    if from_samplegen:
        responses["samplegen_eval_file"] = [config.samplegen_eval_file] * num_samples
    response_df = pd.DataFrame.from_dict(responses)
    response_df.to_csv(out_file, sep="\t", mode='w', index=False)

    if task not in ["squad", 
                    "squad_v2", 
                    "dart", 
                    "wmt16/ro-en", 
                    "wmt16/de-en", 
                    "gigaword", 
                    "gsm8k",
                    "common_gen"
                    ]:
        # Compute BERTScore and write to separate file
        if not from_samplegen:
            input_model = config.args.run_name
        else:
            input_model = config.args.samplegen_model
        bertscore.get_bertscore(input_for_bertscore=eval_name,
                                input_model=input_model,
                                use_exemplar_gen=config.use_exemplar_gen,
                                task_name=task
                                )
    return eval_name


if __name__ == "__main__":
    run_test(model_name=config.args.run_name,
                         prompt_type=config.test_prompt_format, 
                         from_samplegen=config.use_exemplar_gen,
                         task=config.test_task, 
                         sample_source=config.sample_source,
                         samplegen_model=config.args.samplegen_model,
                         samplegen_eval_file=config.args.samplegen_eval_file,
                         unified_prompts=config.unified_prompts,
                         translation_type=config.translation_type,
                         gp_ablation_setup = config.gp_ablation_setup
                         )