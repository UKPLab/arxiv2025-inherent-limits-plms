import config
from evaluate import load
import math
import numpy as np
import os
import pandas as pd
from sklearn.metrics import accuracy_score
from tqdm import trange

'''
Computes the BERTScore given a model response file as input. 
The file is a .csv that has the following info: original model response, target, gold options
'''

def compute_bertscore(preds, targs, options, task_name):
    '''
    For a prediction, compute the BERTScore against each gold option. 
    Set the option with the highest BERTScore to be the model response.
    '''
    matched_opts = []
    correctness = []
    modified_targs = []
    targ_type_mod = False
    scorer = load("bertscore")
    for idx in trange(len(preds)):
        pred = preds[idx]
        actual = str(targs[idx])
        try:
            if math.isnan(pred):
                pred = ""
        except TypeError:
            pass
        if task_name in ["bool_q"]:
            pred = str(pred)
            actual = str(actual)
            actual = actual.replace("tensor(", "").rstrip(")")
            modified_targs.append(actual)
            targ_type_mod = True

        indiv_options = options[idx].split("###")
        pred = [pred] * len(indiv_options)
        score_dict = scorer.compute(predictions=pred, 
                                references=indiv_options,
                                model_type="roberta-large",
                                lang="en",
                                )
        pred_f1s = score_dict["f1"]
        matched_opt = indiv_options[np.argmax(pred_f1s)]
        matched_opts.append(matched_opt)
        if matched_opt == actual:
            correctness.append("T")
        else:
            correctness.append("F")
    if targ_type_mod:
        targs = modified_targs
    accuracy = accuracy_score(matched_opts, targs) * 100
    return accuracy, matched_opt

def parse_response_file(file):
    preds, refs, options = [], [], []
    response_data = pd.read_csv(file, sep="\t")
    preds = response_data["proc_model_response"]
    refs = response_data["target"]
    options = response_data["gold_options"]
    return preds, refs, options

def test_bertscore():
    scorer = load("bertscore")
    preds = ["Yes", "'Yes'", "\"Yes\""]
    refs = ["Yes", "Yes", "Yes"]
    score_dict = scorer.compute(predictions=preds, 
                                references=refs,
                                model_type="roberta-large",
                                lang="en",
                                )
    pred_f1s = score_dict["f1"]
    return
    
def get_bertscore(input_for_bertscore,
                  input_model,
                  use_exemplar_gen,
                  task_name
                  ):
    eval_log = pd.read_csv(config.path + "eval_logs.csv", sep=",")
    new_eval_info = {"job_id": "job-" + str(config.slurm_job_id),
                     "input_file": input_for_bertscore,
                     "input_model": input_model,
                     }
    for i in range(len(eval_log)):
        row = eval_log.loc[i]
        if row["eval_name"] == input_for_bertscore:
            new_eval_info["seed"] = row["seed"]
            new_eval_info["test_task"] = row["test_task"]
            new_eval_info["test_prompt_type"] = row["test_prompt_type"]
            new_eval_info["model_type"] = row["model_type"]
            new_eval_info["samplegen_sample_source"] = row["samplegen_sample_source"]
            new_eval_info["num_test"] = row["num_test"]
            new_eval_info["eval_mode"] = row["eval_mode"]

    if use_exemplar_gen:
        path_string = "{}/saved_models/{}/samplegen_pipeline_evals/{}.csv"
    else:
        if input_model != "base":
            path_string = "{}/saved_models/{}/evals/{}.csv"
        else:
            path_string = "{}/saved_models/{}_model_evals/{}.csv"

    path = path_string.format(config.path,
                              input_model,
                              input_for_bertscore
                              )
    preds, refs, options = parse_response_file(path)
    print("Computing BERTScore Accuracy...")
    accuracy, matched_opts = compute_bertscore(preds, refs, options, task_name)
    print("BERTSCORE ACCURACY:", accuracy)

    new_eval_info["bertscore_accuracy"] = accuracy
    if "bertscore_evals.csv" in os.listdir(config.path):
        header = False
    else:
        header = True
    new_eval_df = pd.DataFrame(new_eval_info, columns=new_eval_info.keys(), index=[0])
    new_eval_df.to_csv(config.path + "bertscore_evals.csv", sep=",", mode='a', header=header, index=False)

    print("Results written to csv.")
    return

if __name__ == "__main__":
    get_bertscore(input_for_bertscore=config.input_for_bertscore,
                  input_model = config.input_model,
                  use_exemplar_gen=config.use_exemplar_gen
                  )