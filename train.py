import config
from create_prompts import *
from datasets import load_from_disk
from data_utils import *
from model_init import *
import os
import pandas as pd
from prompts.flan_orig import *
import time
import torch
from transformers import TrainingArguments
from trl import SFTTrainer
import wandb

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
run_name = time.strftime('%Y%m%d_%H%M%S', time.localtime())
out_dir = config.path + "saved_models/" + run_name
wandb.login(key=os.getenv(config.wandb_key))

model, tokenizer, peft_config, _ = init_model(config.init_id, do_train=True)

dataset_name = ["mix_prompts_mixture",          #0   SampleGen Default
                "flan_prompts_mixture",         #1   Regular Prompt
                # From prompt engineering experiments
                "mix_prompts_mixture_corrupted",#2
                "mix_prompts_mixture_relevance",#3
                "mix_prompts_mixture_noanswer", #4
                "mix_prompts_mixture_unified",  #5
                "flan_prompts_mixture_unified", #6
                # Ablation setups start here
                "mix_prompts_mixture_setup1",   #7
                "flan_prompts_mixture_setup1",  #8
                "mix_prompts_mixture_setup2",   #9
                "flan_prompts_mixture_setup2",  #10
                "mix_prompts_mixture_setup3",   #11
                "flan_prompts_mixture_setup3",  #12
                "mix_prompts_mixture_setup4",   #13
                "flan_prompts_mixture_setup4",  #14
                "mix_prompts_mixture_setup5",   #15
                "flan_prompts_mixture_setup5",  #16
                "mix_prompts_mixture_setup6",   #17
                "flan_prompts_mixture_setup6",  #18
                "mix_prompts_mixture_setup7",   #19
                "flan_prompts_mixture_setup7",  #20
                ][config.prompts_type]

print("TRAINING ON:", dataset_name)

run = wandb.init(
    # Set the project where this run will be logged
    project="Inherent Limits PLMs",
    # Track hyperparameters and run metadata
    config={
        "learning_rate": 0.00005,
        "epochs": config.num_epochs,
    },
)

indices = [i for i in range(config.num_train)]
indices_val = [i for i in range(2)]

raw_train = load_from_disk(config.path + "data/" + dataset_name + "/train").select(indices).shuffle()
raw_val = load_from_disk(config.path + "data/" + dataset_name + "/test").select(indices_val).shuffle()


training_args = TrainingArguments(
        run_name=run_name,
        seed=config.seed,
        do_train=True,
        do_eval=True,
        use_cpu=False,
        evaluation_strategy="steps",
        eval_steps=0.1,
        disable_tqdm=False,
        num_train_epochs=config.num_epochs,
        prediction_loss_only=True,
        save_strategy="steps",
        save_steps=0.2,
        logging_steps=0.1,
        output_dir=out_dir,
        report_to="wandb",
        per_device_train_batch_size=config.batch_size,
        warmup_ratio = 0.03,
        group_by_length=True,
        gradient_checkpointing=True
)

run_info = {
        "job_id": "job-" + str(config.slurm_job_id),
        "seed": config.seed,
        "run_name": run_name,
        "model_name": config.init_id,
        "dataset": dataset_name,
        "outer_template": config.outer_template,
        "inner_template": config.inner_template,
        "num_inner": config.num_inner,
        "epochs": config.num_epochs,
        "num_train": config.num_train,
        "batch_size": config.batch_size,
        "device": torch.cuda.get_device_name(torch.cuda.current_device())
    }

# SFTTrainer does the tokenization for you
trainer = SFTTrainer(
        model=model,
        peft_config=peft_config,
        args=training_args,
        train_dataset=raw_train,
        eval_dataset=raw_val,
        dataset_text_field="text",
        dataset_batch_size=1,
        tokenizer=tokenizer,
        max_seq_length=4096,
        packing=False
)

trainer.train()

print("Finished training.")

model.save_pretrained(save_directory=out_dir)
tokenizer.save_pretrained(save_directory=out_dir)

run_df = pd.DataFrame(run_info, columns=run_info.keys(), index=[0])

if config.train_log_name in os.listdir(config.path):
    header = False
else:
    header = True
run_df.to_csv(config.train_log, sep=",", mode='a', header=header, index=False)

print("Saved model to " + out_dir)
print("Saved train info to " + config.train_log)
