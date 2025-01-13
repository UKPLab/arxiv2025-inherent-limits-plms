import argparse

def create_parser():
    parser = argparse.ArgumentParser(
                    description='Command line arguments for running the code on SLURM.',
                )
    
    # Common arguments
    parser.add_argument('--run_name', type=str, default='base')
    parser.add_argument('--slurm_job_id', type=int, default=None)
    parser.add_argument('--batch_size', type=int, default=1)

    # Args for creating datasets
    parser.add_argument('--create_custom_prompts', type=bool, default=False)
    parser.add_argument('--dataset_type', type=str, default="train+val", choices=["train+val", "test"])
    parser.add_argument('--num_datapoints', type=int, default=10)
    parser.add_argument('--num_inner', type=int, default=2)
    parser.add_argument('--no_answer_samplegen', type=str, default="False")
    parser.add_argument('--corrupt_samples', type=str, default="False")
    parser.add_argument('--make_adv_prompts', type=str, default="False")
    parser.add_argument('--for_samplegen_pipeline', type=str, default="False")
    parser.add_argument('--do_unify', type=str, default="False")
    parser.add_argument('--convert_letter_choices', type=bool, default=True)

    # Args for training models
    parser.add_argument('--model_path_idx', type=int, default=1)
    parser.add_argument('--model_size_idx', type=int, default=0)
    parser.add_argument('--train_task', type=int, default=0)
    parser.add_argument('--num_epochs', type=int, default=1)
    parser.add_argument('--num_train', type=int, default=None)
    parser.add_argument('--prompts_type', type=int, default=0)
    
    # Args for testing models
    parser.add_argument('--test_task', type=int, default=None)
    parser.add_argument('--num_test', type=int, default=None)
    parser.add_argument('--use_exemplar_gen', type=str, default="False")
    parser.add_argument('--samplegen_eval_file', type=str, default=None)
    parser.add_argument('--samplegen_model', type=str, default="None")
    parser.add_argument('--sample_source', type=str, choices=["model", "train_data"], default=None)
    parser.add_argument('--input_for_bertscore', type=str, default="None")
    parser.add_argument('--input_model', type=str, default="None")    
    parser.add_argument('--unified_prompts', type=str, default="False")
    parser.add_argument('--translation_type', type=str, choices=["full", "instr_only"], default="")
    parser.add_argument('--gp_ablation_setup', type=str, default=None, choices=["setup1",
                                                                               "setup2",
                                                                               "setup3",
                                                                               "setup4",
                                                                               "setup5",
                                                                               "setup6",
                                                                               "setup7",
                                                                               ])
    parser.add_argument('--num_samples_ablation', type=int, default=2)
    parser.add_argument('--use_vllm', type=str, default="False")
    parser.add_argument('--test_prompt_format', type=str, choices=["mix_prompts", "flan_prompts"])
    parser.add_argument('--bigbench_test_task', type=int, default=None)
    parser.add_argument('--misc_test_task', type=int, default=None)
    parser.add_argument('--trans_test_task', type=int, default=None)
    parser.add_argument('--bb_test_prompt_format', 
                        type=str, 
                        default=None, 
                        choices=[
                                 "closed",
                                 "closed-adv",
                                 "plain",
                                 "open",
                                 "bad-prompt-1",
                                 "bad-prompt-2",
                                 "bad-prompt-3",
                                 "bad-prompt-4",
                                 "bad-prompt-5",
                                 "bad-prompt-6",
                                ])
    return parser