from args import create_parser

parser = create_parser()
args = parser.parse_args()

# Set the current working path
PATH = "arxiv2025-inherent-limits-plms/"
path = PATH

# SLURM Job ID
slurm_job_id = args.slurm_job_id

# API Keys
openai_api_key = ""
wandb_key = ""
hf_key = ""
api_model = ""

# Random seed
seed = [42, 1234, 10203][0]

# Tasks and categories
bigbench_tasks = [## Emergent Tasks
                   "figure_of_speech_detection",
                   "logical_deduction",
                   "social_iqa",
                   "strange_stories",
                   ## Non-emergent tasks
                   "causal_judgment",
                   "tracking_shuffled_objects/three",
                   "tracking_shuffled_objects/five",
                   "tracking_shuffled_objects/seven",
                ]
other_tasks = ["gsm8k"]
translated_tasks = ["social_iqa_translated"]
task_list = ["anli_r1",
             "anli_r2",
             "anli_r3",
             "rte",
             "snli",
             "cb",
             "mnli",
             "wnli",
             "qnli",
             "copa",
             "hellaswag",
             "piqa",
             "story_cloze",
             "imdb_reviews",
             "sentiment140",
             "sst2",
             "yelp_polarity_reviews",
             "mrpc",
             "qqp",
             "paws_wiki",
             "stsb",
             "arc",
             "arc_easy",
             ##"natural_questions",      # Drop: length
             "trivia_qa_full",
             "trivia_qa_wiki",
             "bool_q",
             "drop",
             "multirc",
             "openbookqa",
             "squad",
             "squad_v2",
             "cosmos_qa",
             "record",
             "dpr_raw",
             "winogrande",
             "wsc273",
             "coqa",
             "quac",
             "trec",
             "cola",
             "wic",
             "math_dataset",
             "fix_punct",
             "common_gen",
             "dart",
             "e2e_nlg",
             "web_nlg",
             "para_crawl/ende",
             "para_crawl/enes",
             "para_crawl/enfr",
             "wmt16/cs-en",
             "wmt16/de-en",
             "wmt16/fi-en",
             "wmt16/ro-en",
             "wmt16/ru-en",
             "wmt16/tr-en",
             "aeslc",
             "ag_news",
             "cnn_dailymail",
             "gigaword",
             ##"multi_news",           # Drop: length
             "newsroom",
             "samsum",
             ##"wiki_lingua/english",  # Drop: length
             "xsum",
             "opin_idebate",
             "opin_movie"
             ]
test_tasks = ["qnli",
             "story_cloze",
             "yelp_polarity_reviews",
             "stsb",
             "trivia_qa_full",
             "trivia_qa_wiki",
             "squad",
             "squad_v2",
             "record",
             "wsc273",
             "dart",
             "wmt16/ro-en",
             # misc tasks,
             "coqa",
             "quac",
             "trec",
             "cola",
             "wic",
             "math_dataset",
             "fix_punct",
             # summariation tasks,
             "aeslc",
             "ag_news",
             "cnn_dailymail",
             "gigaword",
             "newsroom",
             "samsum",
             "xsum",
             "opin_idebate",
             "opin_movie",
             "bool_q",
             "winogrande",
             "snli",
             "copa",
             "common_gen",
             "wmt16/de-en",
             "mnli_matched",        
             "mnli_mismatched",     
             "dpr_raw"              
            ]
task_clusters = {
                 "nli": ["anli_r1",
                         "anli_r2",
                         "anli_r3",
                         "rte",
                         "cb",
                         "snli",
                         "mnli",
                         "wnli",
                         "qnli"
                         ],
                 "commonsense": ["copa",
                                 "hellaswag",
                                 "piqa",
                                 "story_cloze"
                                 ],
                 "sentiment": ["imdb_reviews",
                               "sentiment140",
                               "sst2",
                               "yelp_polarity_reviews"
                               ],
                 "paraphrase": ["mrpc",
                                "qqp",
                                "paws_wiki",
                                "stsb"
                                ],
                 "closed_book_qa": ["arc",
                                    "arc_easy",
                                    #"natural_questions",   # Drop: length
                                    "trivia_qa_full",
                                    "trivia_qa_wiki"
                                    ],
                 "reading_comp": ["bool_q",
                                  "drop",
                                  "multirc",
                                  "openbookqa",
                                  "squad",
                                  "squad_v2",
                                  ],
                 "reading_comp_with_cs": ["cosmos_qa",
                                          "record"
                                          ],
                 "coreference": ["dpr_raw",
                                 "winogrande",
                                 "wsc273"
                                 ],
                 "misc": ["coqa",
                          "quac",
                          "trec",
                          "cola",
                          "wic",
                          "math_dataset",
                          "fix_punct"
                          ],
                 "struct_to_text_nlg": ["common_gen",
                                        "dart",
                                        "e2e_nlg",
                                        "web_nlg"
                                        ],
                 "translation_nlg": ["para_crawl/ende",
                                     "para_crawl/enes",
                                     "para_crawl/enfr",
                                     "wmt16/cs-en",
                                     "wmt16/de-en",
                                     "wmt16/fi-en",
                                     "wmt16/ro-en",
                                     "wmt16/ru-en",
                                     "wmt16/tr-en"
                                     ],
                 "summarization_nlg": ["aeslc",
                                       "ag_news",
                                       "cnn_dailymail",
                                       "gigaword",
                                       #"multi_news",    # Drop: Length
                                       #"newsroom",      # Drop: Length
                                       "samsum",
                                       #"wiki_lingua/english",     # Drop: Length
                                       "xsum",
                                       "opin_idebate",
                                       "opin_movie"
                                       ]
                }
cluster_categories = {
                      "nlu": [
                              "nli",
                              "commonsense",
                              "sentiment",
                              "paraphrase",
                              "closed_book_qa",
                              "reading_comp",
                              "reading_comp_with_cs",
                              "coreference",
                              "misc"
                             ],
                      "nlg": [
                              "struct_to_text_nlg",
                              "translation_nlg",
                              "summarization_nlg" 
                             ]
                }
train_task = task_list[args.train_task]
bb_test_prompt_format = args.bb_test_prompt_format

if args.bigbench_test_task != None:
    test_task = bigbench_tasks[args.bigbench_test_task]
elif args.misc_test_task != None:
    test_task = other_tasks[args.misc_test_task]
elif args.trans_test_task != None:
    test_task = translated_tasks[args.trans_test_task]
elif args.test_task != None:
    test_task = test_tasks[args.test_task]
else:    
    test_task = ""

# Arguments for generating datasets with create_prompts
create_custom_prompts = args.create_custom_prompts
num_datapoints = args.num_datapoints
dataset_type = args.dataset_type
num_inner = args.num_inner
convert_letter_choices = args.convert_letter_choices
make_gold_pipeline_test = args.make_gold_pipeline_test
do_unify = args.do_unify

# Paths for saving created datasets
train_data_save_path = path + "data/{prompt_format}/{task}/{num_data}"
test_data_save_path = path + "data/{prompt_format}/{task}/test_data/{num_data}"

# Paths for loading test data
test_data_load_path = path + "data/{prompt_format}_test/" + test_task + "/test_data"

# Paths to manually-downloaded datasets
path_manual = path + "data/manual_downloads"

# Training params
use_vllm = True if args.use_vllm == "True" else False
num_train = args.num_train
prompts_type = args.prompts_type
batch_size = args.batch_size
num_epochs = args.num_epochs

# Testing params
num_test = args.num_test
test_prompt_format = args.test_prompt_format
samplegen_eval_file = args.samplegen_eval_file
samplegen_model = args.samplegen_model
sample_source = args.sample_source
input_for_bertscore = args.input_for_bertscore
input_model = args.input_model
gp_ablation_setup = args.gp_ablation_setup
num_samples_ablation = args.num_samples_ablation
temperature = 0.001
top_k = 1
max_new_tokens = 200

# Params to set when creating new datasets
make_adv_prompts = True if args.make_adv_prompts == "True" else False 
corrupt_samples = True if args.corrupt_samples == "True" else False
use_exemplar_gen = True if args.use_exemplar_gen == "True" else False
no_answer_samplegen = True if args.no_answer_samplegen == "True" else False
unified_prompts = True if args.unified_prompts == "True" else False
translation_type = args.translation_type

# Configure the inner and outer templates for SampleGen prompts
# Templates are found in prompts/templates.json
outer_template = "outer-template-4"
inner_template = "gen-inner-template-num2"
inner_template_main = "main-inner-template"
test_template = "test-template"
sys_prompt = "sys-prompt"

# Model path and name params
model_path = ["<PATH_TO_LOCAL_MODEL>"][args.model_path_idx]
model_size = ["7B", "13B", "70B", "13B-Chat", "7B-Chat"][args.model_size_idx]
if "llama-2-hf" in model_path:
    init_id = model_path + model_size
elif "mistralai" in model_path:
    init_id = "mistralai/" + "Mistral-7B-v0.1"

# Info to log during saving
train_log_name = "train_logs.csv"
train_log = path + train_log_name
eval_log_name = "eval_logs.csv"
eval_log = path + eval_log_name
