import config
from datasets import Dataset, load_dataset, disable_caching, DownloadMode
import evaluate
import json
import pandas as pd
import random
import re
import string
import time

disable_caching()

CODE_TO_LANG = {"cs": "Czech",
                "en": "English",
                "es": "Spanish",
                "de": "German",
                "fi": "Finnish",
                "fr": "French",
                "ro": "Romanian",
                "ru": "Russian",
                "tr": "Turkish"}


def partition_dataset(dataset):
    shuffled_dataset = dataset.shuffle(seed=25)
    half_dataset = len(shuffled_dataset) // 2
    part_1 = shuffled_dataset.select(range(0, half_dataset)).flatten_indices()      # Flatten indices to speed up data processing
    part_2 = shuffled_dataset.select(range(half_dataset, len(shuffled_dataset))).flatten_indices()
    return part_1, part_2


def preprocess_flan_dataset(dataset, exemplars, spl=None, tokenizer=None):
    data_list = []
    if exemplars:
        data_path = config.path + "data/flan/{}/exemplars.txt".format(dataset)
    else:
        data_path = config.path + "data/flan/{}/no-exemplars.txt".format(dataset)
    with open(data_path, "r") as file:
        data = file.read()
        data_list.append(data)
        return data_list


def translate_test_data(text, translation_type):
    prompt_prefix = "Translate the following text to Scottish Gaelic:\n\n"
    if translation_type == "full":
        text = get_azure_response(prompt_prefix + text)
        # In case the translation model writes an explanation on a newline
        text = text.split("\n")[0]
    elif translation_type == "instr_only":
        pass
    return text


def create_bigbench_task_prompts(sample,
                                 template=None,
                                 template_name=None,
                                 sys_prompt=None,
                                 task_name=None,
                                 add_answer=False,
                                 translation_type=None
                                 ):
    """
    Create prompts for BigBench and GSM8K. These are only for testing.
    """
    def _normalize_item(item):
        item = item.rstrip(".")
        item = '"{}"'.format(item)
        return item

    # Prevent extra printing of the choices
    if task_name in ["social_iqa", "strange_stories"]:
        sample["text"] = re.sub(r'Q: ', '', sample["text"])
        sample["text"] = re.sub(r' choice:.*', '', sample["text"], flags=re.DOTALL)

    # Remove task-specific formatting artifacts
    if task_name in ["causal_judgment"]:
        sample["text"] = re.sub(r'A: ', '', sample["text"])
        sample["text"] = re.sub(r'Options: ', '', sample["text"])

    if task_name in ["social_iqa_translated"]:
        # Remove the options in the newline to isolate the text
        sample["text"] = sample["text"].split("\n")[0]

    # Optionally translate the dataset
    if translation_type == "full":
        template_name = template_name + "-translated"
        sample["text"] = translate_test_data(sample["text"], translation_type)
        time.sleep(5)
        # Save the index of the target in the options list
        raw_opts = sample["multiple_choice_targets"]
        target_loc = raw_opts.index(sample["labels"])
        # Translate the options
        translated_opts = []
        for opt in raw_opts:
            translated_opts.append(translate_test_data(opt, translation_type))
            time.sleep(5)
        sample["multiple_choice_targets"] = translated_opts
        # Pick the target from the translated options, rather than translating the target anew
        sample["labels"] = translated_opts[target_loc]
        time.sleep(5)
    elif translation_type == "instr_only":
        # We want to hard-code these translations into a separate prompt format, including the system prompt
        template = template + "-translated"
        pass
    if template_name == ["plain", "plain-translated"]:
        formatted_sample = template.format(question=sample["text"]
                                           )
    elif template_name in ["closed", "closed-translated"]:
        options_list = [_normalize_item(item) for item in sample["multiple_choice_targets"]]
        options = ", ".join(options_list)
        formatted_sample = template.format(question=sample["text"],
                                           options=options
                                           )
    elif "bad-prompt" in template_name:
        options_list = [_normalize_item(item) for item in sample["multiple_choice_targets"]]
        options = ", ".join(options_list)
        formatted_sample = template.format(question=sample["text"].rstrip(" ").rstrip("\n"),
                                           options=options,
                                           answer=sample["labels"]
                                           )
    elif template_name == ["closed-adv", "closed-adv-translated"]:
        options_list = [_normalize_item(item) for item in sample["multiple_choice_targets"]]
        adv_labels = ["{})".format(l) for l in string.ascii_uppercase]
        adv_labels = adv_labels[0:len(options_list)]
        numbered_options = ["{} {}".format(item[0], item[1]) for item in zip(adv_labels, options_list)]
        numbered_options = ", ".join(numbered_options)
        formatted_sample = template.format(question=sample["text"],
                                           numbered_options=numbered_options
                                        ) 
    elif template_name == "adv":
        options_list = [_normalize_item(item) for item in sample["multiple_choice_targets"]]
        adv_labels = ["({})".format(l) for l in string.ascii_lowercase]
        adv_labels = adv_labels[0:len(options_list)]
        numbered_options = ["{} {}".format(item[0], item[1]) for item in zip(adv_labels, options_list)]
        numbered_options = ", ".join(numbered_options)
        formatted_sample = template.format(question=sample["text"],
                                           numbered_options=numbered_options
                                        )
    elif "samplegen" in template_name:
        options_list = [_normalize_item(item) for item in sample["multiple_choice_targets"]]
        joined_options = ", ".join(options_list)
        target_idx = sample["multiple_choice_targets"].index(sample["labels"])
        formatted_sample = template.format(question=sample["text"],
                                           options=joined_options,
                                           answer=sample["labels"]
                                        )
    if sys_prompt is not None:
        formatted_sample = sys_prompt.format(formatted_main_sample=formatted_sample)
    sample["text"] = formatted_sample
    return sample


def format_bigbench_task_labels(sample):
    # Based on: https://stackoverflow.com/questions/16060899/alphabet-range-in-python
    #adv_labels = ["{})".format(l) for l in string.ascii_uppercase]
    adv_labels = ["({})".format(l) for l in string.ascii_lowercase]
    sample["labels"] = sample["labels"][0]
    target_idx = sample["multiple_choice_targets"].index(sample["labels"])
    sample["adv_labels"] = adv_labels[target_idx]
    return sample


def format_gsm8k(sample):
    sample["explanation"], sample["answer"] = sample["answer"].split("####")
    sample["explanation"] = sample["explanation"].rstrip(" ")
    sample["answer"] = sample["answer"].lstrip(" ")
    numbers = re.findall(r'[0-9]+', sample["explanation"])
    if sample["answer"] in numbers:
        numbers.remove(sample["answer"])
    random.shuffle(numbers)
    unique_numbers = []
    count = 0
    for n in numbers:
        if count < 3 and n not in unique_numbers:
                unique_numbers.append(n)
                count += 1
    unique_numbers.append(sample["answer"])  # Make sure the correct choice is included
    random.shuffle(unique_numbers)
    sample["multiple_choice_targets"] = unique_numbers
    return sample


def remove_list_labels(example):
    try:
        while type(example["label"]) == list:
            example["label"] = example["label"][0]
    except KeyError:
        pass
    except TypeError:
        pass
    except IndexError:
        # Label is not a list
        pass
    else:
        pass
    return example


def old_make_obqa_samples(example):
    letter_choices = ["A", "B", "C", "D"]
    letter_text_pairs = zip(letter_choices, example["options"])
    options_list = []
    for item in letter_text_pairs:
        choice = "{}) {}".format(item[0], item[1])
        options_list.append(choice)
        options = "; ".join(options_list)
    example["options"] = options
    return example

def make_obqa_samples(example):
    example["options"] = "; ".join(example["options"])
    return example


def make_paracrawl_wmt_components(example):
    langs = [key for key in example.keys()]
    sents = [val for val in example.values()]
    sent1, sent2 = sents[0], sents[1]
    lang1_code, lang2_code = langs[0], langs[1]
    lang1_code = re.sub(r'[a-z]+\.', '', lang1_code)
    lang2_code = re.sub(r'[a-z]+\.', '', lang2_code)
    lang1, lang2 = CODE_TO_LANG[lang1_code], CODE_TO_LANG[lang2_code]
    example["lang1"], example["lang2"] = lang1, lang2
    example["sent1"], example["sent2"] = sent1, sent2
    return example
    
def make_coqa_labels(example):
    numbered_questions, numbered_answers = [], []
    numbers = list(range(1, len(example["questions"]) + 1))

    answers = [a for a in example["answers.input_text"]]
    num_answ_pairs = zip(numbers, answers)
    for item in num_answ_pairs:
        ans = "{}. {}".format(item[0], item[1])
        numbered_answers.append(ans)  
    example["label"] = " ".join(numbered_answers)

    questions = [c for c in example["questions"]]
    num_quest_pairs = zip(numbers, questions)
    for item in num_quest_pairs:
        quest = "{}. {}".format(item[0], item[1])
        numbered_questions.append(quest)
    example["questions"] = " ".join(numbered_questions)
    return example

def make_quac_labels(example):
    example["label"] = example["label"][0][0]
    return example

def make_cb_labels(example):
    # Source: https://github.com/google-research/FLAN/blob/main/flan/tasks.py#L1783
    options = ['yes', 'no', 'it\'s impossible to say']
    example["label"] = options[example["label"]]
    return example

def make_story_cloze_labels(example):
    example["label"] = str(example["label"])
    return example

def fix_dropped_words(example):
    orig_text = example["utterance"]
    no_punct = example["corrupted_cased"]
    try:
        last_word = re.search(r'[\S]+.{1}$', orig_text).group()
    except AttributeError:
        # Ignore edge cases like words enclosed in parentheses
        last_word = ""
    last_word = re.sub(r"[\.,\?!]", "", last_word)
    no_punct = no_punct + last_word
    example["corrupted_cased"] = no_punct
    return example

def make_label_field(example):
    example["label"] = ""
    return example

def load_hf_dataset(dataset_name, split=None, do_partitioning=False):
    print("LOADING DATA SPLIT:", split)
    cutoff_num = 100000        # Set this to avoid fully loading massive datasets
    args_dict = {"split": split,
                 "trust_remote_code": True}

    # First determine the loading paths for HF datasets
    if dataset_name in ["snli", 
                        "piqa",
                        "sentiment140",
                        "cosmos_qa",
                        "quac",
                        "trec",
                        "dart",
                        "e2e_nlg",
                        "aeslc",
                        "gigaword",
                        "multi_news",
                        "samsum",
                        "dart",
                        ]:
        # These datasets can simply be loaded by name
        args_dict["path"] = dataset_name

    # Other datasets have their own specific loading path
    elif dataset_name == "xsum":
        args_dict["path"] = "knkarthick/xsum"
    
    elif dataset_name == "cnn_dailymail":
        args_dict["path"] = dataset_name
        args_dict["name"] = "3.0.0"   # FLAN uses 3.1.0, but this is unavailable on HF
    
    elif dataset_name == "web_nlg":
        args_dict["path"] = dataset_name
        args_dict["name"] = "release_v1"
        args_dict["split"] = "full"

    elif "math_dataset" in dataset_name:
        # Load the algebra subset
        # Source: https://github.com/google-research/FLAN/blob/main/flan/tasks.py#L421
        args_dict["path"] = "math_dataset"
        args_dict["name"] = "algebra__linear_1d"
    
    elif "winogrande" in dataset_name:
        args_dict["path"] = dataset_name
        args_dict["name"] = "winogrande_xl"

    elif dataset_name in ["mnli", "mnli_matched", "mnli_mismatched", "rte", "wnli", "qnli", "mrpc", "qqp", "stsb", "cola"]:
        # The dataset is a subset of GLUE
        args_dict["path"] = "glue"
        args_dict["name"] = dataset_name

        # NOTE: Mnli matched/mismatched are loaded for testing only. Train is always on full mnli train.
        # Test is done on validation set, because the test set has no labels.
        if dataset_name in ["mnli_matched", "mnli_mismatched"]:
            if split == "train":
                args_dict["split"] = "validation"

    elif dataset_name in ["copa", "multirc", "record", "wic", "cb"]:
        # The dataset is a subset of SuperGLUE
        args_dict["path"] = "super_glue"
        args_dict["name"] = dataset_name
     
    elif "para_crawl" in dataset_name:
        # The dataset is a subset of ParaCrawl. Only train split exists
        subset = re.sub(r".*/", "", dataset_name)
        args_dict["path"] = "para_crawl"
        args_dict["name"] = subset
        args_dict["split"] = "train"

    elif "wmt16" in dataset_name:
        # The dataset is a subset of WMT16
        langs = re.search(r'[a-z][a-z]-[a-z][a-z]', dataset_name).group()
        args_dict["path"] = "wmt/wmt16"
        args_dict["name"] = langs

    elif dataset_name == "wiki_lingua/english":
        # The dataset is a subset of WikiLingua
        args_dict["path"] = "wiki_lingua"
        args_dict["name"] = "english"

    elif dataset_name in ["dpr_raw", "wsc273"]:
        if "dpr_raw" in dataset_name:
            args_dict["path"] = "coref-data/dpr_raw"

        elif "wsc273" in dataset_name:
            # Only the test split is available
            args_dict["path"] = "coref-data/davis_wsc_raw"
            args_dict["name"] = "wsc273"
            args_dict["split"] = "test"  

    elif dataset_name == "paws_wiki":
        args_dict["path"] = "paws" 
        args_dict["name"] = "labeled_final"
        
    elif dataset_name == "anli_r1":
        args_dict["path"] = "anli"
        args_dict["split"] = split + "_r1"

    elif dataset_name == "anli_r2":
        args_dict["path"] = "anli"
        args_dict["split"] = split + "_r2"

    elif dataset_name == "anli_r3":
        args_dict["path"] = "anli"
        args_dict["split"] = split + "_r3"

    elif dataset_name == "hellaswag":
        args_dict["path"] = "Rowan/hellaswag"

    elif dataset_name in ["imdb_reviews", "sst2", "coqa"]:
        if dataset_name == "imdb_reviews":
            args_dict["path"]= "stanfordnlp/imdb"
        else:
            args_dict["path"]= "stanfordnlp/" + dataset_name

    elif dataset_name in ["arc", "arc_easy", "openbookqa", "common_gen"]:
        if dataset_name == "arc":
            args_dict["path"] = "allenai/ai2_" + dataset_name
            args_dict["name"] = "ARC-Challenge"
        elif dataset_name == "arc_easy":
            args_dict["path"] = "allenai/ai2_arc"
            args_dict["name"] = "ARC-Easy"
        else:
            args_dict["path"] = "allenai/" + dataset_name
        if dataset_name == "openbookqa":
            # This split contains the "fact" column
            args_dict["name"] = "additional"    

    elif dataset_name == "yelp_polarity_reviews":
        args_dict["path"] = "yelp_polarity"
    
    elif dataset_name == "natural_questions":
        args_dict["path"] = "rojagtap/natural_questions_clean"

    elif dataset_name == "trivia_qa_full":
        # Load the rc.nocontext split to avoid the unnecessary columns in rc
        # The samples themselves are the same
        args_dict["path"] = "mandarjoshi/trivia_qa"
        args_dict["name"] = "rc.nocontext"

    elif dataset_name == "trivia_qa_wiki":
        # Load the rc.nocontext split to avoid the unnecessary columns in rc
        # The samples themselves are the same
        args_dict["path"] = "mandarjoshi/trivia_qa"
        args_dict["name"] = "rc.wikipedia.nocontext"
    
    elif dataset_name == "bool_q":
        args_dict["path"] = "google/boolq"

    elif dataset_name == "drop":
        args_dict["path"] = "ucinlp/drop"
    
    elif dataset_name == "squad":
        args_dict["path"] = "rajpurkar/squad"

    elif dataset_name == "squad_v2":
        args_dict["path"] = "rajpurkar/squad_v2"
    
    elif dataset_name == "ag_news":
        args_dict["path"] = "sh0416/ag_news"

    elif dataset_name == "fix_punct":
        args_dict["path"] = "nbroad/fix_punctuation"
    
    elif dataset_name in config.bigbench_tasks:
        args_dict["path"] = "tasksource/bigbench"
        if "tracking_shuffled_objects" in dataset_name:
            args_dict["name"] = "tracking_shuffled_objects"
        else:
            args_dict["name"] = dataset_name

    elif dataset_name == "gsm8k":
        # Load main version, only for testing
        args_dict["path"] = dataset_name
        args_dict["name"] = "main"
        args_dict["split"] = "test"

    # Now that we have the paths, we proceed with the data loading
    # Some datasets must be loaded from local files
    if dataset_name in ["story_cloze", "newsroom", "opin_idebate", "opin_movie"]:
        load_path = config.path_manual
        if dataset_name == "story_cloze":
            data_df = pd.read_csv(load_path + "/" + dataset_name + "/cloze_test_spring2016.tsv", sep="\t")
            dataset = Dataset.from_pandas(data_df)
            dataset = dataset.rename_column("AnswerRightEnding", "label")

        elif dataset_name == "opin_idebate":
            def fix_spacing(line):
                line = line.replace(" ,", ",")
                line = line.replace(" .", ".")
                line = line.replace(" ;", ";")
                return line
            data = {"claim": [],
                    "argument_sentences": [],
                    "debate_name": []
                    }
            with open(load_path + "/opinion_abstracts/idebate.json") as file:
                raw_data = json.load(file)
                for line in raw_data:
                    data["claim"].append(line["_claim"])
                    data["argument_sentences"].append(" ".join([fix_spacing(item) for item in line["_argument_sentences"].values()]))
                    data["debate_name"].append(line["_debate_name"])
            data_df = pd.DataFrame.from_dict(data)
            dataset = Dataset.from_pandas(data_df)
            dataset = dataset.remove_columns([c for c in dataset.column_names if c not in data.keys()])

        elif dataset_name == "opin_movie":
            # Source: https://github.com/google-research/FLAN/blob/main/flan/tasks.py#L592
            def enumerate_lines(line_list):
                numbered_list = enumerate(line_list)
                joined_line = " ".join([str(item[0]) + ")" + item[1] for item in numbered_list])
                return joined_line
            data = {"movie": [],
                    "numbered_reviews": [],
                    "critic_consensus": [],
                    "first_review": []
                    }
            with open(load_path + "/opinion_abstracts/rottentomatoes.json") as file:
                raw_data = json.load(file)
                for line in raw_data:
                    data["movie"].append(line["_movie_name"])
                    data["numbered_reviews"].append(enumerate_lines(line["_critics"].values()))
                    data["critic_consensus"].append(line["_critic_consensus"])
                    data["first_review"].append(list(line["_critics"].values())[0])
            data_df = pd.DataFrame.from_dict(data)
            dataset = Dataset.from_pandas(data_df)
            dataset = dataset.remove_columns([c for c in dataset.column_names if c not in data.keys()])
        
        elif dataset_name == "newsroom":
            # Each line in the file is a JSON object.
            data = {"title": [],
                    "text": [],
                    "summary": []
                    }
            path = load_path + "/newsroom-release/release/{}.jsonl/{}-stats.jsonl".format(split, split)
            with open(path) as file:
                for line in file.readlines()[0:cutoff_num]:
                    raw_data = json.loads(line)
                    data["title"].append(raw_data["title"])
                    data["text"].append(raw_data["text"])
                    data["summary"].append(raw_data["summary"])
            data_df = pd.DataFrame.from_dict(data)
            dataset = Dataset.from_pandas(data_df)
            dataset = dataset.remove_columns([c for c in dataset.column_names if c not in data.keys()])
    
    # In all other cases, load from HF
    else:
        # To save memory, first stream cutoff_num samples, then convert to map-style
        # If len(dataset) < cutoff_num, the whole dataset will be loaded.
        if dataset_name in ["logical_deduction"]:
            args_dict["download_mode"] = DownloadMode.FORCE_REDOWNLOAD # Do not reuse cached data, since past modifications to the dataset will linger here
        if dataset_name in config.bigbench_tasks or dataset_name in config.other_tasks or dataset_name in ["cosmos_qa", "web_nlg"]:
            dataset = load_dataset(**args_dict)
        else:
            args_dict["streaming"] = True
            streamed_dataset = load_dataset(**args_dict)
            dataset_portion = pd.DataFrame(streamed_dataset.take(cutoff_num))
            dataset = Dataset.from_pandas(dataset_portion)

    # Some datasets require post-processing and filtering
    if dataset_name == "snli":
        # Remove samples without a consensus label. Source: https://huggingface.co/datasets/stanfordnlp/snli
        dataset = dataset.filter(lambda item: item["label"] != -1)  
        dataset = dataset.add_column("options", [""] * len(dataset))
    elif dataset_name == "fix_punct":
        dataset = dataset.map(fix_dropped_words)
        dataset = dataset.rename_column("utterance", "label")
        dataset = dataset.rename_column("corrupted_cased", "no_punct")
        dataset = dataset.remove_columns([c for c in dataset.column_names if c not in ["label", "no_punct"]])
    elif dataset_name == "ag_news":
        dataset = dataset.rename_column("description", "text")
        dataset = dataset.add_column("options", [""] * len(dataset))
    elif dataset_name in ["anli_r1", "anli_r2", "anli_r3"]:
        dataset = dataset.rename_column("premise", "context")
        dataset = dataset.remove_columns(["uid", "reason"])
        dataset = dataset.add_column("options", [""] * len(dataset))
    elif dataset_name in ["wnli", "qnli", "mnli", "mnli_matched", "mnli_mismatched", "cola"]:
        dataset = dataset.add_column("options", [""] * len(dataset))
    elif dataset_name == "rte":
        dataset = dataset.rename_column("sentence1", "premise")
        dataset = dataset.rename_column("sentence2", "hypothesis")
        dataset = dataset.add_column("options", [""] * len(dataset))
    elif dataset_name == "stsb":
        dataset = dataset.add_column("options", [""] * len(dataset))
    elif dataset_name == "hellaswag":
        dataset = dataset.rename_column("ctx", "context")
        dataset = dataset.remove_columns(["ctx_a", "ctx_b", "activity_label", "source_id", "split", "split_type"])
    elif dataset_name in ["trivia_qa_full", "trivia_qa_wiki"]:
        dataset = dataset.flatten()
        dataset = dataset.rename_column("answer.normalized_value", "label")
        dataset = dataset.remove_columns([c for c in dataset.column_names if c not in ["question", "label"]])
    elif dataset_name == "sentiment140":
        dataset = dataset.rename_column("sentiment", "label")
        dataset = dataset.remove_columns(["user", "query", "date"])
        dataset = dataset.add_column("options", [""] * len(dataset))
    elif dataset_name in ["arc", "arc_easy"]:
        dataset = dataset.flatten()
        dataset = dataset.rename_column("answerKey", "label")
        dataset = dataset.rename_column("choices.text", "options")
        dataset = dataset.remove_columns([c for c in dataset.features if c not in ["question", "label", "options"]])
        dataset = dataset.map(make_obqa_samples)
    elif dataset_name == "bool_q":
        dataset = dataset.rename_column("passage", "text")
        dataset = dataset.rename_column("answer", "label")
        dataset = dataset.add_column("options", [""] * len(dataset))
    elif dataset_name == "drop":
        dataset = dataset.flatten()
        dataset = dataset.rename_column("passage", "context")
        dataset = dataset.rename_column("answers_spans.spans", "label")
        dataset = dataset.remove_columns(["section_id", "query_id", "answers_spans.types"])
    elif dataset_name in ["cosmos_qa", "piqa"]:
        dataset = dataset.add_column("options", [""] * len(dataset))
    elif dataset_name == "multirc":
        dataset = dataset.rename_column("answer", "response")
        dataset = dataset.add_column("options", [""] * len(dataset))
    elif dataset_name in ["mrpc", "qqp"]:
        dataset = dataset.add_column("options", [""] * len(dataset))
    elif dataset_name == "wic":
        dataset = dataset.remove_columns(["start1", "start2", "end1", "end2"])
    elif dataset_name == "openbookqa":
        dataset = dataset.flatten()
        dataset = dataset.rename_column("fact1", "fact")
        dataset = dataset.rename_column("answerKey", "label")
        dataset = dataset.rename_column("choices.text", "options")
        dataset = dataset.remove_columns([c for c in dataset.features if c not in ["question_stem", "fact", "options", "label"]])
        dataset = dataset.map(make_obqa_samples)
    elif dataset_name in ["squad", "squad_v2"]:
        dataset = dataset.flatten()
        dataset = dataset.rename_column("answers.text", "label")
        dataset = dataset.remove_columns(["answers.answer_start"])
    elif dataset_name == "record":
        dataset = dataset.rename_column("answers", "label")
        dataset = dataset.rename_column("entities", "options")
        dataset = dataset.remove_columns(["entity_spans"])
    elif dataset_name == "winogrande":
        dataset = dataset.rename_column("sentence", "context")
        dataset = dataset.rename_column("answer", "label")
        dataset = dataset.add_column("options", [""] * len(dataset))
    elif dataset_name == "wsc273":
        dataset = dataset.rename_column("text", "context")
        dataset = dataset.remove_columns(["quote_loc", "source"])
    elif dataset_name == "coqa":
        dataset = dataset.flatten()
        dataset = dataset.map(make_coqa_labels)
        dataset = dataset.rename_column("story", "text")
        dataset = dataset.rename_column("questions", "numbered_questions")
        dataset = dataset.remove_columns(["answers.answer_start", "answers.answer_end", "answers.input_text", "source"])
    elif dataset_name == "quac":
        dataset = dataset.flatten()
        dataset = dataset.rename_column("answers.texts", "label")
        dataset = dataset.rename_column("questions", "question")
        dataset = dataset.remove_columns(["dialogue_id", 
                                          "wikipedia_page_title", 
                                          "turn_ids", 
                                          "followups", 
                                          "yesnos",
                                          "answers.answer_starts" ,
                                          "orig_answers.texts",
                                          "orig_answers.answer_starts"
                                          ])
    elif dataset_name in ["math_dataset", "natural_questions"]:
        dataset = dataset.rename_column("answer", "label")
    elif dataset_name == "common_gen":
        dataset = dataset.rename_column("target", "label")
    elif dataset_name == "dart":
        dataset = dataset.flatten()
        dataset = dataset.rename_column("annotations.text", "label")
        dataset = dataset.remove_columns(["subtree_was_extended", "annotations.source"])
    elif dataset_name == "e2e_nlg":
        dataset = dataset.rename_column("human_reference", "label")
    elif dataset_name == "web_nlg":
        dataset = dataset.flatten()
        dataset = dataset.rename_column("lex.text", "label")
        dataset = dataset.rename_column("original_triple_sets.otriple_set", "input_string")
        dataset = dataset.remove_columns([c for c in dataset.features if c not in ["input_string", "label"]])
    elif dataset_name == "wiki_lingua/english":
        # Remove samples with empty values
        dataset = dataset.flatten()
        dataset = dataset.filter(lambda item: len(item["article.summary"]) != 0)  
    elif dataset_name == "xsum":
        dataset = dataset.rename_column("dialogue", "text")
    elif dataset_name == "cb":
        dataset = dataset.map(make_cb_labels)
        dataset = dataset.add_column("options", [""] * len(dataset))
    elif dataset_name == "story_cloze":
        dataset = dataset.map(make_story_cloze_labels)
        dataset = dataset.remove_columns(["InputStoryid"])
        dataset = dataset.add_column("options", [""] * len(dataset))
        dataset = dataset.add_column("context", [""] * len(dataset))
    elif dataset_name == "trec":
        dataset = dataset.add_column("options", [""] * len(dataset))
    elif dataset_name in ["paws_wiki", "sst2", "yelp_polarity_reviews", "imdb_reviews"]:
        dataset = dataset.add_column("options", [""] * len(dataset))
    elif "para_crawl" in dataset_name or "wmt16" in dataset_name:
        dataset = dataset.flatten()
        dataset = dataset.add_column("lang1", [""] * len(dataset))
        dataset = dataset.add_column("lang2", [""] * len(dataset))
        dataset = dataset.add_column("sent1", [""] * len(dataset))
        dataset = dataset.add_column("sent2", [""] * len(dataset))
        dataset = dataset.map(make_paracrawl_wmt_components)
        dataset = dataset.remove_columns([c for c in dataset.features if "translation" in c])
    elif dataset_name == "common_gen":
        dataset = dataset.remove_columns(["concept_set_idx"])
    elif dataset_name in config.bigbench_tasks:
        dataset = dataset.rename_column("inputs", "text")
        dataset = dataset.rename_column("targets", "labels")
        dataset = dataset.map(format_bigbench_task_labels)
        if dataset_name == "tracking_shuffled_objects/three":
            dataset = dataset.filter(lambda item: len(item["multiple_choice_targets"]) == 3)
        elif dataset_name == "tracking_shuffled_objects/five":
            dataset = dataset.filter(lambda item: len(item["multiple_choice_targets"]) == 5)  
        elif dataset_name == "tracking_shuffled_objects/seven":
            dataset = dataset.filter(lambda item: len(item["multiple_choice_targets"]) == 7) 
    elif dataset_name == "gsm8k":
        dataset = dataset.add_column("multiple_choice_targets", [""] * len(dataset))
        dataset = dataset.map(format_gsm8k)
        dataset = dataset.rename_column("question", "text")
        dataset = dataset.rename_column("answer", "labels")

    # For NLG datasets with multiple possible targets in a list, pick the first one
    if dataset_name in ["web_nlg", 
                        "dart", 
                        "trivia_qa_full",
                        "trivia_qa_wiki", 
                        "drop", 
                        "squad", 
                        "record", 
                        "quac"
                        ]:
        dataset = dataset.map(remove_list_labels)

    # If dataset has no label column, create an empty one
    # This is necessary for interleave_datasets()
    elif dataset_name not in config.bigbench_tasks and dataset_name not in config.other_tasks:
        if "label" not in dataset.features:
            dataset = dataset.map(make_label_field)

    # If we are loading for custom prompt generation, partition the dataset
    if do_partitioning:
        part_1, part_2 = partition_dataset(dataset)
        return part_1, part_2
    elif not do_partitioning:
        return dataset

def preprocess_dataset(dataset_name, split=None, do_partitioning=True):
    return load_hf_dataset(dataset_name, 
                           split=split, 
                           do_partitioning=do_partitioning
                           )

    
def calc_metrics(model_output):
    preds = model_output[0]
    targets = model_output[1]
    if config.task_name == "snli":
        metric = evaluate.load("rouge")
    metric.add_batch(predictions=preds, references=targets)
    score = metric.compute()
    return score

if __name__ == "__main__":
    pass
