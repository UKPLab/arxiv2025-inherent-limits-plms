from config import *
from data_utils import *
from datasets import Dataset, load_from_disk
import json
from prompts.flan_orig import PATTERNS
import random
import re
from tqdm import tqdm

"""
Task preprocessing reimplemented based on: https://github.com/google-research/FLAN/blob/main/flan/tasks.py
Reference to the correct formats: https://huggingface.co/datasets/Muennighoff/flan/viewer/default/train?q=coqa&row=1589610
"""

random.seed(12345)

def remove_punct(string):
    string = string.rstrip('.').lower()
    string = string.replace("_", " ")
    string = string.capitalize()
    return string

def format_sample(template, 
                  template_targ_field, 
                  data_item, 
                  task_name, 
                  dataset,
                  main_sample=False, 
                  ic_example_num=None,
                  corrupt_samples=False,
                  do_unify=False,
                  convert_letter_choices=False
                  ):

    template_targ_field = template_targ_field.replace("{", "").replace("}", "")
    data_fields = [key for key in data_item.keys()]
    index_names = ["idx", "id", "ind", "concept_set_idx", "uid"]
    for name in index_names:
        if name in data_fields:
            data_fields.remove(name)

    
    options = ""
    options_list = ""
    args_dict = {}

    # For each task, format the options and label
    if "label" in dataset.features:
        label = dataset.features["label"]

    if task_name == "sentiment140":
        # Cast the labels to binary
        # Based on: https://github.com/google-research/FLAN/blob/main/flan/tasks.py#L983
        options_list = ["negative", "positive"]
        options = "; ".join(options_list)
        label = ("negative" if data_item["label"] in [0, 1, 2]
                 else "positive")
        data_item["options"] = options
        
    elif task_name == "rte":
        options_list = ["yes", "no"]
        options = "; ".join(options_list)
        label = options_list[int(data_item["label"])]

    elif task_name == "snli":
        options_list = ['yes', 'it is not possible to tell', 'no']
        options = "; ".join(options_list)
        label = options_list[int(data_item["label"])]
        args_dict["premise"] = data_item["premise"]
        args_dict["hypothesis"] = data_item["hypothesis"]
        data_item["options"] = options

    elif task_name == "squad_v2":
        label = data_item["label"]
        if label == []:
            label = 'unanswerable'
        else:
            label = label[0]

    elif task_name in ["anli_r1",
                       "anli_r2",
                       "anli_r3",
                       "mnli",
                       "mnli_matched",
                       "mnli_mismatched",
                       "wnli",
                       "qnli",
                       "copa",
                       "piqa",
                       "imdb_reviews",
                       "sst2",
                       "yelp_polarity_reviews",
                       "mrpc",
                       "qqp",
                       "paws_wiki",
                       "multirc",
                       "cosmos_qa",
                       "dpr_raw",
                       "wsc273",
                       "cola",
                    ]:
        if task_name in ["anli_r1", "anli_r2", "anli_r3"]:
            options_list = ['Yes', 'It\'s impossible to say', 'No']
            label = options_list[data_item["label"]]
            options = "; ".join(options_list)
            data_item["options"] = options
        elif task_name in ["mnli", "mnli_matched", "mnli_mismatched"]:
            options_list = ['yes', 'it is not possible to tell', 'no']
            label = options_list[data_item["label"]]
            options = "; ".join(options_list)
            data_item["options"] = options
            args_dict["hypothesis"] = data_item["hypothesis"]
            args_dict["premise"] = data_item["premise"]

        elif task_name in ["wnli", "mrpc", "qqp", "paws_wiki", "multirc"]:
            options_list = ["no", "yes"]
            label = options_list[data_item["label"]]
            options = "; ".join(options_list)
            data_item["options"] = options
        elif task_name == "qnli":
            options_list = ["yes", "no"]
            label = options_list[data_item["label"]]
            options = "; ".join(options_list)
            data_item["options"] = options
            data_item["sentence"] = '"{}"'.format(data_item["sentence"])
            data_item["question"] = '"{}"'.format(data_item["question"])
        elif task_name == "copa":
            options_list = [data_item["choice1"], data_item["choice2"]]
            label = options_list[data_item["label"]]
            options = "; ".join(options_list)
            data_item["options"] = options
            data_fields.remove("choice1")
            data_fields.remove("choice2")
        elif task_name == "piqa":
            options_list = [data_item["sol1"], data_item["sol2"]]
            label = options_list[data_item["label"]]
            options = "; ".join(options_list)
            data_item["options"] = options
            data_fields.remove("sol1")
            data_fields.remove("sol2")
        elif task_name in ["imdb_reviews", "sst2", "yelp_polarity_reviews"]:
            options_list = ['negative', 'positive']
            label = options_list[data_item["label"]]
            options = "; ".join(options_list)
            data_item["options"] = options
        elif task_name == "cosmos_qa":
            options_list = [data_item["answer0"].replace(" .", "."),
                   data_item["answer1"].replace(" .", "."),
                   data_item["answer2"].replace(" .", "."),
                   data_item["answer3"].replace(" .", ".")
                ]
            options = "; ".join(options_list)
            data_item["options"] = options
            data_item["question"] = '"{}"'.format(data_item["question"])
            label = options_list[data_item["label"]]
            data_fields.remove("answer0")
            data_fields.remove("answer1")
            data_fields.remove("answer2")
            data_fields.remove("answer3")

        elif task_name == "dpr_raw":
            pronoun = " " + data_item["pronoun"] + " "
            parts = data_item["sentence"].split(pronoun)
            suffix = " " + parts[-1].rstrip(".")
            options_list = [c + suffix for c in data_item["candidates"]]
            options = "; ".join(options_list)
            label = options_list[data_item["label"]]
            data_fields.remove("candidates")

        elif task_name == "wsc273":
            quote = data_item["quote"]
            possessives = ["his ", "her ", "its "]
            has_pos = False
            for pos in possessives:
                if pos in quote:
                    has_pos = True
            options_list = [c for c in data_item["options"]]
            pronoun = data_item["pronoun"]
            pattern_str = "(?<![A-Za-z]){}".format(pronoun)
            pattern = re.compile(pattern_str)
            new_options_list = []
            for opt in options_list:
                if has_pos:
                    opt = opt + "\'s"
                opt = re.sub(pattern=pattern, repl=opt, string=data_item["quote"])
                new_options_list.append(opt)
            options = "; ".join(new_options_list)
            label = options_list[data_item["label"]]
            data_fields = ["context", "options", "label"]
            
        elif task_name == "cola":
            options_list = ['unacceptable', 'acceptable']
            label = options_list[data_item["label"]]
            options = "; ".join(options_list)
            data_item["options"] = options

    elif task_name == "e2e_nlg":
        # Based on https://github.com/google-research/FLAN/blob/main/flan/tasks.py#L1596
        data = data_item["meaning_representation"]
        data = re.sub(r"\[", " = ", data)
        data = re.sub(r"\]", "", data)
        data_item["meaning_representation"] = data
        label = data_item["label"]

    elif task_name in ["bool_q"]:
        options_list = ["True", "False"]
        label = str(data_item["label"])
        options = "; ".join(options_list)
        data_item["options"] = options
    
    elif task_name == "wic":
        options_list = ["True", "False"]
        label = options_list[int(data_item["label"])]
        options = "; ".join(options_list)

    elif ("label" in dataset.features 
          and hasattr(dataset.features["label"], "names")):
        options_list = dataset.features["label"].names
        options = "; ".join(options_list)

    elif task_name in ["openbookqa", "arc", "arc_easy"]:
        options = data_item["options"]
        options_list = [item.strip() for item in data_item["options"].split(";")]
        if convert_letter_choices:
            # By default, the labels of these tasks are uppercase letter choices
            letters = [i for i in string.ascii_uppercase]
            numbers = ["1", "2", "3", "4"]
            if data_item["label"] in letters:
                label_idx = letters.index(data_item["label"])
            elif data_item["label"] in numbers:
                label_idx = numbers.index(data_item["label"])
            label = options_list[label_idx]
        else:
            label = data_item["label"]
        if task_name == "openbookqa":
            if '"' not in data_item["question_stem"]:
                data_item["question_stem"] = '"{}"'.format(data_item["question_stem"])

    elif task_name == "hellaswag":
        # Based on: https://github.com/google-research/FLAN/blob/main/flan/tasks.py#L2147
        if type(data_item["endings"]) == list:
            options_list = [opt.rstrip(".") for opt in data_item["endings"]]
            options = "; ".join(options_list)
            data_item["endings"] = "; ".join(data_item["endings"])
        else:
            options_list = data_item["endings"].split(";")
            options = "; ".join([opt.rstrip(".") for opt in options_list])
            data_item["endings"] = options
        args_dict["endings"] = options
        label = options_list[int(data_item["label"])]

    elif task_name in ["trivia_qa_full", "trivia_qa_wiki"]:
        # The target is normalized_value
        # Source: https://github.com/google-research/FLAN/blob/main/flan/tasks.py#L352
        label = data_item["label"]

    elif task_name == "stsb":
       options_list = ["0", "1", "2", "3", "4", "5"]
       options = "; ".join(options_list)
       label = int(float(data_item["label"]))
       data_item["options"] = options

    elif task_name == "record":
        query_right = data_item["query"].split("@placeholder")[1]
        options_list = [opt + query_right for opt in data_item["options"]]
        options = "; ".join(options_list)
        label = data_item["label"] + query_right
        args_dict["options_"] = options
    
    elif task_name == "winogrande":
        options_list = [data_item["option1"], data_item["option2"]]
        label = options_list[int(data_item["label"]) - 1]
        options = "; ".join(options_list)
        data_item["options"] = options
        data_fields = ["context", "options", "label"]

    elif task_name == "coqa":
        options = ""
        label = data_item["label"]

    elif task_name == "quac":
        # Pick the first pair from the questions and answers
        # Based on: https://github.com/google-research/FLAN/blob/main/flan/tasks.py#L836
        # and: https://www.tensorflow.org/datasets/catalog/quac
        if type(data_item["question"]) == list:
            data_item["question"] = data_item["question"][0]
        args_dict["question"] = data_item["question"]
        if type(data_item["label"]) == list:
            label = data_item["label"][0]
        else:
            label = data_item["label"]

    elif task_name == "trec":
        # Use the coarse labels
        # Based on: https://github.com/google-research/FLAN/blob/main/flan/tasks.py#L2024
        options_list = ['ABBR', 'ENTY', 'DESC', 'HUM', 'LOC', 'NUM']
        label = options_list[data_item["coarse_label"]]
        options = "; ".join(options_list)
        data_item["options"] = options
        data_fields = ["text", "options", "label"]
        data_item["text"] = '"{}"'.format(data_item["text"])

    elif task_name == "math_dataset":
        # The samples in the dataset are loaded by HF as strings despite the b''.
        # So we remove the b''.
        data_item["question"] = data_item["question"].replace("b'", "").replace("\\n'", "").replace("\n", "")
        label = data_item["label"].replace("b'", "").replace("\\n'", "").replace("\n", "")

    elif task_name == "common_gen":
        if type(data_item["concepts"]) == list:
            concepts_str = "; ".join(data_item["concepts"])
            data_item["concepts"] = concepts_str
        label = data_item["label"]
        
    elif task_name == "dart":
        # Based on: https://github.com/google-research/FLAN/blob/main/flan/tasks.py#L1559
        tripleset = ""
        for triple in data_item["tripleset"]:
            tripleset += " ; ".join(triple)
        tripleset = re.sub(r'\t', '', tripleset)
        tripleset = re.sub(r'\[(.*?)\]', '', tripleset)
        data_item["tripleset"] = tripleset
        label = data_item["label"]
    
    elif task_name == "web_nlg":
        # Based on: https://github.com/google-research/FLAN/blob/main/flan/tasks.py#L1629
        input_str = data_item["input_string"]
        if type(input_str) == list and len(input_str) > 1:
            input_str = ["; ".join(item_list) for item_list in input_str]
            input_str = "; ".join(input_str)
        elif type(input_str) == list and len(input_str) == 1:
            input_str = input_str[0]
            input_str = "; ".join(input_str)
        input_str = re.sub("_", " ", input_str)
        input_str = re.sub("\|", ",", input_str)
        data_item["input_string"] = input_str
        label = data_item["label"]
        args_dict["label"] = label

    elif "para_crawl" in task_name or "wmt16" in task_name:
        label = data_item[template_targ_field]
        if "1" in template_targ_field:
            data_fields.remove("sent1")
        elif "2" in template_targ_field:
            data_fields.remove("sent2")
    
    elif task_name in ["aeslc", 
                       "cnn_dailymail", 
                       "gigaword", 
                       "multi_news", 
                       "samsum",
                       "xsum"]:
        targ_field = template_targ_field
        label = data_item[targ_field]
        if task_name == "multi_news":
            data_item["summary"] = data_item["summary"].rstrip("– ")
        data_fields.remove(targ_field)

    
    elif task_name == "ag_news":
        # Source: https://github.com/google-research/FLAN/blob/main/flan/tasks.py#L902
        options_list = ["World", "Sports", "Business", "Sci/Tech"]
        options = "; ".join(options_list)
        # The indices of the data source are shifted by 1
        label = options_list[data_item["label"] - 1]

    elif task_name == "wiki_lingua/english":
        args_dict["source"] = data_item["article.document"][0]
        args_dict["target"] = data_item["article.summary"][0]
        label = args_dict[template_targ_field]
    
    elif task_name == "cb":
        options_list = ['yes', 'no', 'it\'s impossible to say']
        options = "; ".join(options_list)
        label = data_item["label"]
        data_item["options"] = options
    
    elif task_name == "story_cloze":
        options_list = [data_item["RandomFifthSentenceQuiz1"], data_item["RandomFifthSentenceQuiz2"]]
        options = "; ".join(options_list)
        label = options_list[int(data_item["label"]) - 1]       # The original labels are 1 and 2
        data_item["context"] = " ".join([data_item["InputSentence1"],
                                        data_item["InputSentence2"],
                                        data_item["InputSentence3"],
                                        data_item["InputSentence4"]
                                        ])
        data_item["options"] = options
        data_fields = ["context", "options", "label"]

    elif task_name in ["opin_idebate", "opin_movie", "newsroom"]:
        label = data_item[template_targ_field]
    
    elif task_name in config.bigbench_tasks:
        label = data_item["labels"]
        args_dict["inputs"] = data_item["text"]
        args_dict["options"] = "; ".join(data_item["multiple_choice_targets"])
        args_dict["labels"] = data_item["labels"]
        
    else:
        label = data_item["label"]

    if len(options) != 0:
        args_dict["options_"] = options

    if ic_example_num and not main_sample:
        if "para_crawl" not in task_name and "wmt16" not in task_name:
            args_dict["example_num"] = ic_example_num
            suffix = " " + str(ic_example_num)
        elif "para_crawl" in task_name or "wmt16" in task_name:
            args_dict["example_num"] = ic_example_num
            suffix = " " + str(ic_example_num)
    elif main_sample:
        args_dict["example_num"] = "Solution"
        suffix = " Solution"

    if "options" in data_fields:
        data_fields.remove("options")
        data_fields.append("options")

    if "{components}" in template:
        components = ""
        form = "\n{}{}: {}"
        count = 1
        for field in data_fields:
            if do_unify and field not in ["label", "answer"]:
                prefix = "Input " + str(count)
                count += 1
            else:
                prefix = field.capitalize()
            if field != "label":
                field_string = ""
                if field == "endings" and task_name == "hellaswag":
                    to_fill = args_dict["endings"]
                if field == "question" and task_name == "quac":
                    to_fill = data_item["question"]
                elif field == "options":
                    to_fill = args_dict["options_"]
                else:
                    # If args_dict has a modified version of a field, use the args_dict version
                    if field in args_dict and args_dict[field] != data_item[field]:
                        to_fill = args_dict[field]
                    # If the field value is the same in data_item and args_dict, add from data_item
                    elif field in args_dict and args_dict[field] == data_item[field]:
                        to_fill = data_item[field]
                    # If a data field isn't modified by args_dict, add it in as-is
                    elif field not in args_dict and data_item[field] not in args_dict.values():
                        to_fill = data_item[field]
                    else:
                        # This will only happen if a task has not been recognized
                        print("Error: task not recognized!")
                if corrupt_samples:
                    to_fill = "None"
                if task_name in config.bigbench_tasks:
                    to_fill = data_item[field]
                field_string = form.format(prefix, suffix, to_fill)
                components += field_string
        args_dict["components"] = components

    else:
        for field in data_fields:
            args_dict[field] = data_item[field]

    # Remove duplicates of the answer field
    if "{answer}" in template:
        args_dict["answer"] = label

    if corrupt_samples:
        for key in args_dict.keys():
            if main_sample and key not in ["example_num", 
                                           "components", 
                                           "answer", 
                                           template_targ_field
                                           ]:
                args_dict[key] = "\n(blank)"
            elif not main_sample and key not in ["example_num"]:
                args_dict[key] = "\n(blank)"

    # For NoAnswerSampleGen
    if config.no_answer_samplegen and main_sample:
        args_dict["answer"] = ""

    if do_unify:
        matches = re.findall(r'\{[A-Za-z_]+\}', template)
        for idx in range(len(matches)):
            if matches[idx] not in ["{options_}", "{answer}", "{components}", "{example_num}", '{example_num}:{components}', '{example_num}: {answer}']:
                new_name = "Input " + str(idx + 1)
                template = template.replace(matches[idx], "{" + new_name + "}")
                args_dict[new_name] = args_dict.pop(matches[idx].replace("}", "").replace("{", ""))

    formatted_sample = template.format(**args_dict)
    formatted_sample = formatted_sample.replace("\nExample Solution:", "")
    
    if do_unify:
        # Clean up double numbers
        formatted_sample = re.sub(r'(?<=Input [0-9]) [0-9]', '', formatted_sample)
    elif "para_crawl" in task_name or "wmt16" in task_name:
        # Clean up field titles for translation tasks due to their specific format
        formatted_sample = re.sub(r'(?<=Lang[12]) [12]', '', formatted_sample)
        formatted_sample = re.sub(r'(?<=Sent[12]) [12]', '', formatted_sample)
        formatted_sample = re.sub(r'(?<=Answer) [12]', '', formatted_sample)

    return formatted_sample, label, options_list


def create_mix_prompts(part_1, 
                       part_2, 
                       task1_name, 
                       num_datapoints, 
                       num_inner, 
                       test_size, 
                       making_test,
                       random_template=True,
                       making_adversarial=False,
                       task2_name=None,
                       task2_data=None,
                       corrupt_samples=False,
                       make_gold_pipeline_test=False,
                       do_unify=False,
                       convert_letter_choices=False
                       ):
    '''
    part_1 and part_2 are equal-size partitions of task1's train split.
    If making_test or making_adversarial, part_1 is the full task1 dataset.
    If making_adversarial, part_2 becomes task2_data.
    '''
    inputs, targets, gold_options = [], [], []

    # Open main templates file
    tmpl_path = config.path + "prompts/templates.json"
    with open(tmpl_path, "r") as main_tempfile:
        main_templates = json.load(main_tempfile)
        outer_template = main_templates[config.outer_template]
        inner_template = main_templates[config.inner_template]
        inner_template_main = main_templates[config.inner_template_main]
        test_template = main_templates[config.test_template]
        sys_prompt = main_templates[config.sys_prompt]

    if not making_test:
        num_samples = min(len(part_1), num_datapoints)
    else:
        # If making test set, load all samples
        num_samples = len(part_1)

    if making_adversarial:
        # This is when we insert IC samples from a different (irrelevant) task
        part_2 = task2_data
    else:
        task2_name = task1_name

    for idx in range(num_samples):
        # Pick 1 sample to serve as the main outer sample
        item_1 = part_1[idx]

        def _get_template(task_name):
            if task_name not in config.bigbench_tasks:
                if task_name in PATTERNS:
                    task_templates = PATTERNS[task_name]
                elif "para_crawl" in task_name:
                    task_templates = PATTERNS["para_crawl"]
                elif "wmt16" in task_name:
                    task_templates = PATTERNS["wmt16"]
                elif "wiki_lingua" in task_name:
                    task_templates = PATTERNS["wiki_lingua_en"]
                elif task_name == "arc_easy":
                    task_templates = PATTERNS["arc"]
                elif "anli" in task_name:
                    task_templates = PATTERNS["anli"]
                elif "mnli" in task_name:
                    task_templates = PATTERNS["mnli"]
                elif "trivia_qa" in task_name:
                    task_templates = PATTERNS["trivia_qa"] 
                first_five = task_templates[:5]
                if random_template:
                    random_idx = random.sample([0, 1, 2, 3, 4], 1)[0]
                    task_template = first_five[random_idx]
                elif not random_template:
                    task_template = first_five[0]
            elif task_name in config.bigbench_tasks:
                task_template = ("Question: {inputs}\nOptions: {options}\nAnswer Solution: {labels}", "{labels}")
            return task_template
        
        # Get template for task_1
        task1_template = _get_template(task1_name)
        question = task1_template[0]
        question = question.replace("\"", "")
        template1_targ_field = task1_template[1]

        # If task_2 provided, get template
        if task2_name != task1_name:
            task2_template = _get_template(task2_name)
            template2_targ_field = task2_template[1]
        else:
            task2_template = task1_template
            template2_targ_field = template1_targ_field

        # Construct the outer prompt
        main_sample, label, main_sample_options = format_sample(question, 
                                           template1_targ_field, 
                                           item_1, 
                                           task1_name, 
                                           part_1,
                                           main_sample=True,
                                           do_unify=do_unify,
                                           convert_letter_choices=convert_letter_choices
                                           )
        main_sample = main_sample.rstrip("\n")

        if not making_test:     
            main_sample = sys_prompt.format(formatted_main_sample=main_sample)
            main_sample_parsed, label, _ = format_sample(inner_template, 
                                                      template1_targ_field, 
                                                      item_1, 
                                                      task1_name, 
                                                      part_1, 
                                                      main_sample=True,
                                                      corrupt_samples=corrupt_samples,
                                                      do_unify=do_unify,
                                                      convert_letter_choices=convert_letter_choices
                                                      )

            # Randomly pick num_inner examples from part_2
            idx_list = range(len(part_2))
            sample_idxs = random.sample(idx_list, num_inner)
            ic_string = ""
            idx_count = 1
            for sample_idx in sample_idxs:
                task_sample = part_2[sample_idx]
                ic_sample, ic_label, _ = format_sample(inner_template, 
                                                    template2_targ_field, 
                                                    task_sample, 
                                                    task2_name, 
                                                    part_2, 
                                                    ic_example_num=idx_count,
                                                    corrupt_samples=corrupt_samples,
                                                    do_unify=do_unify,
                                                    convert_letter_choices=convert_letter_choices
                                                    )
                ic_string = ic_string + ic_sample
                idx_count += 1

            # Construct training samples according to the pattern
            sys_ending = main_templates["sys-prompt-ending"]
            target_prompt = outer_template.format(
                                                sys_and_main=main_sample, 
                                                ic_string=ic_string, 
                                                main_sample_parsed=main_sample_parsed, 
                                                ending=sys_ending
                                                )
            inputs.append(target_prompt)

        elif making_test and make_gold_pipeline_test:
            # If formatting a test sample for the samplegen pipeline, don't add IC samples
            # Simply format the test sample and return it to the calling code in test.py
            main_sample_parsed, label, main_sample_options = format_sample(
                                                    inner_template, 
                                                    template1_targ_field, 
                                                    item_1, 
                                                    task1_name, 
                                                    part_1, 
                                                    main_sample=True,
                                                    do_unify=do_unify,
                                                    convert_letter_choices=convert_letter_choices
                                                    )
            main_sample_parsed = re.sub(r'(?<=Answer Solution: ).*', '', main_sample_parsed, flags=re.DOTALL)
            inputs.append(main_sample_parsed)
        else:
            sys_ending = ""
            #options = ""
            main_sample = main_sample + "\n"# + " ".join(options)
            main_sample = sys_prompt.format(formatted_main_sample=main_sample)
            inputs.append(main_sample)

        targets.append(label)

        # For tasks without options, convert the empty string to a list containing the empty string.
        # This ensures that these tasks will be compatible during mixture creation.
        if main_sample_options == "":
            main_sample_options = [""]
        gold_options.append(main_sample_options)

    dataset = Dataset.from_dict({"text": inputs, 
                                 "labels": targets, 
                                 "gold_options": gold_options
                                 })

    # Test sets should not be split
    if not making_test:
        dataset = dataset.train_test_split(test_size=test_size)

    return dataset


def create_flan_format_prompts(dataset, 
                               task_name, 
                               num_datapoints,  
                               test_size, 
                               making_test,
                               random_template=True,
                               do_unify=False,
                               convert_letter_choices=False
                               ):
    """
    Create a dataset with the original FLAN instruction prompt format ('Regular Prompt').
    """
    # Open main templates file
    tmpl_path = config.path + "prompts/templates.json"
    with open(tmpl_path, "r") as main_tempfile:
        main_templates = json.load(main_tempfile)
        sys_prompt_flan = main_templates["sys-prompt-flan"]
        sys_prompt_flan_test = main_templates["sys-prompt-flan-test-new"]

    inputs, targets, gold_options = [], [], []

    if not making_test:
        num_samples = min(len(dataset), num_datapoints)
    else:
        # If making test set, load all samples
        num_samples = len(dataset)

    for idx in range(num_samples):
        item = dataset[idx]
        # Pick a task template
        if task_name in PATTERNS:
            task_templates = PATTERNS[task_name]
        elif "para_crawl" in task_name:
            task_templates = PATTERNS["para_crawl"]
        elif "wmt16" in task_name:
            task_templates = PATTERNS["wmt16"]
        elif "wiki_lingua" in task_name:
            task_templates = PATTERNS["wiki_lingua_en"]
        elif task_name == "arc_easy":
            task_templates = PATTERNS["arc"]
        elif "anli" in task_name:
            task_templates = PATTERNS["anli"]
        elif "mnli" in task_name:
            task_templates = PATTERNS["mnli"]
        elif "trivia_qa" in task_name:
            task_templates = PATTERNS["trivia_qa"]
        first_five = task_templates[:5]
        if random_template:
            random_idx = random.sample([0, 1, 2, 3, 4], 1)[0]
            task_template = first_five[random_idx]
        elif not random_template:
            task_template = first_five[0]
        
        # Construct the outer prompt
        question = task_template[0]
        question = question.replace("\"", "")
        template_targ_field = task_template[1]
        main_sample, label, options = format_sample(question, 
                                                    template_targ_field, 
                                                    item, 
                                                    task_name, 
                                                    dataset, 
                                                    do_unify=do_unify, 
                                                    convert_letter_choices=convert_letter_choices
                                                    )
        if not making_test:
            main_sample = main_sample.rstrip("\n")
            main_sample = sys_prompt_flan.format(formatted_main_sample=main_sample, target_field=label)
        else:
            main_sample = sys_prompt_flan_test.format(formatted_main_sample=main_sample)

        # For tasks without options, convert the empty string to a list containing the empty string.
        # This ensures that these tasks will be compatible during mixture creation.
        if options == "":
            options = [""]
        inputs.append(main_sample)
        targets.append(label)
        gold_options.append(options)

    dataset = Dataset.from_dict({"text": inputs, "gold_options": gold_options, "labels": targets})

    # Test sets should not be split
    if not making_test:
        dataset = dataset.train_test_split(test_size=test_size)
    return dataset


def create_and_save_dataset(dataset_name, 
                            num_datapoints, 
                            num_test, 
                            num_inner, 
                            save_path, 
                            making_test_set,
                            create_custom_prompts,
                            making_adv,
                            corrupt_samples,
                            make_gold_pipeline_test,
                            do_unify,
                            convert_letter_choices,
                            ablation_setup_name=None
                            ):
    if making_test_set:
        # The test sets for these tasks either have no labels, or have names other than "test"
        # Handle these according to the specific case
        if dataset_name in ["sst2",
                            "bool_q",
                            "winogrande",
                            "qnli", 
                            "coqa", 
                            "quac", 
                            "squad", 
                            "squad_v2", 
                            "record",
                            "hellaswag",
                            "stsb",
                            "trivia_qa_full",
                            "trivia_qa_wiki",
                            "common_gen",
                            "mnli_matched",
                            "mnli_mismatched"
                            ]:
            # The "test" split has no labels, so we load the val split
            split = "validation"
        elif dataset_name in ["fix_punct"]:
            # If no val/test split exists, load the "train" split (for test-only tasks)
            split = "train"
        else:
            # Otherwise, we can load the test split normally
            split = "test"     
    elif not making_test_set and dataset_name in config.bigbench_tasks:
        split = "validation"
    else:
        split = "train"
    
    if create_custom_prompts:
        if not making_test_set:
            do_partitioning=True
            part_1, part_2 = preprocess_dataset(dataset_name, 
                                                split=split, 
                                                do_partitioning=do_partitioning)
        else:
            do_partitioning=False
            part_2 = None
            part_1 = preprocess_dataset(dataset_name, 
                                                split=split, 
                                                do_partitioning=do_partitioning)
        if making_adv == "same_cat_as_target":
            # Determine task2_name depending on task1_cat
            for cluster in config.task_clusters:
                if dataset_name in config.task_clusters[cluster]:
                    task1_cluster = cluster

            for cat in config.cluster_categories:
                if task1_cluster in config.cluster_categories[cat]:
                    task1_cat = cat

            # We randomly choose task_2 from a different cluster, but same category as task_1
            task2_cluster_options = [c for c in config.cluster_categories[task1_cat] if c != task1_cluster]
            task2_cluster_idx = random.sample(list(range(len(task2_cluster_options))), 1)[0]
            task2_cluster = task2_cluster_options[task2_cluster_idx]
            task2_options = config.task_clusters[task2_cluster]
            task2_name_idx = random.sample(list(range(len(task2_options))), 1)[0]
            task2_name = task2_options[task2_name_idx]
            task2_data = preprocess_dataset(task2_name,
                                            split=split,
                                            do_partitioning=False
                                            )
        elif making_adv == "same_cat_as_training":
            # We take samples from the val set of the training mixture (folder is called "test")
            task2_name=ablation_setup_name
            task2_data = load_from_disk(config.path + "data/mix_prompts_mixture_" + ablation_setup_name)

        else:
            task2_name = None
            task2_data = None

        dataset = create_mix_prompts(part_1, 
                                     part_2, 
                                     task1_name=dataset_name, 
                                     num_datapoints=num_datapoints, 
                                     num_inner=num_inner, 
                                     test_size=num_test,
                                     making_test=making_test_set,
                                     making_adversarial=making_adv,
                                     task2_name=task2_name,
                                     task2_data=task2_data,
                                     corrupt_samples=corrupt_samples,
                                     make_gold_pipeline_test=make_gold_pipeline_test,
                                     do_unify=do_unify,
                                     convert_letter_choices=convert_letter_choices
                                     )
        save_path = save_path.format(prompt_format="mix_prompts", task=dataset_name)
    
    else:
        full_dataset = preprocess_dataset(dataset_name, 
                                                split=split, 
                                                do_partitioning=False)
        
        dataset = create_flan_format_prompts(full_dataset,
                                     task_name=dataset_name, 
                                     num_datapoints=num_datapoints, 
                                     test_size=num_test,
                                     making_test=making_test_set,
                                     do_unify=do_unify,
                                     convert_letter_choices=convert_letter_choices
                                     )
        save_path = save_path.format(prompt_format="flan_prompts", task=dataset_name)
    dataset.save_to_disk(save_path) 
    return dataset


def make_dataset(type, 
                 dataset_name, 
                 num_data,
                 create_custom_prompts,
                 making_adv,
                 corrupt_samples,
                 make_gold_pipeline_test,
                 do_unify,
                 convert_letter_choices,
                 ablation_setup_name=None
                ):
    '''
    Create a dataset either with the custom prompt style or the default flan style
    It will have two splits called "train" and "test" (dev)
    '''
    if type == "train+val":
        making_test_set = False
        save_path = config.train_data_save_path.format(prompt_format=("mix_prompts" if create_custom_prompts else "flan_prompts"),
                                                         task=dataset_name,
                                                         num_data=num_data
                                                        )
        sample_file = save_path + "dataset_samples_train.json"
    elif type == "test":
        making_test_set = True
        save_path = config.test_data_save_path.format(prompt_format=("mix_prompts" if create_custom_prompts else "flan_prompts"),
                                                         task=dataset_name,
                                                         num_data=""   #  The test set will take all the test data available
                                                        )
        sample_file = save_path + "dataset_samples_test.json"

    # num_data is the desired number of training samples the dataset will have
    # num_datapoints is the number of datapoints to generate across the train and val splits, 
    # given the desired value for num_data and the given test_percent
    test_percent = 0.2
    dataset = create_and_save_dataset(
            dataset_name=dataset_name, 
            num_datapoints=int(-(num_data // -(1 - test_percent))),  # -(a // -b) for ceiling division. Source: https://stackoverflow.com/questions/14822184/is-there-a-ceiling-equivalent-of-operator-in-python
            num_test=test_percent,
            num_inner=config.num_inner,
            save_path=save_path,
            making_test_set=making_test_set,
            create_custom_prompts=create_custom_prompts,
            making_adv=making_adv,
            corrupt_samples=corrupt_samples,
            make_gold_pipeline_test=make_gold_pipeline_test,
            do_unify=do_unify,
            convert_letter_choices=convert_letter_choices,
            ablation_setup_name=ablation_setup_name
            )
    
    if not making_test_set:
        sample_data1 = dataset["train"][0:2]
    else:
        sample_data1 = dataset[0:2]

    json_object = json.dumps(sample_data1, indent=4)
    with open(sample_file, "w") as outfile:
        outfile.write(json_object)
    
    return dataset


def gen_all_tasks(num_data, 
                  create_custom_prompts,
                  dataset_type,
                  making_adv=False,
                  corrupt_samples=False,
                  make_gold_pipeline_test=False,
                  do_unify=False,
                  convert_letter_choices=False,
                  ablation_setup_name=None
                  ):
    
    if dataset_type == "train+val":
        tasks = config.task_list
    elif dataset_type == "test":
        tasks = config.test_tasks

    print("Generating source datasets...")

    for idx in tqdm(range(len(tasks))):
        task = tasks[idx]
        print("Generating task:", task)
        dataset = make_dataset(type=dataset_type, 
                               dataset_name=task, 
                               num_data=num_data,
                               create_custom_prompts=create_custom_prompts,
                               making_adv=making_adv,
                               corrupt_samples=corrupt_samples,
                               make_gold_pipeline_test=make_gold_pipeline_test,
                               do_unify=do_unify,
                               convert_letter_choices=convert_letter_choices,
                               ablation_setup_name=ablation_setup_name
                            )
        

if __name__ == "__main__":
    gen_all_tasks(num_data=config.num_datapoints,
                  create_custom_prompts=config.create_custom_prompts,
                  dataset_type=config.dataset_type,
                  making_adv=config.make_adv_prompts,          
                  corrupt_samples=config.corrupt_samples,
                  make_gold_pipeline_test=config.make_gold_pipeline_test,
                  do_unify=config.do_unify,
                  convert_letter_choices=config.convert_letter_choices,
                  )
