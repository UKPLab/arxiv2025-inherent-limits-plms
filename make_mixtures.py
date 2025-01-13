"""
Our implementation of dataset mixing, based on:
https://github.com/google-research/FLAN/blob/main/flan/v2/mixtures.py
https://github.com/google-research/FLAN/blob/main/flan/task_splits.py

To modify interleave_datasets (through arrow module): https://github.com/huggingface/datasets/blob/0f1f27c69f6cc8d085b66a8a2ba0440a39bc5bce/src/datasets/arrow_dataset.py#L6260
"""
from datasets import interleave_datasets, load_from_disk
from data_utils import *
from tqdm import tqdm


def create_mixture(dataset_names: list,
                     prompt_format, 
                     maximum=3000,
                     size_limit=30000,
                     temperature=1.0,
                     scale=1.0,
                     do_mixed_gold_pipeline=False,

                     ):
    '''
    Based on seqio.mixing_rate_num_examples: 
    https://seqio.readthedocs.io/en/latest/_modules/seqio/utils.html#mixing_rate_num_examples
    mixing_rate_3k = functools.partial(seqio.mixing_rate_num_examples, maximum=3000)

    And on the formula from Raffel et al. (2020): https://jmlr.org/papers/v21/20-074.html
    The rate of sampling (r) from dataset (m) is:
    r(m) = min(num_samples(m) / sum([min(e(n), k) for all n in [1...num(datasets)]])
    '''
    minima = []
    loaded_datasets = []
    if do_mixed_gold_pipeline:
        # Load from the gold pipeline data
        load_path = config.path + "data/mix_prompts_train/{task}/30000/train"
    else:
        load_path = config.path + "data/{prompt_format}/{task}/30000/train"
    
    for idx in tqdm(range(len(dataset_names))):
        dataset_name = dataset_names[idx]
        dataset = load_from_disk(load_path.format(prompt_format=prompt_format, task=dataset_name))
        k = min(len(dataset), size_limit)
        dataset = dataset.select(list(range(k)))
        loaded_datasets.append(dataset)
        minimum = min(len(dataset), maximum)
        minima.append(minimum)
    sum_minima = sum(minima)

    def get_rate(dataset):
        '''
        Based on seqio.mixing_rate_num_examples
        '''
        num_samples = len(dataset)
        rate = num_samples
        rate *= scale
        if maximum:
            rate = min(num_samples, maximum) / sum_minima
        if temperature != 1.0:
            rate = rate ** (1.0 / temperature)
        return rate

    mixing_rates = [get_rate(dataset) for dataset in loaded_datasets]

    print("MIXING RATES:")
    to_print = [i for i in zip(dataset_names, mixing_rates)]
    for item in to_print:
        print(item[0] + "\t" + str(item[1]))

    dataset_mixture = interleave_datasets(loaded_datasets, probabilities=mixing_rates, seed=config.seed)
    return dataset_mixture


def mix_tasks(prompt_format, do_mixed_gold_pipeline, setup_num):
    tasks_to_mix = []

    # Initialize dict of tasks to drop
    tasks_to_drop = {"nli": [],
                     "commonsense": [],
                     "sentiment": [],
                     "paraphrase": [],
                     "closed_book_qa": [],
                     "reading_comp": [],
                     "reading_comp_with_cs": [],
                     "coreference": [],
                     "misc": [],
                     "struct_to_text_nlg": [],
                     "translation_nlg": [],
                     "summarization_nlg": []
                    }

    # Default: Drop only 2 clusters
    clusters_to_drop = ["misc",
                          "summarization_nlg"]

    # The following setups are for our task-wise ablation experiments.
    # Setup 1: Drop all non-NLG clusters
    clusters_to_drop = ["nli",
                        "commonsense",
                        "sentiment",
                        "paraphrase",
                        "closed_book_qa",
                        "reading_comp",
                        "reading_comp_with_cs",
                        "coreference",
                        "misc",
                        ]
    
    # Setup 2: Drop NLG and NLI clusters
    clusters_to_drop = ["nli", "summarization_nlg", "translation_nlg", "struct_to_text_nlg"]

    # Setup 3: Drop everything, train only on QNLI
    clusters_to_drop = ["commonsense", "sentiment", "paraphrase", "closed_book_qa", "reading_comp", "reading_comp_with_cs", "coreference", "misc", "summarization_nlg", "struct_to_text_nlg", "translation_nlg"]
    tasks_to_drop["nli"] = ["anli_r1", "anli_r2", "anli_r3", "rte", "cb", "snli", "mnli", "wnli"]

    # Setup 4: Drop everything, train only on MNLI
    clusters_to_drop = ["commonsense", "sentiment", "paraphrase", "closed_book_qa", "reading_comp", "reading_comp_with_cs", "coreference", "misc", "summarization_nlg", "struct_to_text_nlg", "translation_nlg"]
    tasks_to_drop["nli"] = ["anli_r1", "anli_r2", "anli_r3", "rte", "cb", "snli", "qnli", "wnli"]
    
    # Setup 5: Drop only CoRef cluster
    clusters_to_drop = ["coreference"]
    
    # Setup 6 & 7: Drop only CoRef cluster, except for Winogrande/DPR
    clusters_to_drop = []
    tasks_to_drop["coreference"] = ["wsc273", "dpr_raw"]
    tasks_to_drop["coreference"] = ["wsc273", "winogrande"]

    # Always drop these tasks from their clusters
    tasks_to_drop["nli"].append("qnli")
    tasks_to_drop["commonsense"].append("story_cloze")
    tasks_to_drop["sentiment"].append("yelp_polarity_reviews")
    tasks_to_drop["paraphrase"].append("stsb")
    tasks_to_drop["closed_book_qa"].append("trivia_qa_full")
    tasks_to_drop["closed_book_qa"].append("trivia_qa_wiki")
    tasks_to_drop["reading_comp"].append("squad")
    tasks_to_drop["reading_comp"].append("squad_v2")
    tasks_to_drop["reading_comp_with_cs"].append("record")
    tasks_to_drop["coreference"].append("wsc273")
    tasks_to_drop["coreference"].append("winogrande")
    tasks_to_drop["misc"].append("fix_punct")
    tasks_to_drop["struct_to_text_nlg"].append("dart")
    tasks_to_drop["translation_nlg"].append("wmt16/ro-en")
    tasks_to_drop["summarization_nlg"].append("samsum")

    for cluster in config.task_clusters:
        if cluster not in clusters_to_drop:
            for task in config.task_clusters[cluster]:
                if task not in tasks_to_drop[cluster]:
                    tasks_to_mix.append(task)

    print("Mixing tasks:", tasks_to_mix)
    
    mixture_dataset = create_mixture(tasks_to_mix, 
                                     prompt_format=prompt_format, 
                                     split="train",
                                     do_mixed_gold_pipeline=do_mixed_gold_pipeline
                                     )
    
    print("Done.")
    mixture_dataset = mixture_dataset.train_test_split(test_size=0.1)
    if not do_mixed_gold_pipeline:
        output_name = config.path + prompt_format + "_mixture_setup" + setup_num
    else:
        output_name = config.path + "/data/ablation_setup{}_train".format(setup_num)

    mixture_dataset.save_to_disk(output_name)
    return


if __name__ == "__main__":
    mix_tasks(prompt_format="mix_prompts",
              do_mixed_gold_pipeline=True,
              setup_num="4"
              )
