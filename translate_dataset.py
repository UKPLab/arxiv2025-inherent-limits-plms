import config
from datasets import load_from_disk
'''
This script was used to edit the translated Social IQA data, which still had
English "Example" and "Solution" headings after API translation.

This translation is hard-coded. Translation was done with ChatGPT4o-mini on 02. October, 2024.
The resulting datasets are already included in the data/translated_socialiqa folder.
'''

def translate_dataset(example):
    text = example["text"]
    text = text.replace("Example 1", "Eisimpleir 1")
    text = text.replace("Example 2", "Eisimpleir 2")
    text = text.replace("Solution", "Fuasgladh")
    example["text"] = text
    return example


if __name__ == "__main__":
    path = config.path + "data/closed-translated+goldic/social_iqa/test_data/2500"
    dataset = load_from_disk(path)
    dataset = dataset.map(translate_dataset)
    dataset.save_to_disk(path)