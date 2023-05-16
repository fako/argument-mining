import os
import re
from glob import glob
import json

import pandas as pd
from invoke import task

from constants import STANCE_NUMBERS_TO_TARGETS


@task(name="load-dataset")
def load_stance_classification_dataset(ctx):
    records = []
    metadata_regex = r"#(?P<key>\w+)=(?P<value>[\w\-]+)"
    stance_classification_pattern = os.path.join(ctx.config.directories.stance_classification, "*/post*")
    for post_file_path in glob(stance_classification_pattern, recursive=True):
        with open(post_file_path, encoding='latin-1') as post_file:
            metadata = {}
            normalizations = {}
            text = ""
            for line in post_file.readlines():
                metadata_match = re.match(pattern=metadata_regex, string=line)
                if metadata_match:
                    metadata[metadata_match.group("key")] = metadata_match.group("value")
                else:
                    text += line.strip() + "\n"
            post_file_path_segments = post_file_path.split(os.path.sep)
            metadata["name"] = post_file_path_segments[-1]
            metadata["topic"] = post_file_path_segments[-2]
            normalizations["stance"] = STANCE_NUMBERS_TO_TARGETS[metadata["stance"]]
            if not normalizations["stance"]:
                continue
            records.append({
                "metadata": metadata,
                "normalizations": normalizations,
                "text": text
            })
    output_file_path = os.path.join(ctx.config.directories.output, "stance_classification.raw.json")
    with open(output_file_path, "w") as output_file:
        json.dump(records, output_file, indent=4)


def save_stance_classification_dataframe(ctx, name, df):
    path = os.path.join(ctx.config.directories.output, "frames")
    os.makedirs(path, exist_ok=True)
    file_path = os.path.join(path, f"{name}.pkl")
    df.to_pickle(file_path)


def load_stance_classification_dataframe(ctx, name):
    file_path = os.path.join(ctx.config.directories.output, "frames", f"{name}.pkl")
    if not os.path.exists(file_path):
        return
    return pd.read_pickle(file_path)


class PostRecordIterator(object):

    def __init__(self, output_directory, scope, max_text_length, limit=None, prompts=None):
        self.output_directory = output_directory
        self.max_text_length = max_text_length
        self.limit = limit
        self.prompts = prompts or []
        data_path = os.path.join(output_directory, f"stance_classification.{scope}.json")
        with open(data_path) as data_file:
            data = json.load(data_file)
        self.data = data[:limit]

    def __iter__(self):
        return self

    def __next__(self):
        try:
            record = self.data.pop(0)
        except IndexError:
            raise StopIteration()
        metadata = record["metadata"]
        identifier = os.path.join(metadata["topic"], metadata["name"])
        if len(record["text"]) > self.max_text_length:
            print(f"Text too long: {identifier}")
            return self.__next__()
        aspects = {}
        for prompt_type, chatgpt in self.prompts.items():
            with open(chatgpt.get_file_path(identifier), "r") as prompt_file:
                prompt_data = json.load(prompt_file)
                aspects[prompt_type] = chatgpt.read_prompt(prompt_data)
        return identifier, record, aspects
