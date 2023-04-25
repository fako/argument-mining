import os
import re
from glob import glob
import json

import pandas as pd
from invoke import task

from constants import STANCE_SIGNAL_WORDS


def normalize_stance_from_text(text):
    if text in STANCE_SIGNAL_WORDS["support"]:
        return "support"
    elif text in STANCE_SIGNAL_WORDS["dispute"]:
        return "dispute"
    else:
        return


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
            normalizations["stance"] = normalize_stance_from_text(metadata["originalStanceText"])
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
