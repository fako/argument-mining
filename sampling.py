import os
import json
from random import shuffle

import pandas as pd
from invoke import task, Exit


@task(
    name="sample-dataset",
    iterable=["exclude"]
)
def sample_stance_classification_dataset(ctx, exclude, sample=10):
    raw_file_path = os.path.join(ctx.config.directories.output, "stance_classification.raw.json")
    with open(raw_file_path) as raw_file:
        records = json.load(raw_file)
    df = pd.DataFrame.from_records(records)
    df["topic"] = df["metadata"].apply(lambda row: row["topic"])
    topics = set(df["topic"].value_counts().keys().tolist())
    for exclusion in exclude:
        if exclusion not in topics:
            raise Exit(f"Unknown topic to exclude: {exclusion}")
        topics.remove(exclusion)
    sample_records = []
    for topic in topics:
        output = df[df["topic"] == topic].sample(sample)
        output.drop("topic", axis=1)
        sample_records += list(output.to_dict(orient="index").values())
    sample_file_path = os.path.join(ctx.config.directories.output, "stance_classification.sample.json")
    with open(sample_file_path, "w") as sample_file:
        json.dump(sample_records, sample_file, indent=4)


@task(name="sample-dataset", iterable=["exclude"])
def sample_discourse_dataset(ctx, discourse, sample=10, print_text=True, randomize=True):
    discourse_file_path = os.path.join(ctx.config.directories.discourse, discourse)
    json_files = []
    with os.scandir(discourse_file_path) as entries:
        for ix, entry in enumerate(entries):
            if not entry.is_file() or not entry.name.endswith(".json"):
                continue
            json_files.append(entry.path)
    if randomize:
        shuffle(json_files)
    else:
        json_files.sort()
    sample_file_paths = json_files[:sample]
    sample_documents = []
    for sample_file_path in sample_file_paths:
        with open(sample_file_path) as sample_file:
            doc = json.load(sample_file)
            sample_documents.append(doc)
            if print_text:
                print(doc["text"])
                print("*"*80)
    sample_file_path = os.path.join(ctx.config.directories.output, f"{discourse}.sample.json")
    with open(sample_file_path, "w") as sample_file:
        json.dump(sample_documents, sample_file, indent=4)
