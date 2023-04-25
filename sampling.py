import os
import json

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
