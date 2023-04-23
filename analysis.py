import os
import json

import pandas as pd
from invoke import task


@task()
def analyse_raw_stance_classification(ctx):
    raw_file_path = os.path.join(ctx.config.directories.output, "stance_classification.raw.json")
    with open(raw_file_path) as raw_file:
        data = json.load(raw_file)
    df = pd.DataFrame.from_records([entry["metadata"] for entry in data])
    pd.set_option("display.max_rows", None)
    print(df.shape)
    print()
    print(df["stance"].value_counts())
    print()
    print(df["originalStanceText"].value_counts())
    print()
    for topic in ["abortion", "creation", "gayRights", "god", "guns", "healthcare"]:
        print(topic)
        print("-"*len(topic))
        print(df[df["topic"] == topic]["originalTopic"].value_counts())
        print()
