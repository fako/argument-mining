import os
import re
from glob import glob
import json

from invoke import task


@task()
def load_stance_classification(ctx):
    records = []
    metadata_regex = r"#(?P<key>\w+)=(?P<value>[\w\-]+)"
    stance_classification_pattern = os.path.join(ctx.config.directories.stance_classification, "*/post*")
    for post_file_path in glob(stance_classification_pattern, recursive=True):
        with open(post_file_path, encoding='latin-1') as post_file:
            metadata = {}
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
            records.append({
                "metadata": metadata,
                "text": text
            })
    output_file_path = os.path.join(ctx.config.directories.output, "stance_classification.raw.json")
    with open(output_file_path, "w") as output_file:
        json.dump(records, output_file, indent=4)
