import os
import re
from glob import glob
import json

import pandas as pd
from invoke import task

from constants import STANCE_NUMBERS_TO_TARGETS
from load.iterators import PostRecordIterator
from prompts.chatgpt import ChatGPTPrompt
from embeddings.chatgpt import ChatGPTEmbeddings


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


def load_claim_vectors(ctx, scope, topic=None, limit=None):
    limit = int(limit) if limit is not None else None
    chatgpt_embeddings = ChatGPTEmbeddings(ctx.config, "claims")
    post_iterator = PostRecordIterator(
        output_directory=ctx.config.directories.output,
        scope=scope,
        max_text_length=ChatGPTPrompt.text_length_limit,
        limit=limit,
        prompts={"splitting": ChatGPTPrompt(ctx.config, "splitting", is_list=True)},
        embeddings={"claims": chatgpt_embeddings},
        clip_embeddings=None,
        min_text_words=15,
        topic=topic,
    )
    claim_identifiers = []
    claim_texts = []
    claim_vectors = []
    claim_labels = []
    for identifier, record, aspects in post_iterator:
        splitting = aspects.get("splitting", None)
        if not splitting:
            print("Missing splitting prompt:", identifier)
            continue
        for splits in splitting:
            if not isinstance(splits, dict):
                print("Invalid splits type:", type(splits))
                print(splits)
                continue
            claims = splits.get("premises", [])
            conclusion = splits.get("conclusion", None)
            if conclusion:
                claims.append(conclusion)
            for claim in claims:
                claim_identifiers.append(identifier)
                claim_texts.append(claim)
                text_hash = chatgpt_embeddings.get_text_hash(claim)
                vector = aspects["claims"][text_hash]
                claim_vectors.append(vector)
                claim_labels.append(
                    record["normalizations"]["stance"] if topic else record["metadata"]["topic"]
                )
    return claim_vectors, claim_labels, claim_texts, claim_identifiers
