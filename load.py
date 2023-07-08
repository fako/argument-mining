import os
import re
from glob import glob
import json

import pandas as pd
from invoke import task

from constants import STANCE_NUMBERS_TO_TARGETS
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


class PostRecordIterator(object):

    def __init__(self, output_directory, scope, max_text_length, limit=None, prompts=None, embeddings=None,
                 clip_embeddings=None, topic=None):
        self.output_directory = output_directory
        self.max_text_length = max_text_length
        self.limit = limit
        self.prompts = prompts or {}
        self.embeddings = embeddings or {}
        self.clip_embeddings = clip_embeddings
        self.topic = topic
        data_path = os.path.join(output_directory, f"stance_classification.{scope}.json")
        with open(data_path) as data_file:
            data = json.load(data_file)
            if self.topic:
                data = [record for record in data if record["metadata"]["topic"] == self.topic]
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
            prompt_file_path = chatgpt.get_file_path(identifier)
            if not os.path.exists(prompt_file_path):
                aspects[prompt_type] = None
                continue
            with open(prompt_file_path, "r") as prompt_file:
                prompt_data = json.load(prompt_file)
                aspects[prompt_type] = chatgpt.read_prompt(prompt_data)
        for embeddings_type, chatgpt in self.embeddings.items():
            embedding_file_paths = chatgpt.search_file_paths(identifier)
            if not len(embedding_file_paths):
                aspects[embeddings_type] = None
                continue
            aspects[embeddings_type] = {}
            for embedding_file_path in embedding_file_paths:
                tail, file_path = os.path.split(embedding_file_path)
                with open(embedding_file_path, "r") as embeddings_file:
                    embeddings_data = json.load(embeddings_file)
                post, embedding_hash, extension = file_path.split(".")
                aspects[embeddings_type][embedding_hash] = chatgpt.read_embedding(embeddings_data, clip=3)
        return identifier, record, aspects


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
        clip_embeddings=3,
        topic=topic,
    )
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
                claim_texts.append(claim)
                text_hash = chatgpt_embeddings.get_text_hash(claim)
                vector = aspects["claims"][text_hash]
                claim_vectors.append(vector)
                claim_labels.append(
                    record["normalizations"]["stance"] if topic else record["metadata"]["topic"]
                )
    return claim_vectors, claim_labels, claim_texts
