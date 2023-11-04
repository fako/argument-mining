import os
import json


class BaseRecordIterator(object):

    def load(self, scope):
        data_path = os.path.join(self.output_directory, f"{self.project}.{scope}.json")
        with open(data_path) as data_file:
            data = json.load(data_file)
            if self.topic:
                data = [record for record in data if record["metadata"]["topic"] == self.topic]
        self.data = data[:self.limit]

    def __init__(self, output_directory, scope, max_text_length, project=None, limit=None, prompts=None,
                 embeddings=None, clip_embeddings=None, topic=None, min_text_words=15, cut_off_text=False):
        # Set basic variables
        self.output_directory = output_directory
        self.max_text_length = max_text_length
        self.cut_off_text = cut_off_text
        self.min_text_words = min_text_words
        self.limit = limit
        self.prompts = prompts or {}
        self.embeddings = embeddings or {}
        self.clip_embeddings = clip_embeddings
        self.topic = topic
        self.data = []
        # Load data based on project or class default
        if project:
            self.project = project
        self.load(scope)

    def __iter__(self):
        return self

    def __next__(self):
        try:
            record = self.data.pop(0)
        except IndexError:
            raise StopIteration()
        metadata = record["metadata"]
        identifier = os.path.join(metadata["topic"], metadata["name"])
        if len(record["text"]) > self.max_text_length and not self.cut_off_text:
            print(f"Text too long: {identifier}")
            return self.__next__()
        elif len(record["text"]) > self.max_text_length:
            record["text"] = record["text"][:self.max_text_length]
        if len(record["text"].split()) < self.min_text_words:
            print(f"Text too short: {identifier}")
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
                try:
                    post, embedding_hash, extension = file_path.split(".")
                except ValueError:
                    post, extension = file_path.split(".")
                    embedding_hash = post
                aspects[embeddings_type][embedding_hash] = chatgpt.read_embedding(
                    embeddings_data,
                    clip=self.clip_embeddings
                )
        return identifier, record, aspects


class PostRecordIterator(BaseRecordIterator):
    project = "stance_classification"


class DiscourseRecordIterator(BaseRecordIterator):
    pass
