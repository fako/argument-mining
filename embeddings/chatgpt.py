import json
from openai import Embedding
import hashlib
from glob import glob

from chatgpt.fetch import ChatGPTFetchBase


class ChatGPTEmbeddings(ChatGPTFetchBase):

    model = "text-embedding-ada-002"

    @staticmethod
    def get_text_hash(text):
        sha1 = hashlib.sha1()
        sha1.update(text.encode("utf-8"))
        return sha1.hexdigest()

    @staticmethod
    def get_cache_key(file_name, text=None):
        if not text:
            return file_name
        return f"{file_name}.{ChatGPTEmbeddings.get_text_hash(text)}"

    def search_file_paths(self, identifier):
        base_file_path = self.get_file_path(identifier)
        file_pattern = base_file_path.replace(".json", "*.json")
        return glob(file_pattern)

    @staticmethod
    def read_embedding(response, clip=None):
        return response["data"][0]["embedding"][:clip]

    def fetch(self, identifier, text, save=True, dry_run=False):
        if not dry_run:
            cached = self.get_cache(identifier)
            if cached:
                return cached
        print(f"Fetching: {identifier}")
        if dry_run:
            return
        response = Embedding.create(
            input=text,
            model=self.model,
            **self.authentication
        )
        self.set_cache(identifier, response)
        if save:
            self.save()
        else:
            return json.loads(self.data[identifier])
