import json
from openai import Embedding

from chatgpt.fetch import ChatGPTFetchBase


class ChatGPTEmbeddings(ChatGPTFetchBase):

    model = "text-embedding-ada-002"

    def fetch(self, identifier, texts, save=True, dry_run=False):
        if not dry_run:
            cached = self.get_cache(identifier)
            if cached:
                return cached
        print(f"Fetching: {identifier}")
        if dry_run:
            return
        responses = [
            Embedding.create(
                input=text,
                model=self.model,
                **self.authentication
            )
            for text in texts
        ]
        self.set_cache(identifier, responses)
        if save:
            self.save()
        else:
            return json.loads(self.data[identifier])
