import os
import json

from chatgpt.encoder import ChatGPTJSONEncoder


class ChatGPTFetchBase(object):

    model = None
    fetch_type = None
    data = {}

    def __init__(self, config, fetch_type):
        self.config = config
        self.fetch_type = fetch_type
        self.data = {}

    @property
    def authentication(self):
        return {
            "api_key": self.config.secrets.chatgpt,
            "organization": self.config.chatgpt.organization
        }

    def get_file_path(self, identifier):
        return os.path.join(self.config.directories.output, "chatgpt", self.fetch_type, f"{identifier}.json")

    def get_cache(self, identifier):
        # Look it up in memory
        if identifier in self.data:
            print(f"Returning: {identifier}")
            return json.loads(self.data[identifier])
        # Loop it up on disk
        file_path = self.get_file_path(identifier)
        if os.path.exists(file_path):
            with open(file_path) as cache_file:
                print(f"Loading: {identifier}")
                return json.load(cache_file)
        # Not found
        return

    def set_cache(self, identifier, data):
        self.data[identifier] = json.dumps(data, cls=ChatGPTJSONEncoder, indent=4) + "\n"

    def save(self):
        for identifier, json_data in self.data.items():
            file_path = self.get_file_path(identifier)
            path, file_name = os.path.split(file_path)
            os.makedirs(path, exist_ok=True)
            with open(file_path, "w") as output_file:
                output_file.write(json_data)
        self.data = {}
