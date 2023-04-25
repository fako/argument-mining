import os
import json

from openai import ChatCompletion


class ChatGPTJSONEncoder(json.JSONEncoder):

    def default(self, obj):
        if hasattr("to_dict", obj):
            return obj.to_dict()


class ChatGPTPrompt(object):

    model = "gpt-3.5-turbo-0301"
    text_length_limit = 1800

    config = None
    prompt_type = None
    context = None
    data = {}

    def __init__(self, config, prompt_type, context=None):
        self.config = config
        self.prompt_type = prompt_type
        self.context = context or {}
        self.data = {}

    def get_file_path(self, identifier):
        return os.path.join(self.config.directories.output, "chatgpt", self.prompt_type, f"{identifier}.json")

    @staticmethod
    def read_prompt(response):
        choice = response["choices"][0]
        raw_content = choice["message"]["content"]
        try:
            json_start = raw_content.index("{")
            json_end = len(raw_content) - raw_content[::-1].index("}")
        except ValueError:  # substring not found
            return
        return json.loads(raw_content[json_start: json_end])

    def fetch(self, identifier, text, save=True, dry_run=False):
        if identifier in self.data and not dry_run:
            print(f"Returning: {identifier}")
            return json.loads(self.data[identifier])
        file_path = self.get_file_path(identifier)
        if os.path.exists(file_path) and not dry_run:
            with open(file_path) as cache_file:
                print(f"Loading: {identifier}")
                return json.load(cache_file)
        print(f"Fetching: {identifier}")
        template = self.config.prompts.get_template(f"{self.prompt_type}.txt")
        prompt = template.render(text=text, **self.context)
        if dry_run:
            print(prompt)
            print("*"*80)
            return
        response = ChatCompletion.create(
            model=self.model,
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,
            max_tokens=2048 - len(prompt),
            top_p=1,
            frequency_penalty=0,
            presence_penalty=0,
            api_key=self.config.secrets.chatgpt,
            organization=self.config.chatgpt.organization
        )
        self.data[identifier] = json.dumps(response, cls=ChatGPTJSONEncoder, indent=4) + "\n"
        if save:
            self.save()
        else:
            return json.loads(self.data[identifier])

    def save(self):
        for identifier, json_data in self.data.items():
            file_path = self.get_file_path(identifier)
            path, file_name = os.path.split(file_path)
            os.makedirs(path, exist_ok=True)
            with open(file_path, "w") as output_file:
                output_file.write(json_data)
        self.data = {}
