import json
from json.decoder import JSONDecodeError
from time import sleep

import openai.error
from openai import ChatCompletion
from chatgpt.fetch import ChatGPTFetchBase


class ChatGPTPrompt(ChatGPTFetchBase):

    model = "gpt-3.5-turbo-0301"
    text_length_limit = 1600
    context = None

    def __init__(self, config, prompt_type, context=None, is_list=False):
        super().__init__(config, prompt_type)
        self.context = context or {}
        self.is_list = is_list
        self.data = {}

    def read_prompt(self, response):
        choice = response["choices"][0]
        raw_content = choice["message"]["content"]
        start_character = "{" if not self.is_list else "["
        end_character = "}" if not self.is_list else "]"
        try:
            json_start = raw_content.index(start_character)
            json_end = len(raw_content) - raw_content[::-1].index(end_character)
        except ValueError:  # substring not found
            return
        try:
            return json.loads(raw_content[json_start: json_end])
        except JSONDecodeError:  # hallucinating output
            return

    def fetch(self, identifier, text, save=True, dry_run=False):
        if not dry_run:
            cached = self.get_cache(identifier)
            if cached:
                return cached
        print(f"Fetching: {identifier}")
        template = self.config.prompts.get_template(f"{self.fetch_type}.txt")
        prompt = template.render(text=text, **self.context)
        if dry_run:
            print(prompt)
            print("*"*80)
            return
        for attempt in range(0, 3):
            try:
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
                    **self.authentication
                )
                break
            except (openai.error.RateLimitError, openai.error.APIError):
                print("Going to sleep to relieve API")
                sleep(30)
                continue
        else:
            raise RuntimeError("Too many ChatGPT requests")
        self.set_cache(identifier, response)
        if save:
            self.save()
        else:
            return json.loads(self.data[identifier])
