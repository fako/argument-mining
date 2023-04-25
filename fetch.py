import os
import json

from invoke import task

from constants import STANCE_ZERO_SHOT_TARGETS
from prompts.chatgpt import ChatGPTPrompt


@task(name="classify")
def classify_stance_classification(ctx, scope, dry_run=False, limit=None):
    limit = int(limit) if limit is not None else None
    chatgpt = ChatGPTPrompt(ctx.config, "classification", context={
        "topics": list(STANCE_ZERO_SHOT_TARGETS.values())
    })
    data_path = os.path.join(ctx.config.directories.output, f"stance_classification.{scope}.json")
    with open(data_path) as data_file:
        data = json.load(data_file)
    for record in data[:limit]:
        metadata = record["metadata"]
        identifier = os.path.join(metadata["topic"], metadata["name"])
        if len(record["text"]) > chatgpt.text_length_limit:
            print(f"Text too long: {identifier}")
            continue
        chatgpt.fetch(identifier, record["text"], dry_run=dry_run)
