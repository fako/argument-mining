import os
import json
from invoke import task
from load.base import load_dump_file
from prompts.chatgpt import ChatGPTPrompt


def _formatter(obj):
    if obj["model"] != "online_discourse.document" or "tika" not in obj["fields"].get("properties", {}):
        return
    pk = obj["pk"]
    properties = obj["fields"]["properties"]
    text = ""
    for content in properties["content"]:
        if not content.strip():
            continue
        text += content + "\n"
    text = text.strip()
    return {
        "metadata": {
            "topic": None,  # filled in by command
            "url": properties["url"],
            "title": properties["title"],
            "source": properties["source"],
            "author": properties["author"],
            "argument_score": properties.get("argument_score", 0),
            "name": f"document_{pk}",
            "terms": properties["term"]
        },
        "text": text[:ChatGPTPrompt.text_length_limit]
    }


@task(name="unpack-dataset")
def unpack_discourse_dataset(ctx, discourse_file, discourse):
    input_file_path = os.path.join(ctx.config.directories.discourse, discourse_file)
    dataset_directory = os.path.join(ctx.config.directories.discourse, discourse)
    os.makedirs(dataset_directory, exist_ok=True)
    records = []
    for batch in load_dump_file(input_file_path, _formatter):
        for obj in batch:
            if not obj:
                continue
            obj["metadata"]["topic"] = discourse
            json_file_path = os.path.join(dataset_directory, f"{obj['metadata']['name']}.json")
            with open(json_file_path, "w") as json_file:
                json.dump(obj, json_file, indent=4)
            records.append(obj)
    output_file_path = os.path.join(ctx.config.directories.output, f"{discourse}.raw.json")
    with open(output_file_path, "w") as output_file:
        json.dump(records, output_file, indent=4)
