import os
import json

import pandas as pd
from invoke import task

from prompts.chatgpt import ChatGPTPrompt


def add_assessment_data(df, chatgpt):
    is_relevant = []
    is_offensive = []
    premises_count = []
    has_conclusion = []
    faults = []
    for ix, row in df.iterrows():
        response_file_path = chatgpt.get_file_path(row["identifier"])
        if not os.path.exists(response_file_path):
            is_relevant.append(None)
            is_offensive.append(None)
            premises_count.append(None)
            has_conclusion.append(None)
            faults.append("missing")
            continue
        with open(response_file_path) as response_file:
            chatgpt_response = json.load(response_file)
            chatgpt_prompt = chatgpt.read_prompt(chatgpt_response)
            if chatgpt_prompt is None:
                is_relevant.append(None)
                is_offensive.append(None)
                premises_count.append(None)
                has_conclusion.append(None)
                faults.append("invalid-json")
                continue
            is_relevant.append(chatgpt_prompt["is_relevant"])
            is_offensive.append(chatgpt_prompt["is_offensive"])
            premises_count.append(len(chatgpt_prompt.get("premises", [])))
            has_conclusion.append(chatgpt_prompt.get("conclusion", None) is not None)
            faults.append(None)
    df["is_relevant"] = pd.Series(is_relevant)
    df["is_offensive"] = pd.Series(is_offensive)
    df["premises_count"] = pd.Series(premises_count)
    df["has_conclusion"] = pd.Series(has_conclusion)
    df["faults"] = pd.Series(faults)


@task(name="analyse-assessments")
def analyse_chatgpt_discourse_assessment(ctx, discourse):
    # We load the dataset into a DataFrame
    raw_file_path = os.path.join(ctx.config.directories.output, f"{discourse}.raw.json")
    with open(raw_file_path) as raw_file:
        data = json.load(raw_file)
    df = pd.DataFrame.from_records([
        {
            "identifier": f"{discourse}/{entry['metadata']['name']}",
            "argument_score": entry["metadata"]["argument_score"],
            "source": entry["metadata"]["source"],
            "text_length": len(entry["text"])
        }
        for entry in data
    ])
    pd.set_option("display.max_rows", None)
    chatgpt = ChatGPTPrompt(ctx.config, "assessment", is_list=False)
    add_assessment_data(df, chatgpt)
    print("GENERAL")
    print(df.shape)
    print(df.head(20))
    print()
    print()
    print("FAULTS")
    print(df["faults"].value_counts())
    print()
    print()
    print("ASSESSMENTS")
    print(df[df["is_relevant"].isin(["yes", "partially"])]["premises_count"].value_counts())
    print(df[df["is_relevant"].isin(["yes", "partially"])]["has_conclusion"].value_counts())
    print()
    print()
    print(df[df["is_offensive"].isin(["yes", "partially"])]["premises_count"].value_counts())
    print(df[df["is_offensive"].isin(["yes", "partially"])]["has_conclusion"].value_counts())
    print()
    print()
    print("IS_RELEVANT SOURCES")
    print(df[df["is_relevant"].isin(["yes", "partially"])]["source"].value_counts().head(20))
    print()
    print()
    print("IS_RELEVANT STATISTICS")
    print("Text length:", df[df["is_relevant"].isin(["yes", "partially"])]["text_length"].mean())
    print("Argument score", df[df["is_relevant"].isin(["yes", "partially"])]["argument_score"].mean())
