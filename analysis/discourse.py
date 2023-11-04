import os
import json
from functools import reduce

import pandas as pd
from invoke import task, Collection
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

from prompts.chatgpt import ChatGPTPrompt
from load.discourse import load_discourse_claim_vectors
from analysis.base import write_tsne_data


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


@task(name="tsne")
def analyse_chatgpt_embedding_tsne(ctx, scope, discourse, limit=None):
    claim_vectors, claim_labels, claim_texts, _ = load_discourse_claim_vectors(ctx, scope, discourse, limit)
    write_tsne_data(claim_vectors, claim_labels, claim_texts)


@task(name="kmeans")
def analyse_chatgpt_embedding_kmeans(ctx, scope, discourse, limit=None):

    claim_vectors, claim_labels, claim_texts, _ = load_discourse_claim_vectors(ctx, scope, discourse, limit)

    models = {}
    scores = {}
    for n_clusters in range(2, 101):
        model = KMeans(n_clusters=n_clusters, n_init="auto")
        claim_clusters = model.fit_predict(claim_vectors)
        score = silhouette_score(claim_vectors, claim_clusters)
        scores[n_clusters] = score
        models[n_clusters] = model

    print(json.dumps(scores, indent=4))

    best_model_key = reduce(lambda rsl, inp: rsl if rsl[1] > inp[1] else inp, scores.items())[0]
    print(f"Found best model {best_model_key} with score {scores[best_model_key]}")
    best_model = models[best_model_key]
    best_model_claim_clusters = best_model.predict(claim_vectors)

    write_tsne_data(
        claim_vectors, claim_labels, claim_texts,
        [int(value) for value in best_model_claim_clusters]
    )


cluster_collection = Collection(
    "dsc-clusters",
    analyse_chatgpt_embedding_tsne,
    analyse_chatgpt_embedding_kmeans,
)
