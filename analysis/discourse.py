import os
import json
from functools import reduce
from copy import deepcopy

import pandas as pd
from invoke import task, Collection
from sklearn.cluster import KMeans, AffinityPropagation
from sklearn.metrics import silhouette_score

from encoders import DataJSONEncoder
from prompts.chatgpt import ChatGPTPrompt
from load.discourse import load_discourse_claim_vectors
from analysis.base import write_tsne_data


def add_splits_data(df, chatgpt):
    stances = []
    premises_count = []
    has_conclusion = []
    faults = []
    for ix, row in df.iterrows():
        response_file_path = chatgpt.get_file_path(row["identifier"])
        if not os.path.exists(response_file_path):
            stances.append(None)
            premises_count.append(None)
            has_conclusion.append(None)
            faults.append("missing")
            continue
        with open(response_file_path) as response_file:
            chatgpt_response = json.load(response_file)
            chatgpt_prompt = chatgpt.read_prompt(chatgpt_response)
            if chatgpt_prompt is None:
                stances.append(None)
                premises_count.append(None)
                has_conclusion.append(None)
                faults.append("invalid-json")
                continue
            stances.append(chatgpt_prompt["stance"])
            premises_count.append(len(chatgpt_prompt.get("premises", [])))
            has_conclusion.append(chatgpt_prompt.get("conclusion", None) is not None)
            faults.append(None)
    df["stance"] = pd.Series(stances)
    df["premises_count"] = pd.Series(premises_count)
    df["has_conclusion"] = pd.Series(has_conclusion)
    df["faults"] = pd.Series(faults)


@task(name="analyse-splits")
def analyse_chatgpt_discourse_splits(ctx, discourse):
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
    chatgpt = ChatGPTPrompt(ctx.config, "splitting_stance", is_list=False)
    add_splits_data(df, chatgpt)
    print("GENERAL")
    print(df.shape)
    print(df.head(20))
    print()
    print()
    print("FAULTS")
    print(df["faults"].value_counts())
    print()
    print()
    print("STANCE SUPPORT")
    print(df[df["stance"].isin(["support"])]["premises_count"].value_counts())
    print(df[df["stance"].isin(["support"])]["has_conclusion"].value_counts())
    print()
    print()
    print("STANCE DISPUTE")
    print(df[df["stance"].isin(["dispute"])]["premises_count"].value_counts())
    print(df[df["stance"].isin(["dispute"])]["has_conclusion"].value_counts())


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


def _analyse_top_clusters(clusters, labels, texts, posts, urls):
    # Here we'll analyse the made clusters and write that to output directory
    cluster_aggregation = {}
    cluster_format = {
        "dispute_posts": 0,
        "dispute": {
            "posts": set(),
            "texts": []
        },
        "support_posts": 0,
        "support": {
            "posts": set(),
            "texts": []
        }
    }
    for cluster, label, text, post, url in zip(clusters, labels, texts, posts, urls):
        if cluster not in cluster_aggregation:
            cluster_aggregation[cluster] = deepcopy(cluster_format)
        cluster_info = cluster_aggregation[cluster]
        cluster_info[label]["texts"].append([url, text])
        if post not in cluster_info[label]["posts"]:
            cluster_info[label]["posts"].add(post)
            cluster_info[f"{label}_posts"] += 1

    def cluster_sort(cluster):
        posts_count = cluster["dispute_posts"] + cluster["support_posts"]
        return posts_count * 100 if cluster["dispute_posts"] and cluster["support_posts"] else posts_count

    return sorted(cluster_aggregation.values(), key=cluster_sort, reverse=True)


@task(name="affinity")
def analyse_chatgpt_embedding_affinity(ctx, scope, discourse, limit=None, top_n=20):

    claim_vectors, claim_labels, claim_texts, claim_posts, claim_urls = load_discourse_claim_vectors(
        ctx, scope, discourse, limit, urls=True
    )

    model = AffinityPropagation(damping=0.75, max_iter=1000)
    claim_clusters = model.fit_predict(claim_vectors)

    write_tsne_data(
        claim_vectors, claim_labels, claim_texts,
        [int(value) for value in claim_clusters]
    )

    top_clusters = _analyse_top_clusters(claim_clusters, claim_labels, claim_texts, claim_posts, claim_urls)
    top_clusters_file_path = os.path.join(
        ctx.config.directories.output,
        f"{discourse}.top-clusters.json"
    )
    with open(top_clusters_file_path, "w") as top_clusters_file:
        json.dump(top_clusters[:top_n], top_clusters_file, indent=4, cls=DataJSONEncoder)


cluster_collection = Collection(
    "dsc-clusters",
    analyse_chatgpt_embedding_tsne,
    analyse_chatgpt_embedding_kmeans,
    analyse_chatgpt_embedding_affinity,
)
