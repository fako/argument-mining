import os
import json
from functools import reduce

import pandas as pd
from invoke import task, Collection
import spacy
from spacy_arguing_lexicon import ArguingLexiconParser
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans, AffinityPropagation, DBSCAN
from sklearn.metrics import silhouette_score
import numpy as np

from constants import STANCE_ZERO_SHOT_TARGETS
from load import save_stance_classification_dataframe, load_stance_classification_dataframe, load_claim_vectors
from prompts.chatgpt import ChatGPTPrompt


@task(name="analyse-dataset")
def analyse_stance_classification_dataset(ctx):
    raw_file_path = os.path.join(ctx.config.directories.output, "stance_classification.raw.json")
    with open(raw_file_path) as raw_file:
        data = json.load(raw_file)
    df = pd.DataFrame.from_records([entry["metadata"] for entry in data])
    pd.set_option("display.max_rows", None)
    print(df.shape)
    print()
    print(df["stance"].value_counts())
    print()
    stance_texts = df["originalStanceText"].value_counts()
    print(stance_texts)
    print()
    for topic in ["abortion", "creation", "gayRights", "god", "guns", "healthcare"]:
        print(topic)
        print("-"*len(topic))
        print(df[df["topic"] == topic]["originalStanceText"].value_counts())
        print()
    print()
    for topic in ["abortion", "creation", "gayRights", "god", "guns", "healthcare"]:
        print(topic)
        print("-"*len(topic))
        print(df[df["topic"] == topic]["originalTopic"].value_counts())
        print()
    for topic in ["abortion", "creation", "gayRights", "god", "guns", "healthcare"]:
        print(topic)
        print("-"*len(topic))
        print(df[df["topic"] == topic]["stance"].value_counts())
        print()


def add_classification_data(df, chatgpt):
    topic_predictions = []
    is_multi_class_topic = []
    stance_predictions = []
    is_multi_class_stance = []
    faults = []
    for ix, row in df.iterrows():
        response_file_path = chatgpt.get_file_path(row["identifier"])
        if not os.path.exists(response_file_path):
            topic_predictions.append(None)
            is_multi_class_topic.append(None)
            stance_predictions.append(None)
            is_multi_class_stance.append(None)
            faults.append("missing")
            continue
        with open(response_file_path) as response_file:
            chatgpt_response = json.load(response_file)
            chatgpt_prompt = chatgpt.read_prompt(chatgpt_response)
            if chatgpt_prompt is None:
                topic_predictions.append(None)
                is_multi_class_topic.append(None)
                stance_predictions.append(None)
                is_multi_class_stance.append(None)
                faults.append("invalid-json")
                continue
            topic_predictions.append(chatgpt_prompt["topic"])
            is_multi_class_topic.append("|" in chatgpt_prompt["topic"])
            stance_predictions.append(chatgpt_prompt["stance"])
            is_multi_class_stance.append("|" in chatgpt_prompt["stance"])
            faults.append(None)
    df["topic_prediction"] = pd.Series(topic_predictions)
    df["is_multi_class_topic"] = pd.Series(is_multi_class_topic)
    df["stance_prediction"] = pd.Series(stance_predictions)
    df["is_multi_class_stance"] = pd.Series(is_multi_class_stance)
    df["faults"] = pd.Series(faults)
    df["topic_success"] = df.apply(
        lambda row: row["topic"] in row["topic_prediction"] if row["topic_prediction"] else False,
        axis=1
    )
    df["stance_success"] = df.apply(
        lambda row: row["stance"] in row["stance_prediction"] if row["stance_prediction"] else False,
        axis=1
    )


def print_value_counts(df, column):
    print(column)
    print("-"*len(column))
    print(df[column].value_counts())


def print_accuracy(df, target):
    success_values = df[f"{target}_success"].value_counts()
    print_value_counts(df, f"{target}_success")
    print("Accuracy:", success_values[True] / df.shape[0])


@task(name="analyse-classification")
def analyse_chatgpt_stance_classification(ctx, write=False):
    # We load the dataset into a DataFrame
    raw_file_path = os.path.join(ctx.config.directories.output, "stance_classification.raw.json")
    nlp = spacy.load("en_core_web_sm")
    nlp.add_pipe(ArguingLexiconParser(lang=nlp.lang))
    with open(raw_file_path) as raw_file:
        data = json.load(raw_file)
    df = load_stance_classification_dataframe(ctx, "chatgpt")
    if df is None:
        documents = nlp.pipe([entry["text"] for entry in data])
        df = pd.DataFrame.from_records([
            {
                "identifier": os.path.join(entry["metadata"]["topic"], entry["metadata"]["name"]),
                "topic": STANCE_ZERO_SHOT_TARGETS[entry["metadata"]["topic"]],
                "stance": entry["normalizations"]["stance"],
                "argument_score": len(list(document._.arguments.get_argument_spans())) / len(list(document.sents)),
                "text_length": len(entry["text"])
            }
            for entry, document in zip(data, documents)
        ])
        save_stance_classification_dataframe(ctx, "chatgpt", df)
    # We enrich the data through ChatGPT
    chatgpt = ChatGPTPrompt(ctx.config, "classification", is_list=False)
    add_classification_data(df, chatgpt)
    # We filter rows with texts that are: too long (672), didn't get JSON output (252) or contain few arguments
    df = df[df["faults"].isnull()]
    df = df[df["argument_score"] > 0.1]
    df = df[df["text_length"] > 100]
    print(df.head(5))
    print(df.shape)
    print(df.describe())
    print()
    print_value_counts(df, "is_multi_class_topic")
    print()
    print_value_counts(df, "is_multi_class_stance")
    print()
    print_accuracy(df, "topic")
    print()
    print_accuracy(df, "stance")
    print()
    for topic in STANCE_ZERO_SHOT_TARGETS.values():
        print(topic.upper())
        topic_frame = df[df["topic"] == topic]
        print_accuracy(topic_frame, "stance")
        print()
        if not write:
            continue
        stance_error_sample_ids = set([
            sample["identifier"]
            for ix, sample in topic_frame[topic_frame["stance_success"] == False].sample(10).iterrows()
        ])
        samples = []
        for entry in data:
            if os.path.join(entry["metadata"]["topic"], entry["metadata"]["name"]) in stance_error_sample_ids:
                samples.append(entry)
        topic = topic.replace(" ", "+")
        errors_file_path = os.path.join(ctx.config.directories.output, f"stance_classification.errors-{topic}.json")
        with open(errors_file_path, "w") as errors_file:
            json.dump(samples, errors_file, indent=4)


def write_tsne_data(vectors, labels, texts, clusters=None, file_name="data.json"):
    clusters = clusters if clusters else [1 for ix in range(0, len(vectors))]
    tsne = TSNE()
    Y = tsne.fit_transform(np.array(vectors))
    data = []
    for x, y, label, text, cluster in zip(Y[:, 0], Y[:, 1], labels, texts, clusters):
        data.append({
            "coordinates": {
                "x": float(x),
                "y": float(y)
            },
            "label": label,
            "cluster": cluster,
            "text": text
        })
    with open(os.path.join("visualizations", "tsne", file_name), "w") as dump_file:
        json.dump(data, dump_file, indent=4)


@task(name="tsne")
def analyse_chatgpt_embedding_tsne(ctx, scope, topic=None, limit=None):
    claim_vectors, claim_labels, claim_texts = load_claim_vectors(ctx, scope, topic, limit)
    write_tsne_data(claim_vectors, claim_labels, claim_texts)


@task(name="kmeans")
def analyse_chatgpt_embedding_kmeans(ctx, scope, topic=None, limit=None):

    claim_vectors, claim_labels, claim_texts = load_claim_vectors(ctx, scope, topic, limit)

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


@task(name="affinity")
def analyse_chatgpt_embedding_affinity(ctx, scope, topic=None, limit=None):

    claim_vectors, claim_labels, claim_texts = load_claim_vectors(ctx, scope, topic, limit)

    model = AffinityPropagation(damping=0.9)
    claim_clusters = model.fit_predict(claim_vectors)

    write_tsne_data(claim_vectors, [int(value) for value in claim_clusters], claim_texts)


@task(name="dbscan")
def analyse_chatgpt_embedding_dbscan(ctx, scope, topic=None, limit=None):

    claim_vectors, claim_labels, claim_texts = load_claim_vectors(ctx, scope, topic, limit)

    model = DBSCAN(min_samples=20)
    claim_clusters = model.fit_predict(claim_vectors)

    write_tsne_data(claim_vectors, [int(value) for value in claim_clusters], claim_texts)


@task(name="affinity-with-dbscan-filter")
def analyse_chatgpt_embedding_affinity_dbscan_filter(ctx, scope, topic=None, limit=None):

    claim_vectors, claim_labels, claim_texts = load_claim_vectors(ctx, scope, topic, limit)

    model = DBSCAN(min_samples=20)
    claim_dbscan_clusters = model.fit_predict(claim_vectors)
    dbscan_mask = np.array([value >= 0 for value in claim_dbscan_clusters])

    affinity_vectors = np.array(claim_vectors)[dbscan_mask]
    affinity_texts = np.array(claim_texts)[dbscan_mask]
    model = AffinityPropagation(damping=0.5, max_iter=1000)
    claim_affinity_clusters = model.fit_predict(affinity_vectors)

    write_tsne_data(affinity_vectors, [int(value) for value in claim_affinity_clusters], affinity_texts)


cluster_collection = Collection(
    "clusters",
    analyse_chatgpt_embedding_tsne,
    analyse_chatgpt_embedding_kmeans,
    analyse_chatgpt_embedding_affinity,
    analyse_chatgpt_embedding_dbscan,
    analyse_chatgpt_embedding_affinity_dbscan_filter
)
