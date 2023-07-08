from invoke import Collection
from jinja2 import Environment, PackageLoader

from load import load_stance_classification_dataset
from analysis import (analyse_stance_classification_dataset, analyse_chatgpt_stance_classification,
                      analyse_chatgpt_embedding_clusters)
from sampling import sample_stance_classification_dataset
from fetch import classify_stance_classification, split_stance_classification, embeddings_stance_classification
from backup import backup_stance_classification_output


collection = Collection(
    "sc",
    load_stance_classification_dataset,
    analyse_stance_classification_dataset,
    sample_stance_classification_dataset,
    classify_stance_classification,
    analyse_chatgpt_stance_classification,
    split_stance_classification,
    embeddings_stance_classification,
    analyse_chatgpt_embedding_clusters,
    backup_stance_classification_output
)
collection.configure({
    "prompts": Environment(loader=PackageLoader("prompts"))
})


namespace = Collection(collection)
