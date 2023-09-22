from invoke import Collection

from fetch.stance_classification import (classify_stance_classification, split_stance_classification,
                                         embeddings_stance_classification)


fetch_collection = Collection(
    "fetch",
    classify_stance_classification,
    split_stance_classification,
    embeddings_stance_classification
)