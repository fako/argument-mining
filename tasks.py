from invoke import Collection
from jinja2 import Environment, PackageLoader

from load import load_stance_classification_dataset, unpack_discourse_dataset
from analysis.stance_classification import (analyse_stance_classification_dataset,
                                            analyse_chatgpt_stance_classification, cluster_collection)
from analysis.discourse import analyse_chatgpt_discourse_assessment, cluster_collection as discourse_cluster_collection
from sampling import sample_stance_classification_dataset, sample_discourse_dataset
from fetch import fetch_collection
from backup import backup_stance_classification_output, sync_discourse_output


stance_classification_collection = Collection(
    "sc",
    load_stance_classification_dataset,
    analyse_stance_classification_dataset,
    analyse_chatgpt_stance_classification,
    sample_stance_classification_dataset,
    backup_stance_classification_output
)


discourse_collection = Collection(
    "dsc",
    unpack_discourse_dataset,
    sample_discourse_dataset,
    sync_discourse_output,
    analyse_chatgpt_discourse_assessment
)


namespace = Collection(
    stance_classification_collection,
    fetch_collection,
    cluster_collection,
    discourse_collection,
    discourse_cluster_collection
)
namespace.configure({
    "prompts": Environment(loader=PackageLoader("prompts"))
})
