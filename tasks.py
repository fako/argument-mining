from invoke import Collection
from jinja2 import Environment, PackageLoader

from load import load_stance_classification
from analysis import analyse_raw_stance_classification
from sampling import sample_stance_classification
from fetch import classify_stance_classification


collection = Collection(
    "sc",
    load_stance_classification,
    analyse_raw_stance_classification,
    sample_stance_classification,
    classify_stance_classification
)
collection.configure({
    "prompts": Environment(loader=PackageLoader("prompts"))
})


namespace = Collection(collection)
