import os
import json

from sklearn.manifold import TSNE
import numpy as np


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
