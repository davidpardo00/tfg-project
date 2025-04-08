import umap, hdbscan
from classix import CLASSIX
import numpy as np
import os

def load_embeddings(embedding_path="embeddings/embeddings.npy"):
    """ Carga los embeddings desde un archivo numpy. """
    if not os.path.exists(embedding_path):
        raise FileNotFoundError(f"El archivo {embedding_path} no existe.")
    return np.load(embedding_path)

def reduce_dimensionality(embeddings, n_components=2):
    """ Reduce la dimensionalidad de los embeddings con UMAP. """
    reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, n_components=n_components, metric='euclidean')
    return reducer.fit_transform(embeddings)

def cluster_embeddings_HDBSCAN(embeddings_2d):
    """ Aplica HDBSCAN para clusterizar los embeddings. """
    clusterer = hdbscan.HDBSCAN(min_cluster_size=2, min_samples=1, metric='euclidean')
    return clusterer.fit_predict(embeddings_2d)

# Probar CLASSIX para clusterizar
def cluster_embeddings_CLASSIX(embeddings_2d):
    """ Aplica CLASSIX para clusterizar los embeddings. """
    clusterer = CLASSIX(sorting='pca', radius=0.5, minPts=2)
    clusterer.fit(embeddings_2d)
    return clusterer.labels_

if __name__ == "__main__":
    embeddings = load_embeddings()
    embeddings_2d = reduce_dimensionality(embeddings)
    labels = cluster_embeddings(embeddings_2d)
    np.save("embeddings/embeddings_2d.npy", embeddings_2d)
    np.save("embeddings/labels.npy", labels)
    print(f"Clustering completado. Se han encontrado {len(set(labels))} clusters.")


