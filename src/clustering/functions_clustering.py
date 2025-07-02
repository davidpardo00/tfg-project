import umap, hdbscan
import os, sys
import numpy as np

# Añadir el path de la carpeta `classix` a Python
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.append(os.path.join(ROOT_DIR, "classix"))
from classix import CLASSIX

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
def cluster_embeddings_CLASSIX(embeddings):
    """ Aplica CLASSIX para clusterizar los embeddings. """
    clusterer = CLASSIX(sorting='pca', group_merging='density', 
                        radius=0.5, minPts=10)
    clusterer.fit(embeddings)
    return clusterer.labels_


