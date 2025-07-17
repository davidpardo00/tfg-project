import umap, os, sys
import numpy as np
from classix import CLASSIX
from hdbscan import HDBSCAN
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

def load_embeddings(embedding_path="embeddings/embeddings.npy"):
    """ Carga los embeddings desde un archivo numpy. """
    if not os.path.exists(embedding_path):
        raise FileNotFoundError(f"El archivo {embedding_path} no existe.")
    return np.load(embedding_path)

def reduce_dimensionality(embeddings, method="umap", n_components=2, random_state=42):
    """
    Reduce los embeddings a 2D usando el método especificado: 'umap' o 'pca'.

    :param embeddings: Array de embeddings de forma (n_samples, n_features)
    :param method: 'umap' o 'pca' 
    :return: Embeddings reducidos a 2D
    """
    method = method.lower()
    
    if method == "umap":
        reducer = umap.UMAP(n_components=n_components, random_state=random_state)
    elif method == "pca":
        reducer = PCA(n_components=n_components, random_state=random_state)
    else:
        raise ValueError(f"Método de reducción no válido: {method}")
    
    return reducer.fit_transform(np.array(embeddings))

def cluster_embeddings_HDBSCAN(embeddings, min_cluster_size=5, min_samples=1):
    """
    Aplica clustering HDBSCAN sobre los embeddings.

    :param embeddings: np.ndarray de forma (N, D)
    :param min_cluster_size: tamaño mínimo de un cluster
    :param min_samples: número mínimo de muestras para ser core point
    :return: array de etiquetas
    """
    clusterer = HDBSCAN(min_cluster_size=min_cluster_size, min_samples=min_samples)
    labels = clusterer.fit_predict(embeddings)
    return labels

def cluster_embeddings_CLASSIX(embeddings, radius=2.0, minPts=2):
    """
    Aplica clustering CLASSIX sobre los embeddings con parámetros ajustables.

    :param embeddings: np.ndarray de forma (N, D)
    :param radius: radio de agrupación
    :param minPts: número mínimo de puntos por cluster
    :return: array de etiquetas
    """
    classix = CLASSIX(radius=radius, minPts=minPts)
    classix.fit(embeddings)
    return classix.labels_

