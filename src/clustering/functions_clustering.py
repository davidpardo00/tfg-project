import umap, os, sys, torch
import numpy as np
from classix import CLASSIX
from hdbscan import HDBSCAN
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics.pairwise import cosine_similarity

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

def describe_clusters_with_clip(embeddings, labels, tokenizer, model, device, candidate_labels=None):
    """
    Asigna una etiqueta textual a cada clúster usando similitud de embeddings con CLIP.

    Args:
        embeddings (np.ndarray): Embeddings visuales (shape: [N, D])
        labels (np.ndarray): Etiquetas de clústeres (shape: [N])
        tokenizer: Tokenizador de CLIP
        model: Modelo CLIP
        device: Dispositivo ('cpu' o 'cuda')
        candidate_labels (List[str]): Lista de etiquetas textuales a comparar

    Returns:
        dict: {cluster_id: best_label}
    """

    if candidate_labels is None:
        candidate_labels = [
            "man", "woman", "people talking", "city", "nature", "car", "indoor", "outdoor",
            "celebration", "interview", "news", "sports", "walking", "dancing", "laughing",
            "crying", "crowd", "building", "room", "daylight", "night", "animal", "close-up"
        ]

    # Prepara textos y calcula sus embeddings
    text_inputs = tokenizer(candidate_labels, padding=True, return_tensors="pt").to(device)
    with torch.no_grad():
        text_features = model.get_text_features(**text_inputs)
        text_features /= text_features.norm(p=2, dim=-1, keepdim=True)  # normaliza

    # Para cada clúster, calcular su media y comparar
    cluster_descriptions = {}
    for label in np.unique(labels):
        cluster_indices = np.where(labels == label)[0]
        cluster_embedding = embeddings[cluster_indices].mean(axis=0)
        cluster_embedding = cluster_embedding / np.linalg.norm(cluster_embedding)

        similarities = cosine_similarity(
            cluster_embedding.reshape(1, -1),
            text_features.cpu().numpy()
        )
        best_match = candidate_labels[np.argmax(similarities)]
        cluster_descriptions[label] = best_match

    return cluster_descriptions
