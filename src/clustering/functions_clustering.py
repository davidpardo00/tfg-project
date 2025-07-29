import umap, os, sys, torch
import numpy as np
import pandas as pd
from classix import CLASSIX
from sklearn.cluster import HDBSCAN
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics.pairwise import cosine_similarity
from collections import defaultdict

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

def cluster_embeddings_HDBSCAN(embeddings, min_cluster_size=5, min_samples=1, 
                               cluster_selection_epsilon=0.5, max_cluster_size=None):
    """
    Aplica clustering HDBSCAN sobre los embeddings.

    :param embeddings: np.ndarray de forma (N, D)
    :param min_cluster_size: tamaño mínimo de un cluster
    :param min_samples: número mínimo de muestras para ser core point
    :return: array de etiquetas
    """
    clusterer = HDBSCAN(min_cluster_size=min_cluster_size, 
                        min_samples=min_samples, 
                        cluster_selection_epsilon=cluster_selection_epsilon,
                        max_cluster_size=max_cluster_size)
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

def describe_clusters_with_clip(embeddings, labels, clip_processor, clip_model, device, top_k=3):
    labels = np.array(labels)
    descriptions = {}
    
    # Descripciones base para probar
    candidate_texts = [
    # Escenas humanas y emociones
    "a person talking", "a person smiling", "a person crying", "a person shouting", "a person arguing",
    "a person dancing", "a person sleeping", "a person studying", "a person reading a book", "a person cooking",
    "a person eating", "a person painting", "a person writing", "a person running", "a person walking",
    "a person jumping", "a person driving", "a person meditating", "a person praying", "a person laughing",

    # Grupos de personas
    "a crowd", "a group of friends", "a family dinner", "a classroom", "a business meeting",
    "a press conference", "a wedding", "a funeral", "a protest", "a concert audience",

    # Profesiones
    "a doctor with a patient", "a scientist in a lab", "a teacher in front of a class", "a news anchor on TV",
    "a construction worker", "a firefighter", "a pilot", "a chef", "a police officer", "a judge in court",

    # Lugares interiores
    "a hospital room", "a kitchen", "a classroom", "a library", "a church interior",
    "a garage", "a movie theater", "a restaurant", "a living room", "a bedroom",

    # Lugares exteriores
    "a forest", "a desert", "a mountain", "a beach", "a city at night",
    "a busy street", "a rural road", "a park", "a market", "a train station",

    # Actividades físicas
    "people playing football", "people playing basketball", "people riding bicycles", "a person swimming",
    "a person skateboarding", "a person climbing", "a person lifting weights", "a yoga class",
    "a boxing match", "a martial arts demonstration",

    # Entornos naturales
    "a waterfall", "a lake", "a stormy sky", "a sunrise", "a snowy mountain",
    "a volcano", "a canyon", "a coral reef", "a cave", "a field of flowers",

    # Fenómenos atmosféricos
    "a lightning storm", "heavy rain", "snow falling", "a rainbow", "fog over a forest",

    # Vehículos y transportes
    "a car driving", "a train passing", "a boat sailing", "a plane taking off", "a helicopter landing",
    "a person riding a horse", "a person riding a motorcycle", "a bicycle race", "a traffic jam", "a taxi ride",

    # Tecnología y dispositivos
    "a person using a computer", "a computer screen", "a smartphone in hand", "a video call", "a VR headset",
    "a robot walking", "a drone flying", "a 3D printer", "a server room", "a smart home device",

    # Escenas de acción
    "a fight scene", "a car chase", "an explosion", "a building on fire", "a robbery in progress",
    "a police arrest", "a sniper aiming", "a military operation", "a spy mission", "a superhero landing",

    # Escenas abstractas o conceptuales
    "a time-lapse of a city", "a surreal landscape", "a glitch effect", "a dream sequence", "a simulation",
    "an abstract animation", "a digital art scene", "a fractal animation", "a generative artwork", "a floating object",

    # Cultura y sociedad
    "a traditional dance", "a cultural festival", "a parade", "a religious ritual", "a street artist performing",
    "a wedding ceremony", "a fashion show", "a tattoo artist working", "a food market", "a cosplay convention",

    # Ciencia y educación
    "a chemistry experiment", "a physics demonstration", "a science fair", "a child learning", "a space lecture",
    "a diagram on a whiteboard", "a DNA model", "a microscope view", "a person coding", "a biology lesson",

    # Deportes y juegos
    "a soccer match", "a basketball game", "a tennis match", "a chess game", "a poker table",
    "a board game", "an eSports competition", "a golf course", "a bowling alley", "a gym workout",

    # Escenas familiares y cotidianas
    "a family watching TV", "a couple cooking", "a child drawing", "a parent helping with homework", "a birthday party",
    "a baby sleeping", "a dog playing", "a cat on a couch", "a person washing dishes", "a person folding laundry",

    # Narrativas visuales
    "a flashback scene", "a dream", "a hallucination", "a crime scene", "a rescue mission",
    "a news report", "a weather forecast", "a documentary interview", "a confessional video", "a surveillance footage",

    # Pantallas y gráficos
    "a stock market graph", "a weather map", "a medical scan", "a video editing timeline", "a game HUD",

    # Medios y entretenimiento
    "a cartoon", "an animated movie", "a puppet show", "a film noir scene", "a vintage TV show",
    "a music video", "a live concert", "a talent show", "a TV commercial", "a horror movie scene",

    # Lugares arquitectónicos
    "a skyscraper", "a cathedral", "an abandoned building", "a castle", "a modern office",
    "a museum hall", "a subway station", "a prison cell", "a warehouse", "a bridge at night",

    # Transiciones y efectos
    "a fade to black", "a zoom in", "a camera pan", "a split screen", "a color filter effect"
    ]

    # Preprocesar texto
    inputs = clip_processor(text=candidate_texts, return_tensors="pt", padding=True).to(device)
    text_embeddings = clip_model.get_text_features(**inputs)
    text_embeddings = text_embeddings / text_embeddings.norm(dim=-1, keepdim=True)
    text_embeddings = text_embeddings.to(dtype=torch.float32)  # por si hay conflicto con dtype

    for label in np.unique(labels):
        if label == -1:
            continue

        cluster_embeddings = embeddings[labels == label]
        if len(cluster_embeddings) == 0:
            continue

        centroid = torch.tensor(cluster_embeddings.mean(axis=0)).unsqueeze(0).to(device)
        centroid = centroid / centroid.norm(dim=-1, keepdim=True)
        centroid = centroid.to(dtype=torch.float32)

        similarity = torch.matmul(centroid, text_embeddings.T).squeeze(0)
        top_indices = similarity.topk(top_k).indices
        best_descriptions = [candidate_texts[i] for i in top_indices]

        descriptions[label] = ", ".join(best_descriptions)

    return descriptions

def calcular_metricas_clustering(labels):
    df = pd.DataFrame({"label": labels})
    num_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    cluster_sizes = df["label"].value_counts()
    max_cluster_size = cluster_sizes[cluster_sizes.index != -1].max()
    mean_cluster_size = cluster_sizes[cluster_sizes.index != -1].mean()
    if -1 in cluster_sizes:
        noise_ratio = cluster_sizes[-1] / len(labels)
    else:
        noise_ratio = 0.0

    return {
        "Nº clústeres": str(num_clusters),
        "Tamaño medio de clúster": f"{mean_cluster_size:.2f}",
        "Tamaño máximo de clúster": str(max_cluster_size),
        "Puntos marcados como ruido": f"{round(noise_ratio * 100, 1)}%"
    }
