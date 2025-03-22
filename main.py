from clip_embedding import process_frames
from clustering import *
from visualization import *

# Paso 1: Generar embeddings
embedding_path = process_frames()

# Paso 2: Cargar y procesar embeddings
embeddings = load_embeddings(embedding_path)
embeddings_2d = reduce_dimensionality(embeddings)
labels = cluster_embeddings(embeddings_2d)

# Paso 3: Visualizar
plot_umap(embeddings_2d)
plot_clusters(embeddings_2d, labels)
