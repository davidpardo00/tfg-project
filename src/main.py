import os, sys
from embedding_extraction.clip_embedding import *
from clustering.clustering_tools import *
from clustering.visualization import *

# Paso 0: Configuracion inicial
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
OUTPUTS_DIR = os.path.join(ROOT_DIR, 'outputs')
output_dir_plots = os.path.join(OUTPUTS_DIR, 'plots')
output_dir_embed = os.path.join(OUTPUTS_DIR, 'embeddings')
setup_output_directories([output_dir_plots, output_dir_embed])

# Paso 1: Inicialización modelo elegido
model_used = "siglip"
preprocess_or_processor, model, model_type = init_model(model_used, device)
print(f"Modelo {model_used} inicializado correctamente.")

# Paso 2: Generar embeddings a partir de todos los frames del video original
video_path = "data/original_videos/Friends_scene.mp4"
embedding_path = process_frames(video_path, model_type, preprocess_or_processor, model, device)

# Paso 3: Cargar, procesar y clusterizar embeddings
embeddings = load_embeddings(embedding_path)
embeddings_2d = reduce_dimensionality(embeddings)
labels = cluster_embeddings_CLASSIX(embeddings_2d)

# Paso 4: Visualizar resultados
plot_umap(embeddings_2d)
plot_clusters(embeddings_2d, labels)
