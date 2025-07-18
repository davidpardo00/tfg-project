import os, sys
import numpy as np
from tqdm import tqdm
from classix import CLASSIX
from embedding_extraction.functions_embedding import *
from clustering.functions_clustering import *
from clustering.visualization import *

# Paso 0: Configuracion inicial
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
MSRVTT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..', 'DB', 'MSRVTT', 'videos', 'all'))
OUTPUTS_DIR = os.path.join(ROOT_DIR, 'outputs')
output_dir_plots = os.path.join(OUTPUTS_DIR, 'plots')
output_dir_embed = os.path.join(OUTPUTS_DIR, 'embeddings')
output_dir_frames_clusters = os.path.join(OUTPUTS_DIR, 'frames_cluster')
if os.path.exists(output_dir_frames_clusters):
    setup_output_directories([output_dir_frames_clusters])

# Paso 1: Inicialización modelo elegido
model_used = "siglip" # Puede ser "clip", "siglip", "jinaclip o "clip4clip"
preprocess_or_processor, model, model_type = init_model(model_used, device)
print(f"Modelo {model_used} inicializado correctamente.")

# Paso 2: Generar embeddings a partir de todos los frames del video original
video_path = "data/original_videos/Friends_scene.mp4"
video_name = os.path.splitext(os.path.basename(video_path))[0]
embedding_filename = f"{video_name}_{model_used}.npy"

# Paso 3: Procesar frames y generar embeddings
embedding_path = os.path.join(output_dir_embed, embedding_filename)
embedding_path = process_frames(
    video_path, model_type, preprocess_or_processor, model, device,
    embedding_path=embedding_path
)

# Paso 4: Guardar los frames para visualizacion interactiva
save_frames_from_video(video_path, output_folder=output_dir_frames_clusters, frame_stride=3)

# Paso 5: Cargar y clusterizar embeddings
embeddings = load_embeddings(embedding_path)
embeddings_2d = reduce_dimensionality(embeddings)
labels = cluster_embeddings_CLASSIX(embeddings_2d)

# Paso 6: Visualizar resultados
plot_umap(embeddings_2d, save_path=os.path.join(output_dir_plots, f"umap_{video_name}_{model_used}.png"))
plot_clusters(embeddings_2d, labels, save_path=os.path.join(output_dir_plots, f"clusters_{video_name}_{model_used}.png"))
plot_umap_interactive(
    embeddings_2d,
    image_folder=output_dir_frames_clusters,
    labels=labels,
    save_path=os.path.join(output_dir_plots, f"umap_interactive_{video_name}_{model_used}.html")
)