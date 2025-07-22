import sys, torch, os
import numpy as np
from tqdm import tqdm
from embedding_extraction.functions_embedding import *
from clustering.functions_clustering import *
from clustering.visualization import *
from classix import CLASSIX

# Paso 0: Configuración inicial
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# device = torch.device("cpu")
print(f">> Dispositivo seleccionado: {device}")

# Directorios
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
VIDEO_DIR = os.path.abspath(os.path.join(ROOT_DIR, '..', 'RESULTADOS EMBEDDINGS', 'clips_videos'))
OUTPUTS_DIR = os.path.join(ROOT_DIR, 'outputs')
output_dir_embed = os.path.join(OUTPUTS_DIR, 'embeddings')
output_dir_plots = os.path.join(OUTPUTS_DIR, 'plots')
output_dir_frames = os.path.join(OUTPUTS_DIR, 'frames_cluster')
os.makedirs(output_dir_embed, exist_ok=True)
os.makedirs(output_dir_plots, exist_ok=True)
os.makedirs(output_dir_frames, exist_ok=True)

# Paso 1: Inicializar modelo
model_used = "clip4clip"  # Puede ser "clip", "siglip", "jinaclip", "clip4clip", "openclip"
print(f">> Inicializando modelo {model_used}...")
preprocess_or_processor, model, model_type = init_model(model_used, device)
print(f"✅ Modelo {model_used} inicializado correctamente en {device}.")

# Paso 2: Procesar todos los vídeos
video_names = []
mean_embeddings = []

video_files = sorted([f for f in os.listdir(VIDEO_DIR) if f.lower().endswith(".mp4")])
print(f">> Detectados {len(video_files)} vídeos: {video_files if len(video_files) <= 3 else video_files[:3] + ['...']}")

if len(video_files) == 0:
    raise ValueError("❌ No se han encontrado vídeos en la carpeta especificada.")

for filename in tqdm(video_files, desc="Procesando vídeos", unit="video"):
    print(f"\n>> Procesando archivo: {filename}")
    video_path = os.path.join(VIDEO_DIR, filename)
    video_name = os.path.splitext(filename)[0]

    try:
        print("   - Extrayendo embeddings de frames...")
        if model_type == "clip4clip":
            embeddings = process_frames_clip4clip_batch(
                video_path, model, device, embedding_path=None
                )
        else:
            embeddings = process_frames(
                video_path, model_type, preprocess_or_processor, 
                model, device, embedding_path=None
                )

        print("   - Embeddings extraídos con shape:", embeddings.shape)

        print("   - Calculando embedding medio...")
        mean_vector = embeddings.mean(axis=0)
        mean_embeddings.append(mean_vector)
        video_names.append(video_name)

        print("   - Guardando frame central...")
        frame_output_path = os.path.join(output_dir_frames, f"{video_name}.jpg")
        save_central_frame_from_video(video_path, frame_output_path)
        print("   - Frame central guardado.")

    except Exception as e:
        print(f"❌ Error procesando {filename}: {e}")

# Paso 3: Guardar embeddings medios
if len(mean_embeddings) == 0:
    raise ValueError("❌ No se pudo procesar ningún vídeo correctamente.")

mean_embeddings = np.vstack(mean_embeddings)
mean_embedding_path = os.path.join(output_dir_embed, f"mean_embeddings_{model_used}.npy")
np.save(mean_embedding_path, mean_embeddings)
print(f"✅ Embeddings medios guardados en: {mean_embedding_path}")

# Paso 4: Guardar los nombres de los vídeos
video_names_path = os.path.join(output_dir_embed, f"video_names_{model_used}.npy")
np.save(video_names_path, np.array(video_names))
print(f"✅ Nombres de vídeos guardados en: {video_names_path}")

# # Paso 5: Clusterizar con embeddings originales (mejor calidad)
# labels = cluster_embeddings_CLASSIX(mean_embeddings)

# # Paso 6: Reducir a 2D solo para visualización
# embeddings_2d = reduce_dimensionality(mean_embeddings)

# # Paso 7: Visualizaciones
# plot_umap(
#     embeddings_2d,
#     save_path=os.path.join(output_dir_plots, f"umap_{model_used}_mean_embeddings.png")
# )
# plot_clusters(
#     embeddings_2d, labels,
#     save_path=os.path.join(output_dir_plots, f"clusters_{model_used}_mean_embeddings.png")
# )
# plot_umap_interactive(
#     embeddings_2d, image_folder=output_dir_frames, labels=labels,
#     save_path=os.path.join(output_dir_plots, f"umap_interactive_{model_used}_mean_embeddings.html")
# )
