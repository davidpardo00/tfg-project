from clip_embedding import *
from clustering import *
from visualization import *

# Paso 0: Configuracion y borrado de los directorios de salida
output_dir_plots = "plots"
output_dir_embed = "embeddings"
setup_output_directories([output_dir_plots, output_dir_embed])

# Paso 1: Inicialización automática
init_clip()

# Paso 2: Generar embeddings a partir de todos los frames del video original
video_path = "Scene_detection/original_videos/Friends_scene.mp4"
embedding_path = process_frames(video_path)

# Paso 3: Cargar y procesar embeddings
embeddings = load_embeddings(embedding_path)
embeddings_2d = reduce_dimensionality(embeddings)
labels = cluster_embeddings_CLASSIX(embeddings_2d)

# Paso 4: Visualizar
plot_umap(embeddings_2d)
plot_clusters(embeddings_2d, labels)
