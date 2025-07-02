from embedding_extraction.functions_embedding import *

# ! Eliminacion de contenido de carpetas
# Directorios de salida
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
OUTPUTS_DIR = os.path.join(ROOT_DIR, 'outputs')
output_dir_plots = os.path.join(OUTPUTS_DIR, 'plots')
output_dir_embed = os.path.join(OUTPUTS_DIR, 'embeddings')
output_dir_transcripts = os.path.join(OUTPUTS_DIR, 'transcripts')
output_dir_images = os.path.join(OUTPUTS_DIR, 'images_scenes')
output_dir_videos = os.path.join(OUTPUTS_DIR, 'clips_video')
output_dir_csv = os.path.join(OUTPUTS_DIR, 'csv_files')

# Eliminar el contenido de las carpetas antes de procesar
setup_output_directories([output_dir_plots, output_dir_embed, 
                          output_dir_transcripts, output_dir_images, 
                          output_dir_videos, output_dir_csv])