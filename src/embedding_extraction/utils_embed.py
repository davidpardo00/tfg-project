from embedding_extraction.functions_embedding import *

# ! Eliminacion de contenido de carpetas
# Directorios de salida
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
OUTPUTS_DIR = os.path.join(ROOT_DIR, 'outputs')
output_dir_plots = os.path.join(OUTPUTS_DIR, 'plots')
output_dir_embed = os.path.join(OUTPUTS_DIR, 'embeddings')

# Eliminar el contenido de las carpetas antes de procesar
setup_output_directories([output_dir_plots, output_dir_embed])