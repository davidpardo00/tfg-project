from functions import * 

# ! Eliminacion de contenido de carpetas
# Directorios de salida
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
OUTPUTS_DIR = os.path.join(ROOT_DIR, 'outputs')
output_dir_images = os.path.join(OUTPUTS_DIR, 'images_scenes')
output_dir_clips = os.path.join(OUTPUTS_DIR, 'clips_video')
output_dir_csv = os.path.join(OUTPUTS_DIR, 'csv_files')

# Eliminar el contenido de las carpetas antes de procesar
setup_output_directories([output_dir_images, output_dir_clips, 
                          output_dir_csv])

# # ! Recortar video
# # Archivo de video a analizar
# video_folder = "Scene_detection/original_videos"
# video_name = "AmIDreaming_MetroBoomin.mp4"
# cut_video_folder = "Scene_detection/cut_videos"
# split_video(video_folder, video_name, "00:00:00", "00:00:30", cut_video_folder)