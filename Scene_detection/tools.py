from functions import * 

# ! Eliminacion de contenido de carpetas
# Directorios de salida
output_dir_images = "Scene_detection/images_scenes"
output_dir_clips = "Scene_detection/clips_video"
output_dir_csv = "Scene_detection/csv_files"
output_dir_cutvideos = "Scene_detection/cut_videos"

# Eliminar el contenido de las carpetas antes de procesar
setup_output_directories([output_dir_images, output_dir_clips, 
                          output_dir_csv, output_dir_cutvideos])

# # ! Recortar video
# # Archivo de video a analizar
# video_folder = "Scene_detection/original_videos"
# video_name = "AmIDreaming_MetroBoomin.mp4"
# cut_video_folder = "Scene_detection/cut_videos"
# split_video(video_folder, video_name, "00:00:00", "00:00:30", cut_video_folder)