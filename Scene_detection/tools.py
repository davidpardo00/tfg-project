import os
import shutil

# ! Eliminacion de contenido de carpetas
# Directorios de salida
output_dir_images = "Scene_detection/images_scenes"
output_dir_clips = "Scene_detection/clips_video"
output_dir_csv = "Scene_detection/csv_files"

# Eliminar el contenido de las carpetas antes de procesar
for folder in [output_dir_images, output_dir_clips, output_dir_csv]:
    if os.path.exists(folder):
        shutil.rmtree(folder)   # Borra toda la carpeta
    os.makedirs(folder)         # La vuelve a crear vacía