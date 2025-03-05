from scenedetect import SceneManager, VideoManager
from scenedetect.detectors import ContentDetector, AdaptiveDetector
import subprocess, os, shutil
from scenedetect.frame_timecode import FrameTimecode
from scenedetect.video_splitter import split_video_ffmpeg
from scenedetect.scene_manager import save_images, StatsManager
from functions import *

# Directorios de salida
output_dir_images = "Scene_detection/images_scenes"
output_dir_clips = "Scene_detection/clips_video"
output_dir_csv = "Scene_detection/csv_files"
stats_file = "video_stats.csv"

# Eliminar el contenido de las carpetas antes de procesar
setup_output_directories([output_dir_images, output_dir_clips, output_dir_csv])

# Archivo de video a analizar
# video_folder = "Scene_detection/original_videos"
video_folder = "Scene_detection/cut_videos"
video_name = "AmIDreaming_MetroBoomin_cut.mp4"
video_path = analyze_video(video_folder, video_name)

# Cargar el video con VideoManager
video_manager = VideoManager([video_path])
scene_manager = SceneManager()

# Agregar detector de contenido (umbral por defecto = 27) o adaptativo
content_detector = False # Cambiar a False para usar el AdaptiveDetector

if content_detector:
    value_threshold = 27
    scene_manager.add_detector(ContentDetector(threshold=value_threshold))
    print("Detector de escenas utilizado: ContentDetector con umbral igual a", value_threshold)
else:
    scene_manager.add_detector(AdaptiveDetector())
    print("Detector de escenas utilizado: AdaptiveDetector")

# Procesar el video
video_manager.set_downscale_factor()  # Escala el video para mejorar rendimiento
video_manager.start()
scene_manager.detect_scenes(frame_source=video_manager)

# Obtener las escenas detectadas
scene_list = scene_manager.get_scene_list()

# Mostrar resultados
print(f"Se detectaron {len(scene_list)} escenas:")
for i, scene in enumerate(scene_list):
    start_timecode, end_timecode = scene
    print(f"Escena {i+1}: {start_timecode} - {end_timecode}")

# Guardar imágenes de cada escena
save_images(scene_list, video_manager, num_images=1, 
            image_extension="jpg", output_dir=output_dir_images,
            image_name_template="Scene-$SCENE_NUMBER")

print("Imágenes de cada escena generadas con éxito. Guardadas en", output_dir_images)

# Dividir el video en escenas
split_scenes(video_path, scene_list, output_dir_clips)

# Crear archivos CSV con el minutaje de las escenas y estadísticas del video
create_csv_files(video_path, output_dir_csv, stats_file)

# Liberar recursos
video_manager.release()
