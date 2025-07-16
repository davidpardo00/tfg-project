from scenedetect import SceneManager, VideoManager
import subprocess, os, shutil
from scenedetect.frame_timecode import FrameTimecode
from scenedetect.video_splitter import split_video_ffmpeg
from scenedetect.scene_manager import save_images, StatsManager
from functions_segmentation import *

# Paso 0: Configuracion inicial de directorios y archivos
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
OUTPUTS_DIR = os.path.join(ROOT_DIR, 'outputs')
output_dir_images = os.path.join(OUTPUTS_DIR, 'images_scenes')
output_dir_clips = os.path.join(OUTPUTS_DIR, 'clips_video')
output_dir_csv = os.path.join(OUTPUTS_DIR, 'csv_files')
output_dir_plots = os.path.join(OUTPUTS_DIR, 'plots')
stats_file = "video_stats.csv"
setup_output_directories([output_dir_images, output_dir_clips, output_dir_csv, output_dir_plots])

# Paso 1: Elegir video a analizar y cargar con VideoManager
DB_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..', 'DB'))
video_folder = os.path.join(DB_DIR, 'BBC')
video_name = "bbc_09.mp4"
video_path = analyze_video(video_folder, video_name)
video_manager = VideoManager([video_path])
scene_manager = SceneManager()

# Paso 2: Elegir detector de contenido
select_scene_detector(scene_manager, "adaptive")

# Paso 3: Procesar el video
video_manager.set_downscale_factor()  # Escala el video para mejorar rendimiento
video_manager.start()
scene_manager.detect_scenes(frame_source=video_manager)

# Paso 4: Obtener las escenas detectadas
scene_list = scene_manager.get_scene_list()

# Paso 5: Mostrar resultados
print(f"Se detectaron {len(scene_list)} escenas:")
for i, scene in enumerate(scene_list):
    start_timecode, end_timecode = scene
    print(f"Escena {i+1}: {start_timecode} - {end_timecode}")

# Paso 6: Guardar imágenes de cada escena
save_images(scene_list, video_manager, num_images=1, 
            image_extension="jpg", output_dir=output_dir_images,
            image_name_template="Scene-$SCENE_NUMBER")

print("Imágenes de cada escena generadas con éxito. Guardadas en", output_dir_images)

# Paso 7: Dividir el video en escenas
split_scenes(video_path, scene_list, output_dir_clips)

# Paso 8: Recortar el primer fotograma de cada clip y sobreescribir el archivo original
trim_first_frame_all_clips(output_dir_clips)

# Paso 9: Crear archivos CSV con el minutaje de las escenas y estadísticas del video
create_csv_files(video_path, output_dir_csv, stats_file)

# Paso 10: Gráfica de valores de contenido
plot_content_value(os.path.join(output_dir_csv, stats_file))

# Paso 11: Liberar recursos
video_manager.release()