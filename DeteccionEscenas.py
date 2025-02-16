from scenedetect import SceneManager, VideoManager
from scenedetect.detectors import ContentDetector, AdaptiveDetector
import subprocess, os, shutil
from scenedetect.frame_timecode import FrameTimecode
from scenedetect.video_splitter import split_video_ffmpeg
from scenedetect.scene_manager import save_images

# Archivo de video a analizar
video_folder = "original_videos"
video_name = "CircleOfLove_RudyMancuso.mp4"
video_path = os.path.join(video_folder, video_name)

# Directorios de salida
output_dir_images = "images_scenes"
output_dir_clips = "clips_video"

# Eliminar el contenido de las carpetas antes de procesar
for folder in [output_dir_images, output_dir_clips]:
    if os.path.exists(folder):
        shutil.rmtree(folder)  # Borra toda la carpeta
    os.makedirs(folder)  # La vuelve a crear vacía

# Cargar el video con VideoManager
video_manager = VideoManager([video_path])
scene_manager = SceneManager()

# Agregar detector de contenido (umbral por defecto = 27) o adaptativo
scene_manager.add_detector(ContentDetector(threshold=27))
# scene_manager.add_detector(AdaptiveDetector())  

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

print("Imágenes de cada escena generadas con éxito.")

# Dividir el video en escenas 

for i, scene in enumerate(scene_list):
    start_time, end_time = scene
    output_file = os.path.join(output_dir_clips, f"Clip_{i+1}.mp4")

    command = [
        "ffmpeg", "-i", video_path,
        "-ss", str(start_time.get_seconds()), "-to", str(end_time.get_seconds()),
        "-c", "copy", output_file
    ]

    subprocess.run(command, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    # ffmpeg silenciado con stdout y stderr a DEVNULL

print("Clips de escenas generados con éxito.")

# Liberar recursos
video_manager.release()
