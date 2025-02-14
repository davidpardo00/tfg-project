from scenedetect import SceneManager, VideoManager
from scenedetect.detectors import ContentDetector
import subprocess

# Archivo de video a analizar
video_path = "CircleOfLove_RudyMancuso.mp4"

# Cargar el video con VideoManager
video_manager = VideoManager([video_path])
scene_manager = SceneManager()

# Agregar detector de contenido con el umbral por defecto (27) o ajustarlo
scene_manager.add_detector(ContentDetector(threshold=30))  

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

# # Dividir el video en escenas  
# for i, scene in enumerate(scene_list):
#     start_time, end_time = scene
#     output_file = f"escena_{i+1}.mp4"

#     command = [
#         "ffmpeg", "-i", video_path,
#         "-ss", str(start_time.get_seconds()), "-to", str(end_time.get_seconds()),
#         "-c", "copy", output_file
#     ]

#     subprocess.run(command)

# print("Cortes de escenas generados con éxito.")

# Liberar recursos
video_manager.release()
