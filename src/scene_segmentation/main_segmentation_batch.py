from scenedetect import SceneManager, VideoManager
import subprocess, os, shutil
from scenedetect.frame_timecode import FrameTimecode
from scenedetect.scene_manager import save_images, StatsManager
from functions_segmentation import *

# Paso 0: Configuración inicial de directorios de salida
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
OUTPUTS_DIR = os.path.join(ROOT_DIR, 'outputs')
output_dir_clips = os.path.join(OUTPUTS_DIR, 'clips_video')
setup_output_directories([output_dir_clips])

# Paso 1: Carpeta con los videos a procesar
DB_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..', 'DB'))
video_folder = os.path.join(DB_DIR, 'MSRVTT', 'videos', 'DB_MSRVTT')

# Paso 2: Procesar cada video secuencialmente
video_files = [f for f in os.listdir(video_folder) if f.lower().endswith(('.mp4', '.avi', '.mov', '.mkv'))]

for video_name in video_files:
    print(f"\n🔄 Procesando video: {video_name}")
    video_path = analyze_video(video_folder, video_name)
    video_manager = VideoManager([video_path])
    scene_manager = SceneManager()
    select_scene_detector(scene_manager, "adaptive")

    try:
        video_manager.set_downscale_factor()
        video_manager.start()
        scene_manager.detect_scenes(frame_source=video_manager)
        scene_list = scene_manager.get_scene_list()

        print(f"🎬 Se detectaron {len(scene_list)} escenas en {video_name}")
        for i, scene in enumerate(scene_list):
            start_timecode, end_timecode = scene
            print(f"   Escena {i+1}: {start_timecode} - {end_timecode}")

        # Dividir video en clips
        split_scenes(video_path, scene_list, output_dir_clips)

    except Exception as e:
        print(f"❌ Error procesando {video_name}: {e}")

    finally:
        video_manager.release()

# Paso 3: Recortar primer frame de cada clip
trim_first_frame_all_clips(output_dir_clips)

print("\n✅ Procesamiento por lote finalizado.")
