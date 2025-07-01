import os, shutil, subprocess
from scenedetect.video_splitter import split_video_ffmpeg
from scenedetect import SceneManager, VideoManager
from scenedetect.detectors import ContentDetector, AdaptiveDetector
from scenedetect.scene_manager import save_images, StatsManager

def analyze_video(video_folder, video_name):
    """
    Analiza el video especificado y devuelve la ruta completa del archivo de video.
    
    :param video_folder: Carpeta donde se encuentra el video.
    :param video_name: Nombre del archivo de video.
    :return: Ruta completa del archivo de video.
    """
    video_path = os.path.join(video_folder, video_name)
    print("Analizando el siguiente video:", video_name)
    return video_path

def setup_output_directories(output_dirs):
    """
    Elimina el contenido de las carpetas especificadas salvo 
    los archivos .gitkeep y las vuelve a crear vacías.
    
    :param output_dirs: Lista de rutas de las carpetas a limpiar.
    """
    for directory in output_dirs:
        if os.path.exists(directory):
            for item in os.listdir(directory):
                item_path = os.path.join(directory, item)
                if os.path.isfile(item_path) and item != '.gitkeep':
                    os.remove(item_path)
                elif os.path.isdir(item_path):
                    shutil.rmtree(item_path)
        else:
            os.makedirs(directory)
    print("Carpetas de salida limpias y listas para guardar resultados.")

def split_video(video_folder, video_name, start_time, end_time, cut_video_folder):
    """
    Recorta un video utilizando ffmpeg.
    
    :param video_folder: Carpeta donde se encuentra el video.
    :param video_name: Nombre del archivo de video.
    :param start_time: Tiempo de inicio del recorte (formato HH:MM:SS).
    :param end_time: Tiempo de fin del recorte (formato HH:MM:SS).
    :param cut_video_folder: Carpeta donde se guardará el video recortado.
    """
    video_path = os.path.join(video_folder, video_name)
    output_name = f"{os.path.splitext(video_name)[0]}_cut.mp4"
    output_path = os.path.join(cut_video_folder, output_name)
    command = [
        'ffmpeg', '-ss', start_time, '-i', video_path, '-to', end_time,
        '-copyts',        # 1. Mantiene los timestamps originales
        '-avoid_negative_ts', '1',
        '-c', 'copy',     # 2. Evita re-encoding para mayor precisión
        output_path
    ]
    subprocess.run(command, check=True)
    print("Video recortado con éxito. Guardado en", output_path)

def split_scenes(video_path, scene_list, output_dir_clips):
    """
    Divide el video en clips basados en las escenas detectadas y guarda los clips en el directorio especificado.
    
    :param video_path: Ruta del archivo de video.
    :param scene_list: Lista de escenas detectadas.
    :param output_dir_clips: Directorio donde se guardarán los clips de video.
    """
    for i, scene in enumerate(scene_list):
        start_time, end_time = scene
        output_file = os.path.join(output_dir_clips, f"Clip_{i+1}.mp4")

        command = [
            "ffmpeg", "-i", video_path, 
            "-ss", str(start_time.get_seconds()), "-to", str(end_time.get_seconds()), 
            output_file
        ]

        subprocess.run(command, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        # ffmpeg silenciado con stdout y stderr a DEVNULL

    print("Clips de escenas generados con éxito. Guardados en", output_dir_clips)

def create_csv_files(video_path, output_dir_csv, stats_file):
    """
    Crea archivos CSV con el minutaje de las escenas y las estadísticas del video.
    
    :param video_path: Ruta del archivo de video.
    :param output_dir_csv: Directorio donde se guardarán los archivos CSV.
    :param stats_file: Nombre del archivo de estadísticas.
    """
    command = [
        "scenedetect", "-i", video_path, "list-scenes",
        "--output", output_dir_csv
    ]
    subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    command = [
        "scenedetect",
        "--input", video_path, "--stats", stats_file,
        "--output", output_dir_csv, "detect-adaptive"
    ]
    subprocess.run(command, check=True)
    print(f"Análisis de escenas completado. Estadísticas guardadas en {stats_file}")

def trim_first_frame_overwrite(file_path, frame_offset, codec="libx264"):
    """
    Recorta el video reencodificando para eliminar el primer frame y
    sobreescribe el archivo original.
    
    :param file_path: Ruta del archivo de video a procesar.
    :param frame_offset: Tiempo en segundos para iniciar el clip.
    :param codec: Códec de video para reencodificar (por defecto "libx264").
    """
    temp_file = file_path + "_trim.mp4"
    command = [
        "ffmpeg", "-hide_banner", "-loglevel", "error",
        "-y",  # Sobrescribe sin preguntar
        "-ss", str(frame_offset), "-i", file_path,
        "-c:v", codec,   # Reencodea video usando libx264
        "-c:a", "copy",  # Copia el audio sin reencodear
        temp_file
    ]
    subprocess.run(command, check=True)
    os.replace(temp_file, file_path)  # Sobrescribe el archivo original con el temporal