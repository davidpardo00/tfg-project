from transcriber_functions import *
import os, sys

# Paso 0: Configuración inicial
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.append(ROOT_DIR)
video_folder = os.path.join(ROOT_DIR, 'data', 'original_videos')
video_name = "BreakingBad_IAmTheDanger.mp4"
video_path = os.path.join(video_folder, video_name)
output_dir = os.path.join(ROOT_DIR, "outputs", "transcripts")

# Paso 1: Transcribir video
result = transcribe_file(video_path, model_name="base", language="en")

# Paso 2: Guardar subtítulos en formato .srt
save_srt(result, output_dir, video_name)

# Paso 3: Leer texto y generar prompt para resumen
txt_path = os.path.join(output_dir, os.path.splitext(video_name)[0] + ".txt")
with open(txt_path, "r", encoding="utf-8") as f:
    texto = f.read()

prompt = resumir_transcripcion(texto, video_name, output_dir)
