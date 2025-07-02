import os
import whisper

def transcribe_file(file_path: str, model_name="base", language="en", format_segments=True):
    # Cargar modelo
    model = whisper.load_model(model_name)

    # Transcribir archivo
    result = model.transcribe(file_path, language=language)

    # Definir nombre y ruta de salida
    filename = os.path.splitext(os.path.basename(file_path))[0]
    output_dir = os.path.join("outputs", "transcripts")
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"{filename}.txt")

    # Guardar transcripción
    with open(output_path, "w", encoding="utf-8") as f:
        if format_segments and "segments" in result:
            for segment in result["segments"]:
                f.write(segment["text"].strip() + "\n\n")
        else:
            f.write(result["text"])

    print(f"Transcripción guardada en: {output_path}")
    return result

def format_timestamp(seconds):
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    ms = int((seconds % 1) * 1000)
    return f"{h:02}:{m:02}:{s:02},{ms:03}"

def save_srt(result, output_dir, original_filename):
    """Guarda los segmentos transcritos en formato .srt (subtítulos)."""
    os.makedirs(output_dir, exist_ok=True)
    filename = os.path.splitext(original_filename)[0] + ".srt"
    output_path = os.path.join(output_dir, filename)

    with open(output_path, "w", encoding="utf-8") as f:
        for i, segment in enumerate(result["segments"], 1):
            start = format_timestamp(segment["start"])
            end = format_timestamp(segment["end"])
            f.write(f"{i}\n{start} --> {end}\n{segment['text'].strip()}\n\n")

    print(f"Subtítulos guardados en: {output_path}")
    return output_path

def resumir_transcripcion(texto: str, video_name: str, output_dir: str, max_chars: int = 3000) -> str:
    """
    Prepara un prompt para resumir la transcripción y lo guarda como .summary.txt.
    """
    if len(texto) > max_chars:
        texto = texto[:max_chars] + "..."

    prompt = (
        "Resume el siguiente fragmento de una transcripción en 5-6 líneas claras y comprensibles, "
        "manteniendo las ideas clave, tono y contexto:\n\n"
        "---- INICIO ----\n"
        f"{texto.strip()}\n"
        "---- FIN ----"
    )

    summary_filename = os.path.splitext(video_name)[0] + ".summary.txt"
    summary_path = os.path.join(output_dir, summary_filename)

    os.makedirs(output_dir, exist_ok=True)
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write(prompt)

    print(f"Prompt para resumen guardado en: {summary_path}")
    return prompt
