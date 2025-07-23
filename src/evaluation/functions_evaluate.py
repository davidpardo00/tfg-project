import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os, csv

def load_manual_data(txt_path):
    """
    Carga un archivo .txt con anotaciones manuales en formato start_frame\tend_frame.

    Parámetros:
    - txt_path (str): Ruta al archivo .txt.

    Retorna:
    - pd.DataFrame: DataFrame con columnas ['start_frame', 'end_frame', 'scene_id'].
    """
    manual_df = pd.read_csv(txt_path, sep="\t", header=None, names=["start_frame", "end_frame"])
    manual_df["scene_id"] = range(1, len(manual_df) + 1)
    return manual_df

def load_auto_data(csv_path):
    """
    Carga un archivo .csv generado automáticamente por el sistema, omitiendo la primera fila
    y esperando que contenga columnas 'Start Frame' y 'End Frame'.

    Parámetros:
    - csv_path (str): Ruta al archivo .csv.

    Retorna:
    - pd.DataFrame: DataFrame con columnas ['start_frame', 'end_frame', 'scene_id'].
    """
    delimiter = detect_delimiter(csv_path)

    auto_df = pd.read_csv(csv_path, sep=delimiter, skiprows=1)
    auto_df.columns = [col.lower().strip().replace(" ", "_") for col in auto_df.columns]

    if "start_frame" not in auto_df.columns or "end_frame" not in auto_df.columns:
        raise ValueError("El CSV automático debe contener columnas 'start_frame' y 'end_frame'.")
    
    auto_df = auto_df[["start_frame", "end_frame"]]
    auto_df["scene_id"] = range(1, len(auto_df) + 1)
    return auto_df

def compute_iou(start1, end1, start2, end2):
    """
    Calcula el Intersection over Union (IoU) entre dos segmentos temporales.

    Parámetros:
    - start1, end1 (int): Intervalo 1 (automático).
    - start2, end2 (int): Intervalo 2 (manual).

    Retorna:
    - float: Valor de IoU entre los dos intervalos.
    """
    intersection = max(0, min(end1, end2) - max(start1, start2) + 1)
    union = max(end1, end2) - min(start1, start2) + 1
    return intersection / union if union > 0 else 0

def match_scenes(auto_df, manual_df, iou_threshold=0.5):
    """
    Empareja escenas automáticas con las manuales usando IoU.

    Parámetros:
    - auto_df (pd.DataFrame): Escenas detectadas automáticamente.
    - manual_df (pd.DataFrame): Escenas anotadas manualmente.
    - iou_threshold (float): Umbral mínimo de IoU para considerar un emparejamiento.

    Retorna:
    - pd.DataFrame: Resultados con columnas ['auto_scene', 'matched_manual_scene', 'iou', 'match'].
    """
    matches = []
    for _, auto in auto_df.iterrows():
        best_iou = 0
        best_manual_id = None
        for _, manual in manual_df.iterrows():
            iou = compute_iou(auto.start_frame, auto.end_frame, manual.start_frame, manual.end_frame)
            if iou > best_iou:
                best_iou = iou
                best_manual_id = manual.scene_id
        matches.append({
            "auto_scene": auto.scene_id,
            "matched_manual_scene": best_manual_id,
            "iou": best_iou,
            "match": best_iou >= iou_threshold
        })
    return pd.DataFrame(matches)

def evaluate_matches(matches_df, total_manual, total_auto):
    """
    Evalúa la precisión, recall y F1-score de los emparejamientos de escenas.

    Parámetros:
    - matches_df (pd.DataFrame): DataFrame de emparejamientos.
    - total_manual (int): Número total de escenas manuales.
    - total_auto (int): Número total de escenas automáticas.

    Retorna:
    - dict: Diccionario con métricas de evaluación.
    """
    true_positives = matches_df["match"].sum()
    precision = true_positives / total_auto if total_auto > 0 else 0
    recall = true_positives / total_manual if total_manual > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0
    return {
        "true_positives": true_positives,
        "false_positives": total_auto - true_positives,
        "false_negatives": total_manual - true_positives,
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1_score": round(f1, 4)
    }

def plot_iou_distribution(matches_df, video_id, output_folder):
    """
    Genera un histograma de la distribución de IoU de los emparejamientos.

    Parámetros:
    - matches_df (pd.DataFrame): DataFrame con columna 'iou'
    - video_id (str): identificador del video (ej. 'bbc_01')
    - output_folder (str): carpeta donde guardar la imagen

    Salida:
    - str: ruta del archivo PNG guardado
    """
    plt.figure(figsize=(8, 4))
    sns.histplot(matches_df["iou"], bins=20, kde=True, color="skyblue")
    plt.axvline(0.5, color="red", linestyle="--", label="Umbral = 0.5")
    plt.title(f"Distribución de IoU - {video_id}")
    plt.xlabel("IoU")
    plt.ylabel("Frecuencia")
    plt.legend()
    output_path = os.path.join(output_folder, f"{video_id}_iou_histograma.png")
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    print(f"📊 Histograma de IoU guardado en: {output_path}")
    return output_path

def detect_delimiter(file_path):
    with open(file_path, 'r', newline='', encoding='utf-8') as f:
        sample = f.read(1024)
        sniffer = csv.Sniffer()
        return sniffer.sniff(sample).delimiter
