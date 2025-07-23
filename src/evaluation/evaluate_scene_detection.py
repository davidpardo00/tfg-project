
from functions_evaluate import *
import os
import pandas as pd
from pathlib import Path
import glob

# Paso 0: Configuración inicial
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..', 'RESULTADOS PYSCENE'))
OUTPUTS_DIR = os.path.join(ROOT_DIR, 'outputs')
TXT_DIR = os.path.join(ROOT_DIR, 'annotations', 'shots')
CSV_DIR = os.path.join(ROOT_DIR, 'automatic_results')
os.makedirs(OUTPUTS_DIR, exist_ok=True)

# Paso 1: Obtener todos los archivos TXT tipo "bbc_*.txt"
manual_files = glob.glob(os.path.join(TXT_DIR, "bbc_*.txt"))
print(f"🔍 Archivos encontrados: {len(manual_files)}")

for manual_txt_path in manual_files:
    video_id = os.path.splitext(os.path.basename(manual_txt_path))[0]
    auto_csv_path = os.path.join(CSV_DIR, f"{video_id}-Scenes.csv")
    output_csv = os.path.join(OUTPUTS_DIR, f"{video_id}_resultados.csv")

    print(f"\n📽️ Procesando video: {video_id}")

    if not os.path.exists(auto_csv_path):
        print(f"⚠️  No se encontró el archivo automático para {video_id}. Saltando.")
        continue

    try:
        # Paso 2: Cargar anotaciones
        print("🚀 Cargando anotaciones manuales...")
        manual_df = load_manual_data(manual_txt_path)
        print(f"✔️  {len(manual_df)} escenas manuales.")

        print("📥 Cargando detecciones automáticas...")
        auto_df = load_auto_data(auto_csv_path)
        print(f"✔️  {len(auto_df)} escenas automáticas.")

        # Paso 3: Emparejamiento
        print("🔍 Emparejando escenas...")
        matches_df = match_scenes(auto_df, manual_df)

        # Paso 4: Evaluación
        results = evaluate_matches(matches_df, len(manual_df), len(auto_df))
        print("📈 Resultados:")
        for key, value in results.items():
            print(f"{key}: {value}")

        # Paso 5: Guardar resultados por video
        matches_df.to_csv(output_csv, index=False)

        # Paso 6: Guardar métricas globales
        metrics_csv = os.path.join(OUTPUTS_DIR, "resumen_metricas.csv")
        results_df = pd.DataFrame([results])
        results_df.insert(0, "video_id", video_id)
        if not os.path.exists(metrics_csv):
            results_df.to_csv(metrics_csv, index=False)
        else:
            results_df.to_csv(metrics_csv, mode='a', header=False, index=False)

        # Paso 7: Histograma de IoU
        plot_iou_distribution(matches_df, video_id, OUTPUTS_DIR)
        print(f"✅ {video_id} procesado correctamente.")

    except Exception as e:
        print(f"❌ Error al procesar {video_id}: {e}")


