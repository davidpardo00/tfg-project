import os, shutil
import pandas as pd

# Ruta al archivo Excel con los nombres de los videos (sin extensión o con ella)
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
excel_path = os.path.join(ROOT_DIR, 'RESULTADOS EMBEDDINGS', 'video_selection.xlsx')
columna_nombre = "video_name"  # Cambia al nombre real de la columna si es distinto

# Carpeta de origen (donde están todos los videos)
source_folder = os.path.join(ROOT_DIR, 'DB', 'MSRVTT', 'videos', 'all')

# Carpeta de destino (donde copiarás los videos seleccionados)
destination_folder = os.path.join(ROOT_DIR, 'DB', 'MSRVTT', 'videos', 'DB_MSRVTT')

# Leer Excel
df = pd.read_excel(excel_path, engine="openpyxl")

# Asegurar que la carpeta destino existe
os.makedirs(destination_folder, exist_ok=True)

# Obtener lista de nombres (pueden incluir o no la extensión)
nombres_videos = df[columna_nombre].astype(str).tolist()

# Extensiones de video comunes (puedes añadir más si quieres)
extensiones = ['.mp4', '.avi', '.mov', '.mkv']

# Copiar los archivos
copiados = 0
no_encontrados = []

for nombre in nombres_videos:
    encontrado = False
    for ext in extensiones:
        nombre_archivo = nombre if nombre.lower().endswith(ext) else nombre + ext
        origen = os.path.join(source_folder, nombre_archivo)
        if os.path.exists(origen):
            shutil.copy(origen, destination_folder)
            copiados += 1
            encontrado = True
            break
    if not encontrado:
        no_encontrados.append(nombre)

print(f"✅ {copiados} videos copiados correctamente.")
if no_encontrados:
    print(f"⚠️ No se encontraron {len(no_encontrados)} videos:")
    for nombre in no_encontrados:
        print(f"- {nombre}")
