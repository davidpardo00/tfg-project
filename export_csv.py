import csv
from DeteccionEscenas import scene_list

# Generar archivo CSV con los datos de las escenas
csv_filename = "detected_scenes.csv"

# Abrir el archivo en modo escritura
with open(csv_filename, mode='w', newline='') as csv_file:
    writer = csv.writer(csv_file)

    # Escribir el encabezado del CSV
    writer.writerow(["Scene", "Start time", "End time"])

    # Escribir cada escena detectada
    for i, scene in enumerate(scene_list):
        start_timecode, end_timecode = scene
        start_time = start_timecode.get_timecode()  # Convierte a formato de tiempo
        end_time = end_timecode.get_timecode()  # Convierte a formato de tiempo
        writer.writerow([i + 1, start_time, end_time])

print(f"Archivo CSV con las escenas generadas: {csv_filename}")
