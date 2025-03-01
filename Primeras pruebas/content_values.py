import pandas as pd
import matplotlib.pyplot as plt

# Cargar el archivo CSV con los resultados de las estadísticas
df = pd.read_csv("csv_files/video_stats.csv")

# Mostrar las primeras filas del DataFrame para verificar la estructura
print(df.head())

# Extraer la columna 'content_val'
content_vals = df['content_val']

# Generar la gráfica
plt.figure(figsize=(10, 5))
plt.plot(content_vals, label='Content Value', color='b')
plt.title('Gráfica de Content Value')
plt.xlabel('Frame')
plt.ylabel('Content Value')
plt.legend()
plt.grid(True)
plt.show()
