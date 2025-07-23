# 🎬 TFG – Análisis y segmentación semántica de contenido audiovisual mediante modelos multimodales

Este Trabajo de Fin de Grado tiene como objetivo desarrollar un sistema automático para la **detección, representación y agrupación semántica de escenas en vídeo**, utilizando modelos multimodales de última generación.

El sistema permite explorar vídeos sin etiquetar, extrayendo información semántica mediante embeddings generados por distintos modelos, agrupando fragmentos similares y visualizando los resultados de forma interactiva.

---

## 🧠 Descripción general

El flujo del sistema se compone de las siguientes etapas:

1. **Segmentación automática de escenas**, utilizando [PySceneDetect](https://github.com/Breakthrough/PySceneDetect).
2. **Transcripción automática de audio**, con [Whisper](https://github.com/openai/whisper).
3. **Extracción de embeddings semánticos**, mediante los siguientes modelos:
   - CLIP (OpenAI)
   - OpenCLIP (LAION)
   - SigLIP (Google)
   - CLIP4Clip (vídeo)
4. **Clustering no supervisado** de escenas utilizando:
   - HDBSCAN
   - CLASSIX
5. **Reducción de dimensionalidad** con:
   - UMAP
   - PCA
6. **Visualización interactiva de resultados** mediante Streamlit, permitiendo:
   - Comparar modelos, algoritmos y parámetros
   - Examinar los clústeres con previsualización de cada escena
   - Descargar visualizaciones y estadísticas

---

## 📁 Estructura del repositorio

```

tfg-project-main/
├── data/                  # Vídeos originales y recortes
├── outputs/               # Resultados intermedios y finales (embeddings, gráficos, transcripciones…)
├── src/
│   ├── scene_segmentation/                   # Segmentación de vídeo con PySceneDetect
│   ├── transcribe_videos/                    # Transcripción automática con Whisper
│   ├── embedding_extraction/                 # Extracción de embeddings con distintos modelos
│   ├── evaluation/                           # Modulo de evaluación numerica de resultados
│   ├── clustering/                           # Algoritmos de agrupamiento y visualización
    │       ├── streamlit\_visualizer.py      # Interfaz interactiva para explorar los resultados
│   └── utils/                                # Utilidades y herramientas auxiliares

├── requirements.txt              # Dependencias del entorno
└── README.md

````

---

## 🚀 Ejecución rápida

1. Instalar dependencias:
   ```bash
   pip install -r requirements.txt
   ````

2. Ejecutar visualización interactiva:

   ```bash
   streamlit run streamlit_visualizer.py
   ```

---

## 👨‍💻 Autor

**David Pardo Solano**
Grado en Ingeniería de Tecnologías de Telecomunicación
Universidad de Zaragoza
Trabajo Fin de Grado – Curso 2024/2025

```
