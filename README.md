# TFG - Análisis y segmentación semántica de contenido audiovisual mediante modelos de lenguaje multimodal

Este Trabajo de Fin de Grado (TFG) tiene como objetivo desarrollar un sistema automático para la **detección, representación y agrupación semántica de escenas en vídeos**, utilizando modelos de lenguaje multimodal de última generación.

----

## 🧠 Descripción general

El sistema realiza las siguientes etapas:

1. **Segmentación automática de escenas** a partir de vídeos, utilizando [PySceneDetect](https://github.com/Breakthrough/PySceneDetect).
2. **Extracción de embeddings** (representaciones vectoriales) de cada escena con:
   - CLIP (OpenAI)
   - CLIP4Clip
   - SigLip
3. **Agrupamiento semántico** de escenas mediante clustering con HDBSCAN.
4. **Visualización de resultados** en 2D mediante reducción de dimensionalidad con UMAP, coloreando las escenas por grupo semántico.

----

## Estructura
- `src/embedding_extraction`: Scripts para extracción de embeddings con modelo a elegir.
- `src/scene_segmentation`: Scripts para detección y corte de escenas usando PySceneDetect.
- `src/clustering`: Scripts para agrupar y visualizar embeddings.
- `classix/`: Algoritmo 'CLASSIX'.
- `sentencepiece/`: Algoritmo 'sentencepiece'.
- `data/videos_originales`: Directorio vídeos de entrada.
- `outputs/...`: Salidas de resultados de ejecucción.

----

## 👨‍💻 Autor
David Pardo Solano
Grado en Ingeniería de Tecnologías de Telecomunicación
Universidad de Zaragoza

Trabajo Fin de Grado (TFG) – 2025
