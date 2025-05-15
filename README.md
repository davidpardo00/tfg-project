# 🎓 Proyecto TFG: Análisis y segmentación semántica de contenido audiovisual mediante modelos de lenguaje multimodal

Este proyecto es parte de mi **Trabajo de Fin de Grado (TFG)** en **Ingeniería de Tecnologías de Telecomunicación**. El objetivo es **analizar contenido audiovisual mediante inteligencia artificial**, utilizando técnicas de **segmentación de video**, **generación de embeddings** y **clustering** para detectar patrones, clasificar contenido y realizar análisis semántico de videos.  

🔍 **El sistema permite:**  
✔️ Detección y segmentación automática de escenas en videos con **PySceneDetect**.  
✔️ Generación de **embeddings** de cada escena usando **CLIP**.  
✔️ **Clustering** de las escenas con **HDBSCAN** para encontrar grupos similares de contenido.  
✔️ Comparación de resultados entre **CLIP**, **CLIP4Clip** y **SisLip**.  
✔️ **Visualización** de resultados mediante **UMAP**. 

---

## 🚀 Instalación y Configuración  

### **1️⃣ Requisitos**  
- **Python 3.8+**  
- **PyTorch** (para usar CLIP y otros modelos)  
- **PySceneDetect**  
- **CLIP** (de OpenAI) o **CLIP4Clip**  
- **HDBSCAN** (para clustering)  
- **UMAP** (para reducción de dimensionalidad)  
- **Matplotlib**, **Seaborn** (para visualización)  
