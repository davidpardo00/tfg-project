import os, sys, glob, clip, torch 
import numpy as np
import streamlit as st
import plotly.express as px
import pandas as pd
from PIL import Image
from transformers import CLIPProcessor, CLIPModel

# Agregar carpeta src al path
SRC_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if SRC_DIR not in sys.path:
    sys.path.append(SRC_DIR)

from clustering.functions_clustering import *

# --- CONFIGURACIÓN GLOBAL ---
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
EMBEDDING_DIR = os.path.join(ROOT_DIR, "RESULTADOS EMBEDDINGS", "embeddings")
FRAME_DIR = os.path.join(ROOT_DIR, "RESULTADOS EMBEDDINGS", "frames_cluster")

# --- CARGAR EMBEDDINGS ---
# Buscar todos los archivos de embeddings disponibles
embedding_files = sorted(glob.glob(os.path.join(EMBEDDING_DIR, "mean_embeddings_*.npy")))
if not embedding_files:
    raise FileNotFoundError("❌ No se encontró ningún archivo 'mean_embeddings_*.npy' en la carpeta de embeddings.")

# Crear una lista legible para el selectbox
embedding_options = [os.path.basename(f).replace("mean_embeddings_", "").replace(".npy", "") for f in embedding_files]
model_name = st.sidebar.selectbox("Selecciona el modelo de embeddings:", embedding_options)

# Cargar los archivos seleccionados
embedding_file = os.path.join(EMBEDDING_DIR, f"mean_embeddings_{model_name}.npy")
video_names_file = os.path.join(EMBEDDING_DIR, f"video_names_{model_name}.npy")

# Mostrar información del modelo cargado
st.sidebar.markdown(f"📌 <b>Modelo cargado:</b> <code>{model_name}</code>", unsafe_allow_html=True)

# Cargar datos
embeddings = np.load(embedding_file)
video_names = np.load(video_names_file)

# --- STREAMLIT CONFIG ---
st.set_page_config(page_title="Visualización Clustering", layout="wide", initial_sidebar_state="expanded")
st.markdown("""
    <style>
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
    </style>
""", unsafe_allow_html=True)

st.title(f"Visualización Interactiva de Embeddings por Clustering - Modelo: {model_name}")

# --- SIDEBAR ---
method = st.sidebar.selectbox("Método de clustering", ["CLASSIX", "HDBSCAN"])
reduction = st.sidebar.selectbox("Reducción de dimensionalidad", ["UMAP", "PCA"])

# Clustering params
if method == "CLASSIX":
    radius = st.sidebar.slider("Radio", 0.1, 5.0, 0.4, 0.1)
    minPts = st.sidebar.slider("Mínimo puntos", 1, 20, 5)
else:
    min_cluster_size = st.sidebar.slider("Tamaño mínimo del cluster", 2, 30, 5)
    min_samples = st.sidebar.slider("Muestras mínimas", 1, 10, 1)
    cluster_selection_epsilon = st.sidebar.slider("Cluster selection epsilon", 0.0, 1.0, 0.0, 0.01)
    use_max_cluster_size = st.sidebar.checkbox("Usar tamaño máximo de clúster", value=False)

    if use_max_cluster_size:
        max_cluster_size = st.sidebar.slider("Tamaño máximo del cluster", 10, 1000, 500, 10)
    else:
        max_cluster_size = None

# Reducción params
embeddings_2d = reduce_dimensionality(embeddings, method=reduction, n_components=2, random_state=42)

# --- CLUSTERING ---
if method == "CLASSIX":
    labels = cluster_embeddings_CLASSIX(embeddings, radius=radius, minPts=minPts)
elif method == "HDBSCAN":
    labels = cluster_embeddings_HDBSCAN(embeddings, min_cluster_size=min_cluster_size, 
                                        min_samples=min_samples, 
                                        cluster_selection_epsilon=cluster_selection_epsilon,
                                        max_cluster_size=max_cluster_size)

# --- DATAFRAME PARA PLOTLY ---
df = pd.DataFrame(embeddings_2d, columns=["x", "y"])
df["label"] = labels
df["video"] = video_names
df["image_path"] = df["video"].apply(lambda x: os.path.join(FRAME_DIR, f"{x}.jpg"))

# --- FILTROS OPCIONALES ---
clusters_to_hide = st.sidebar.multiselect("Ocultar clusters", options=sorted(df["label"].unique()))
if clusters_to_hide:
    df = df[~df["label"].isin(clusters_to_hide)]

# --- GRAFICA INTERACTIVA ---
st.subheader("Embeddings agrupados por clustering")
fig = px.scatter(
    df,
    x="x",
    y="y",
    color=df["label"].astype(str),
    hover_name="video",
    custom_data=["image_path", "label"],  # Añadimos label aquí
    color_discrete_sequence=px.colors.qualitative.Set1,
)

fig.update_traces(
    hovertemplate="""
    <b>%{hovertext}</b><br><br>
    <img src='%{customdata[0]}' width='150'><br>
    x: %{x}<br>
    y: %{y}<br>
    Cluster: %{customdata[1]}<br>
    <extra></extra>
    """
)

st.plotly_chart(fig, height=2000)

# --- DESCARGA DE GRÁFICO ---
st.download_button(
    "📥 Descargar gráfico como imagen",
    fig.to_image(format="png"),
    file_name=f"clustering_{model_name}.png",
    mime="image/png"
)

# --- INFORMACIÓN DE CLUSTERS ---
st.markdown("### 📊 Información de clusters")
cluster_sizes = df["label"].value_counts().sort_index()
st.write(cluster_sizes.rename("Nº vídeos").to_frame())

# --- HISTOGRAMA DE TAMAÑOS DE CLÚSTERES ---
import matplotlib.pyplot as plt

st.markdown("### 📊 Distribución del tamaño de los clústeres")
cluster_sizes = df["label"].value_counts()
fig, ax = plt.subplots()
cluster_sizes[cluster_sizes.index != -1].plot(kind="hist", bins=20, ax=ax)
ax.set_xlabel("Tamaño de clúster")
ax.set_ylabel("Frecuencia")
st.pyplot(fig)

# --- MÉTRICAS DE CLUSTERING ---
metricas = calcular_metricas_clustering(labels)
metricas_df = pd.DataFrame(metricas, index=[method])
st.markdown("### 📈 Métricas del clustering")
st.table(metricas_df)

# --- DESCARGA DE MÉTRICAS ---
csv = metricas_df.to_csv(index=True).encode("utf-8")
st.download_button(
    label="📥 Descargar métricas como CSV",
    data=csv,
    file_name=f"metricas_clustering_{model_name}_{method}.csv",
    mime="text/csv"
)

# --- SELECCIÓN DIRECTA DE CLUSTER ---
cluster_options = ["Todos"] + sorted(df["label"].unique())
selected_cluster = st.selectbox("🔎 Selecciona un cluster para ver sus frames", cluster_options)

# --- DESCRIPCIÓN SEMÁNTICA ---
descriptions = {}
if st.button("📌 Describir clusters semánticamente con CLIP"):
    clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32", use_safetensors=True)
    clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    clip_model.to(device)

    descriptions = describe_clusters_with_clip(
        embeddings, labels, clip_processor, clip_model, device
    )

# --- VISTA PREVIA POR CLUSTER ---
if selected_cluster == "Todos":
    st.subheader("🖼️ Vista previa de videos divididos por clusters")

    ordered_labels = df["label"].value_counts().sort_values().index

    for label in ordered_labels:
        cluster_df = df[df["label"] == label]
        desc = descriptions.get(label, "Sin descripción disponible")
        st.markdown(f"### Cluster {label} ({len(cluster_df)} videos)")
        st.markdown(f"**🧠 Descripción semántica:** *{desc}*")

        cols = st.columns(5)
        for i, (_, row) in enumerate(cluster_df.iterrows()):
            with cols[i % 5]:
                st.image(row["image_path"], caption=row["video"], use_container_width=True)

else:
    cluster_df = df[df["label"] == selected_cluster]
    desc = descriptions.get(selected_cluster, "Sin descripción disponible")
    st.subheader(f"🖼️ Vista previa de videos - Cluster {selected_cluster}")
    st.markdown(f"**🧠 Descripción semántica:** *{desc}*")

    cols = st.columns(5)
    for i, (_, row) in enumerate(cluster_df.iterrows()):
        with cols[i % 5]:
            st.image(row["image_path"], caption=row["video"], use_container_width=True)