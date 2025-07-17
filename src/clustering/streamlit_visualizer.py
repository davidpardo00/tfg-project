import os, sys, glob
import numpy as np
import streamlit as st
import plotly.express as px
import pandas as pd
from PIL import Image

# Agregar carpeta src al path
SRC_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if SRC_DIR not in sys.path:
    sys.path.append(SRC_DIR)

from clustering.functions_clustering import *

# --- CONFIGURACIÓN GLOBAL ---
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
EMBEDDING_DIR = os.path.join(ROOT_DIR, "outputs", "embeddings")
FRAME_DIR = os.path.join(ROOT_DIR, "outputs", "frames_cluster")

# --- CARGAR EMBEDDINGS ---
# Buscar automáticamente el archivo de embeddings más reciente
embedding_files = glob.glob(os.path.join(EMBEDDING_DIR, "mean_embeddings_*.npy"))
if not embedding_files:
    raise FileNotFoundError("❌ No se encontró ningún archivo 'mean_embeddings_*.npy' en la carpeta de embeddings.")

# Elegir el primero encontrado
embedding_file = sorted(embedding_files)[-1]

# Extraer el nombre del modelo desde el nombre del archivo
model_name = os.path.basename(embedding_file).replace("mean_embeddings_", "").replace(".npy", "")

# Buscar el archivo correspondiente de nombres de video
video_names_file = os.path.join(EMBEDDING_DIR, f"video_names_{model_name}.npy")
st.sidebar.markdown(f"📌 <b>Modelo cargado:</b> <code>{model_name}</code>", unsafe_allow_html=True)

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
    radius = st.sidebar.slider("Radio (radius)", 0.1, 5.0, 0.4, 0.1)
    minPts = st.sidebar.slider("Mínimo puntos (minPts)", 1, 20, 5)
else:
    min_cluster_size = st.sidebar.slider("Tamaño mínimo del cluster", 2, 30, 5)
    min_samples = st.sidebar.slider("Muestras mínimas", 1, 10, 1)

# Reducción params
embeddings_2d = reduce_dimensionality(embeddings, method=reduction, n_components=2, random_state=42)

# --- CLUSTERING ---
if method == "CLASSIX":
    labels = cluster_embeddings_CLASSIX(embeddings, radius=radius, minPts=minPts)
elif method == "HDBSCAN":
    labels = cluster_embeddings_HDBSCAN(embeddings, min_cluster_size=min_cluster_size, min_samples=min_samples)

# --- DATAFRAME PARA PLOTLY ---
df = pd.DataFrame(embeddings_2d, columns=["x", "y"])
df["label"] = labels
df["video"] = video_names
df["image_path"] = df["video"].apply(lambda x: os.path.join(FRAME_DIR, f"{x}.jpg"))

# --- FILTROS OPCIONALES ---
cluster_filter = st.sidebar.multiselect("Filtrar clusters", options=sorted(df["label"].unique()))
if cluster_filter:
    df = df[df["label"].isin(cluster_filter)]

# --- GRAFICA INTERACTIVA ---
st.subheader("Embeddings agrupados por clustering")
fig = px.scatter(
    df,
    x="x",
    y="y",
    color=df["label"].astype(str),
    hover_name="video",
    custom_data=["image_path"],
    color_discrete_sequence=px.colors.qualitative.Set1,
)

fig.update_traces(
    hovertemplate="""
    <b>%{hovertext}</b><br><br>
    <img src='%{customdata[0]}' width='150'><br>
    x: %{x}<br>
    y: %{y}<br>
    Cluster: %{marker.color}<br>
    <extra></extra>
    """
)

st.plotly_chart(fig, height=2000)

# --- INFORMACIÓN DE CLUSTERS ---
st.markdown("### 📊 Información de clusters")
cluster_sizes = df["label"].value_counts().sort_index()
st.write(cluster_sizes.rename("Nº vídeos").to_frame())


# --- VISTA PREVIA POR CLUSTER ---
st.subheader("🖼️ Vista previa de videos divididos por clusters")

for label in df["label"].value_counts().index:
    cluster_df = df[df["label"] == label]
    st.markdown(f"### Cluster {label} ({len(cluster_df)} videos)")
    cols = st.columns(5)
    for i, (_, row) in enumerate(cluster_df.iterrows()):
        with cols[i % 5]:
            st.image(row["image_path"], caption=row["video"], width=100)

# --- DESCARGA DE GRÁFICO ---
st.download_button(
    "📥 Descargar gráfico como imagen",
    fig.to_image(format="png"),
    file_name="clustering.png",
    mime="image/png"
)



