import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os, sys, base64
import plotly.express as px
import pandas as pd

def plot_umap_interactive(embeddings_2d, image_folder, labels, save_path=None):
    """
    Visualiza los embeddings 2D con clusters y miniaturas interactivas al pasar el ratón.
    """

    image_files = sorted([f for f in os.listdir(image_folder) if f.endswith(".jpg")])
    image_paths = [os.path.join(image_folder, f) for f in image_files]

    if len(image_paths) != len(embeddings_2d):
        raise ValueError(f"Se esperaban {len(embeddings_2d)} imágenes pero se encontraron {len(image_paths)}")

    # Relative paths para que se sirvan desde el servidor local
    rel_paths = [os.path.basename(path) for path in image_paths]

    df = pd.DataFrame({
        "x": embeddings_2d[:, 0],
        "y": embeddings_2d[:, 1],
        "image": rel_paths,
        "cluster": labels
    })

    fig = px.scatter(
        df, x="x", y="y", color=df["cluster"].astype(str),
        hover_data={"image": False, "cluster": True},
        custom_data=["image"]
    )

    # Personalizar tooltip para que muestre la miniatura de la imagen
    fig.update_traces(
        marker=dict(size=8, opacity=0.8),
        hovertemplate="<b>Cluster: %{marker.color}</b><br><img src='%{customdata[0]}' width='100'><extra></extra>"
    )

    fig.update_layout(title="Visualización UMAP Interactiva con Miniaturas")

    if save_path:
        fig.write_html(save_path, full_html=True)
        print(f"Visualización interactiva guardada en {save_path}")
    else:
        fig.show()

def plot_umap(embeddings_2d, save_path="outputs/plots/umap.png"):
    """ Genera un scatter plot de los embeddings reducidos con UMAP. """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    plt.figure(figsize=(8, 6))
    plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], s=100, edgecolor="black")
    plt.title("Embeddings de embeddings en 2D (UMAP)")
    plt.xlabel("UMAP Dim 1")
    plt.ylabel("UMAP Dim 2")
    plt.savefig(save_path)
    plt.close()
    print(f"UMAP guardado en {save_path}")

def plot_clusters(embeddings_2d, labels, save_path="outputs/plots/clusters.png"):
    """ Visualiza los clusters obtenidos. """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    plt.figure(figsize=(10, 6))
    palette = sns.color_palette("husl", np.unique(labels).size)
    sns.scatterplot(x=embeddings_2d[:, 0], y=embeddings_2d[:, 1], hue=labels, palette=palette, s=100, edgecolor="black")
    plt.title("Clustering de embeddings")
    plt.xlabel("UMAP Dim 1")
    plt.ylabel("UMAP Dim 2")
    plt.legend(title="Clusters", loc="best", bbox_to_anchor=(1.05, 1), borderaxespad=0.)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"Clustering guardado en {save_path}")
