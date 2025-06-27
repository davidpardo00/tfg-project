import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os

def plot_umap(embeddings_2d, save_path="plots/umap.png"):
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

def plot_clusters(embeddings_2d, labels, save_path="plots/clusters.png"):
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
