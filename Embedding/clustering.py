import hdbscan, umap
import seaborn as sns
import matplotlib.pyplot as plt
from clip_embedding import *

# Clustering
clusterer = hdbscan.HDBSCAN(min_cluster_size=2, min_samples=1, metric='euclidean')
labels = clusterer.fit_predict(embeddings_2d)

# Visualización de clusters
plt.figure(figsize=(10, 6))
palette = sns.color_palette("husl", np.unique(labels).size)
sns.scatterplot(x=embeddings_2d[:, 0], y=embeddings_2d[:, 1], hue=labels, palette=palette, s=100, edgecolor="black")

# Etiquetas de los puntos
for i, txt in enumerate(range(len(embeddings))):
    plt.annotate(txt, (embeddings_2d[i, 0], embeddings_2d[i, 1]), fontsize=12, color="black")

plt.title("Clustering de Escenas con HDBSCAN")
plt.xlabel("UMAP Dim 1")
plt.ylabel("UMAP Dim 2")
plt.legend(title="Clusters", loc="best")
plt.show()


