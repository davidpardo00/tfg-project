import torch, clip
from PIL import Image
import cv2, glob, umap
import numpy as np
import matplotlib.pyplot as plt

# Cargar modelo CLIP
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

def get_embedding(video_path):
    """ Extrae un frame del video y genera su embedding con CLIP """
    cap = cv2.VideoCapture(video_path)
    success, frame = cap.read()
    cap.release()
    
    if not success:
        print(f"No se pudo leer {video_path}")
        return None

    # Convertir frame a formato PIL y preprocesarlo para CLIP
    image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    image = preprocess(image).unsqueeze(0).to(device)

    # Obtener embedding
    with torch.no_grad():
        embedding = model.encode_image(image).cpu().numpy()

    return embedding

# Obtener embeddings de todas las escenas
video_files = sorted(glob.glob("Scene_detection/clips_video/*.mp4"))  # Buscar los archivos de video
embeddings = []
for v in video_files:
    embedding = get_embedding(v)
    if embedding is not None:
        embeddings.append(embedding)

# Convertir a numpy array
embeddings = np.vstack(embeddings)
np.save("Embedding/embeddings.npy", embeddings)  # Guardar embeddings para futuras pruebas

print("Embeddings generados con éxito. Dimensión:", embeddings.shape)

# Reducir a 2D
umap_reducer = umap.UMAP(n_components=2, random_state=42)
embeddings_2d = umap_reducer.fit_transform(embeddings)

# Visualizar
plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], s=100, edgecolor="black")
plt.title("Embeddings de escenas en 2D (UMAP)")
plt.xlabel("UMAP Dim 1")
plt.ylabel("UMAP Dim 2")
plt.show()
