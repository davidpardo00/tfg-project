import shutil, torch, clip
from PIL import Image
import os
import numpy as np

# Forzar CUDA incluso si is_available() falla
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Cargar modelo CLIP
model, preprocess = clip.load("ViT-L/14", device=device, jit=False)
model = model.to(device)
 
def setup_output_directories(output_dirs=["plots", "embeddings"]):
    """
    Elimina el contenido de las carpetas especificadas salvo 
    los archivos .gitkeep y las vuelve a crear vacías.
    
    :param output_dirs: Lista de rutas de las carpetas a limpiar.
    """
    for directory in output_dirs:
        if os.path.exists(directory):
            for item in os.listdir(directory):
                item_path = os.path.join(directory, item)
                if os.path.isfile(item_path) and item != '.gitkeep':
                    os.remove(item_path)
                elif os.path.isdir(item_path):
                    shutil.rmtree(item_path)
        else:
            os.makedirs(directory)
    print("Carpetas de salida limpias y listas para guardar resultados.")

def generate_clip_embedding(image_path):
    """ Genera el embedding de una imagen usando CLIP. """
    image = preprocess(Image.open(image_path)).unsqueeze(0).to(device)
    with torch.no_grad():
        embedding = model.encode_image(image).cpu().numpy()
    return embedding

def process_frames(frame_dir="Scene_detection/images_scenes", save_path="embeddings/embeddings.npy"):
    """ Procesa todos los frames en un directorio y guarda sus embeddings. """
    if not os.path.exists(frame_dir):
        raise FileNotFoundError(f"El directorio {frame_dir} no existe.")
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    frame_files = sorted([f for f in os.listdir(frame_dir) if f.endswith(".jpg")])
    embeddings = []

    for frame in frame_files:
        frame_path = os.path.join(frame_dir, frame)
        embedding = generate_clip_embedding(frame_path)
        embeddings.append(embedding)

    embeddings = np.vstack(embeddings)
    np.save(save_path, embeddings)
    print(f"Embeddings generados y guardados en {save_path}. Dimensión: {embeddings.shape}")
    return save_path  # Retorna la ruta para que otros scripts lo usen

if __name__ == "__main__":
    print("Utilizando", device)
    process_frames()
