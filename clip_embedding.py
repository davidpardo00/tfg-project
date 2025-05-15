import shutil, torch, clip
from PIL import Image
import os, cv2
import numpy as np
from tqdm import tqdm

# Forzar CUDA incluso si is_available() falla y declarar variables globales
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = None
preprocess = None

def init_clip():
    """
    Inicializa el modelo CLIP. Esta función se ejecuta automáticamente al
    importar el módulo, pero también se puede llamar explícitamente.
    """
    global model, preprocess
    model, preprocess = clip.load("ViT-L/14", device=device, jit=False)
    model = model.to(device)
    print("CLIP inicializado en", device)

def setup_output_directories(output_dirs=["plots", "embeddings"]):
    """
    Elimina el contenido de las carpetas especificadas salvo los archivos .gitkeep
    y las vuelve a crear vacías.
    
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

def generate_clip_embedding(image):
    """
    Genera el embedding de una imagen usando CLIP.
    
    :param image: Objeto PIL Image.
    :return: Numpy array con el embedding.
    """
    input_image = preprocess(image).unsqueeze(0).to(device)
    with torch.no_grad():
        embedding = model.encode_image(input_image).cpu().numpy()
    return embedding

def process_frames(video_path, save_path="embeddings/embeddings.npy"):
    """
    Procesa todos los frames del video original para generar embeddings.
    Se recorre el video frame a frame usando cv2.VideoCapture.
    
    :param video_path: Ruta del video original.
    :param save_path: Ruta donde se guardará el archivo numpy con los embeddings.
    :return: Ruta del archivo guardado.
    """
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"El video '{video_path}' no existe.")
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise Exception(f"No se pudo abrir el video '{video_path}'")
    
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    embeddings = []
    frame_count = 0
    
    # Usamos tqdm para mostrar el progreso
    with tqdm(total=total_frames, desc="Procesando frames") as pbar:
        while True:
            ret, frame = cap.read()
            if not ret:
                break  
            # Convertir el frame de BGR (formato OpenCV) a RGB y a un objeto PIL Image
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image = Image.fromarray(frame_rgb)
            embedding = generate_clip_embedding(image)
            embeddings.append(embedding)
            frame_count += 1
            pbar.update(1)
    
    cap.release()
    
    embeddings = np.vstack(embeddings)
    np.save(save_path, embeddings)
    print(f"Embeddings generados y guardados en {save_path}. Dimensión: {embeddings.shape}. Frames procesados: {frame_count}")
    return save_path


