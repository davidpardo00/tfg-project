import os, shutil, torch, cv2, clip
import numpy as np
from PIL import Image
from tqdm import tqdm
from transformers import AutoProcessor, AutoModel

def init_model(model_name: str, device):
    """
    Inicializa y devuelve el modelo, su preprocesador y el nombre del modelo.
    """
    model_name = model_name.lower()
    
    if model_name == "clip":
        model, preprocess = clip.load("ViT-L/14", device=device, jit=False)
        model = model.to(device)
        print(f"CLIP inicializado en {device}")
        return preprocess, model, "clip"

    elif model_name == "siglip":
        sig_name = "google/siglip-base-patch16-224"
        processor = AutoProcessor.from_pretrained(sig_name)
        model = AutoModel.from_pretrained(sig_name)
        model.eval()
        model.to(device)
        print(f"SigLIP inicializado en {device}")
        return processor, model, "siglip"

    else:
        raise ValueError(f"Modelo no soportado: {model_name!r}")
    
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

def generate_clip_embedding(image, preprocess, model, device):
    """
    Genera el embedding de una imagen usando CLIP.

    :param image: Objeto PIL Image.
    :param preprocess: Preprocesador de CLIP.
    :param model: Modelo CLIP cargado.
    :param device: Dispositivo ("cpu" o "cuda").
    :return: Numpy array 1D con el embedding de la imagen.
    """

    input_image = preprocess(image).unsqueeze(0).to(device)
    with torch.no_grad():
        embedding = model.encode_image(input_image).cpu().numpy()
    return embedding

def generate_siglip_embedding(image, processor, model, device):
    """
    Genera el embedding de una imagen usando SigLIP.

    :param image: Objeto PIL Image.
    :param processor: Procesador de Hugging Face.
    :param model: Modelo SigLIP.
    :param device: Dispositivo de PyTorch ("cuda" o "cpu").
    :return: Numpy array con el embedding.
    """
    inputs = processor(images=image, text=["image embedding"], return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)
        embedding = outputs.image_embeds.squeeze(0).cpu().numpy()

    return embedding

def process_frames(video_path, model_type, preprocess_or_processor, model, device, save_path="outputs/embeddings/embeddings.npy"):
    """
    Procesa todos los frames del video original para generar embeddings.
    Se recorre el video frame a frame usando cv2.VideoCapture.
    
    :param video_path: Ruta del video original.
    :param modelo: Modelo a utilizar ("CLIP" o "SigLIP").
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
    
    model_type = model_type.lower()
    print("Modelo seleccionado:", model_type)
    
    # Usamos tqdm para mostrar el progreso
    with tqdm(total=total_frames, desc="Procesando frames") as pbar:
        while True:
            ret, frame = cap.read()
            if not ret:
                break  
            # Convertir el frame de BGR (formato OpenCV) a RGB y a un objeto PIL Image
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image = Image.fromarray(frame_rgb)
            
            if model_type == "clip":
                embedding = generate_clip_embedding(image, preprocess_or_processor, model, device)
            elif model_type == "siglip":
                embedding = generate_siglip_embedding(image, preprocess_or_processor, model, device)
            
            embeddings.append(embedding)
            frame_count += 1
            pbar.update(1)
    
    cap.release()
    
    embeddings = np.vstack(embeddings)
    np.save(save_path, embeddings)
    print(f"Embeddings generados y guardados en {save_path}. Dimensión: {embeddings.shape}. Frames procesados: {frame_count}")
    return save_path


