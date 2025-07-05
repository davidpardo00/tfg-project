import os, shutil, torch, cv2, clip
import numpy as np
from PIL import Image
from tqdm import tqdm
from transformers import AutoProcessor, AutoModel
from transformers import CLIPVisionModelWithProjection
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize, InterpolationMode

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

    elif model_name == "jinaclip":
        jina_model = "jinaai/jina-clip-v2"
        processor = AutoProcessor.from_pretrained(jina_model, trust_remote_code=True)
        model = AutoModel.from_pretrained(jina_model, trust_remote_code=True).to(device)
        return processor, model, "jinaclip"

    elif model_name == "clip4clip":
        clip4clip_model = "Searchium-ai/clip4clip-webvid150k"
        processor = None  # No usamos AutoProcessor aquí
        model = CLIPVisionModelWithProjection.from_pretrained(clip4clip_model).to(device).eval()
        return processor, model, "clip4clip"

    else:
        raise ValueError(f"Modelo no soportado: {model_name!r}")
    
def setup_output_directories(output_dirs):
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

def generate_jinaclip_embedding(image, processor, model, device):
    """
    Genera un embedding de imagen utilizando un modelo JinaClip.
    
    :param image: Objeto PIL Image.
    :param processor: Procesador de Hugging Face.
    :param model: Modelo JinaClip.
    :param device: Dispositivo de PyTorch ("cuda" o "cpu").
    :return: Numpy array con el embedding.
    """
    inputs = processor(images=image, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model.get_image_features(**inputs)
        # Conversion de formato para poder convertir a numpy
        outputs = outputs.to(torch.float32) 
    return outputs.cpu().numpy()

def generate_clip4clip_embedding(image, _, model, device):
    """
    Genera el embedding de una imagen usando Clip4Clip.
    :param image: Objeto PIL Image.
    :param _: Parámetro ignorado (processor no se usa aquí).
    :param model: Modelo CLIPVisionModelWithProjection.
    :param device: Dispositivo de PyTorch ("cuda" o "cpu").
    :return: Numpy array con el embedding.
    """
    transform = Compose([
        Resize(224, interpolation=InterpolationMode.BICUBIC),
        CenterCrop(224),
        lambda img: img.convert("RGB"),
        ToTensor(),
        Normalize((0.48145466, 0.4578275, 0.40821073),
                  (0.26862954, 0.26130258, 0.27577711)),
    ])

    image_tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(image_tensor)["image_embeds"]
        output = output / output.norm(dim=-1, keepdim=True)
        return output.squeeze(0).cpu().numpy()

def process_frames(video_path, model_type, preprocess_or_processor, 
                   model, device, frame_stride = 3,
                   embedding_path="outputs/embeddings/embeddings.npy"):
    """
    Procesa los frames del video original para generar embeddings.
    Se recorre el video frame a frame (o con los saltos definidos 
    por frame-_stride) usando cv2.VideoCapture.

    :param video_path: Ruta del video original.
    :param model_type: Modelo a utilizar ("clip", "siglip", etc.)
    :param embedding_path: Ruta donde se guardará el archivo numpy con los embeddings.
    :param frame_stride: Número de frames a saltar entre cada embedding (default=3).
    :return: Ruta del archivo guardado.
    """
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"El video '{video_path}' no existe.")
    os.makedirs(os.path.dirname(embedding_path), exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise Exception(f"No se pudo abrir el video '{video_path}'")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    embeddings = []
    frame_count = 0

    model_type = model_type.lower()
    print("Modelo seleccionado:", model_type)

    from tqdm import tqdm
    from PIL import Image

    # Usamos tqdm para mostrar el progreso
    current_frame = 0
    with tqdm(total=total_frames, desc="Procesando frames") as pbar:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Saltar frames según stride
            if current_frame % frame_stride != 0:
                current_frame += 1
                pbar.update(1)
                continue

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image = Image.fromarray(frame_rgb)

            if model_type == "clip":
                embedding = generate_clip_embedding(image, preprocess_or_processor, model, device)
            elif model_type == "siglip":
                embedding = generate_siglip_embedding(image, preprocess_or_processor, model, device)
            elif model_type == "jinaclip":
                embedding = generate_jinaclip_embedding(image, preprocess_or_processor, model, device)
            elif model_type == "clip4clip":
                embedding = generate_clip4clip_embedding(image, preprocess_or_processor, model, device)
            else:
                raise ValueError(f"Modelo '{model_type}' no reconocido.")

            embeddings.append(embedding)
            frame_count += 1
            current_frame += 1
            pbar.update(1)
    cap.release()

    embeddings = np.vstack(embeddings)
    np.save(embedding_path, embeddings)
    print(f"Embeddings generados y guardados en {embedding_path}. Dimensión: {embeddings.shape}. Frames procesados: {frame_count}")
    print(f"Total frames leídos: {total_frames}, Embeddings generados: {len(embeddings)}")
    return embedding_path



