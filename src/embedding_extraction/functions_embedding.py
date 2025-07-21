import os, shutil, torch, cv2, clip
import numpy as np
from PIL import Image
from tqdm import tqdm
from transformers import AutoProcessor, AutoModel, AutoModelForCausalLM
from transformers import CLIPVisionModelWithProjection
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize, InterpolationMode

def init_model(model_name: str, device):
    """
    Inicializa y devuelve el modelo, su preprocesador y el tipo de modelo.
    """
    model_name = model_name.lower()

    if model_name == "clip":
        model, preprocess = clip.load("ViT-B/32", device=device, jit=False)
        model = model.to(device)
        print(f"CLIP inicializado en {device}")
        return preprocess, model, "clip"

    elif model_name == "siglip":
        sig_name = "google/siglip-base-patch16-224"
        processor = AutoProcessor.from_pretrained(sig_name)
        model = AutoModel.from_pretrained(sig_name)
        model.eval().to(device)
        print(f"SigLIP inicializado en {device}")
        return processor, model, "siglip"

    elif model_name == "jinaclip":
        jina_model = "jinaai/jina-clip-v2"
        processor = AutoProcessor.from_pretrained(jina_model, trust_remote_code=True)
        model = AutoModel.from_pretrained(jina_model, trust_remote_code=True)
        model = model.to(device)
        print(f"JinaCLIP inicializado en {device}")
        return processor, model, "jinaclip"

    elif model_name == "clip4clip":
        clip4clip_model = "Searchium-ai/clip4clip-webvid150k"
        model = CLIPVisionModelWithProjection.from_pretrained(clip4clip_model).to(device).eval()
        print(f"CLIP4CLIP inicializado en {device}")
        return None, model, "clip4clip"

    elif model_name == "openclip":
        openclip_model = "laion/CLIP-ViT-B-32-laion2B-s34B-b79K"
        processor = AutoProcessor.from_pretrained(openclip_model)
        model = AutoModel.from_pretrained(openclip_model).to(device).eval()
        print(f"OpenCLIP inicializado en {device}")
        return processor, model, "openclip"

    elif model_name == "git":
        processor = AutoProcessor.from_pretrained("microsoft/git-base")
        model = AutoModel.from_pretrained("microsoft/git-base")
        model = model.to(device)
        print("GIT inicializado en", device)
        return processor, model, "git"

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

def generate_openclip_embedding(frame, processor, model, device):
    """
    Extrae un embedding de una imagen usando OpenCLIP.
    """
    inputs = processor(images=frame, return_tensors="pt").to(device)

    with torch.no_grad():
        outputs = model.get_image_features(**inputs)
        embedding = outputs.cpu().numpy().squeeze()

    return embedding

def generate_git_embedding(frame, processor, model, device):
    """
    Genera un embedding de una imagen usando el modelo GIT.
    """
    inputs = processor(images=frame, return_tensors="pt").to(device)

    with torch.no_grad():
        vision_outputs = model.image_encoder(**inputs)
        last_hidden_state = vision_outputs.last_hidden_state 
        embedding = last_hidden_state.mean(dim=1).squeeze().cpu().numpy()

    return embedding

def process_frames(video_path, model_type, preprocess_or_processor,
                   model, device, frame_stride=3,
                   embedding_path=None):
    """
    Procesa los frames del video original para generar embeddings.
    Recorre el video frame a frame (con saltos definidos por frame_stride) usando cv2.VideoCapture.

    :param video_path: Ruta del video original.
    :param model_type: Modelo a utilizar ("clip", "siglip", etc.)
    :param preprocess_or_processor: Preprocesador del modelo.
    :param model: Modelo cargado.
    :param device: Dispositivo de PyTorch ("cuda" o "cpu").
    :param frame_stride: Número de frames a saltar entre cada embedding.
    :param embedding_path: Ruta donde guardar los embeddings (.npy), o None para no guardar.
    :return: np.ndarray con todos los embeddings generados.
    """
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"El video '{video_path}' no existe.")

    if embedding_path is not None:
        os.makedirs(os.path.dirname(embedding_path), exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise Exception(f"No se pudo abrir el video '{video_path}'")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    embeddings = []
    current_frame = 0
    frame_count = 0

    model_type = model_type.lower()
    print(f"Procesando: {os.path.basename(video_path)} con modelo: {model_type}")

    with tqdm(total=total_frames, desc="Procesando frames", leave=False) as pbar:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

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
            elif model_type == "git":
                embedding = generate_git_embedding(image, preprocess_or_processor, model, device)
            elif model_type == "openclip":
                embedding = generate_openclip_embedding(image, preprocess_or_processor, model, device)
            else:
                raise ValueError(f"Modelo '{model_type}' no reconocido.")

            embeddings.append(embedding)
            frame_count += 1
            current_frame += 1
            pbar.update(1)

    cap.release()

    embeddings = np.vstack(embeddings)

    if embedding_path is not None:
        np.save(embedding_path, embeddings)
        print(f"Guardado en {embedding_path}. Embeddings shape: {embeddings.shape}")

    return embedding

def save_frames_from_video(video_path, output_folder, frame_stride=3):
    """
    Guarda frames extraídos de un video cada 'frame_stride' frames.
    Devuelve una lista con las rutas de los frames guardados.
    """
    os.makedirs(output_folder, exist_ok=True)
    cap = cv2.VideoCapture(video_path)

    frame_paths = []
    current_frame = 0
    saved_index = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if current_frame % frame_stride == 0:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image = Image.fromarray(frame_rgb)
            frame_path = os.path.join(output_folder, f"frame_{saved_index:04d}.jpg")
            image.save(frame_path)
            frame_paths.append(frame_path)
            saved_index += 1

        current_frame += 1

    cap.release()
    return frame_paths

def save_central_frame_from_video(video_path, output_path):
    """
    Extrae y guarda el frame central de un video como imagen JPG.
    
    :param video_path: Ruta al archivo de video.
    :param output_path: Ruta donde se guardará la imagen del frame central.
    """
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        raise FileNotFoundError(f"No se pudo abrir el video: {video_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    middle_frame_index = total_frames // 2

    cap.set(cv2.CAP_PROP_POS_FRAMES, middle_frame_index)
    ret, frame = cap.read()
    cap.release()

    if not ret:
        raise ValueError(f"No se pudo leer el frame {middle_frame_index} del video: {video_path}")

    # Convertir a RGB y guardar
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    image = Image.fromarray(frame_rgb)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    image.save(output_path, format="JPEG")

def save_central_frames_from_folder(video_folder, output_folder):
    """
    Extrae el frame central de todos los videos en una carpeta.
    
    :param video_folder: Carpeta con videos .mp4
    :param output_folder: Carpeta donde se guardarán las imágenes.
    """
    os.makedirs(output_folder, exist_ok=True)
    video_files = [f for f in os.listdir(video_folder) if f.lower().endswith(".mp4")]

    print(f"Procesando {len(video_files)} vídeos desde '{video_folder}'...")
    
    for filename in tqdm(video_files, desc="Extrayendo frame central", unit="video"):
        video_path = os.path.join(video_folder, filename)
        image_name = os.path.splitext(filename)[0] + ".jpg"
        output_path = os.path.join(output_folder, image_name)
        try:
            save_central_frame_from_video(video_path, output_path)
        except Exception as e:
            tqdm.write(f"Error en {filename}: {e}")

