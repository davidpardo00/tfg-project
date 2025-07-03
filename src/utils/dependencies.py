import subprocess
import sys
import importlib

# Diccionario con: clave = nombre a importar, valor = cómo instalarlo con pip
required_packages = {
    "torch": "torch",
    "torchvision": "torchvision",
    "sentencepiece": "sentencepiece",
    "cv2": "opencv-python",
    "PIL": "Pillow",
    "tqdm": "tqdm",
    "transformers": "transformers",
    "scenedetect": "scenedetect",
    "numpy": "numpy",
    "matplotlib": "matplotlib",
    "seaborn": "seaborn",
    "umap": "umap-learn",
    "hdbscan": "hdbscan",
    "clip": "git+https://github.com/openai/CLIP.git",
    "whisper": "openai-whisper",
    "sklearn": "scikit-learn",
    "pandas": "pandas",
    "scipy": "scipy",
    "google.protobuf": "protobuf",
    "timm": "timm",
    "einops": "einops",
    "safetensors": "safetensors"
}

def install(package_name: str):
    print(f"⏳ Instalando: {package_name}")
    subprocess.check_call([sys.executable, "-m", "pip", "install", package_name])

def check_and_install_packages():
    for import_name, install_name in required_packages.items():
        try:
            importlib.import_module(import_name)
        except ImportError:
            install(install_name)

if __name__ == "__main__":
    check_and_install_packages()
    print("✅ Todas las dependencias están instaladas correctamente.")
