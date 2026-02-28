"""
Merkezi konfigürasyon dosyası.
Tüm ayarlar buradan yönetilir — modül yazarları kendi config'lerini buradan alır.
"""
import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# --- Paths ---
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
AMI_DIR = DATA_DIR / "ami"
SAMPLE_DIR = DATA_DIR / "sample"
OUTPUT_DIR = PROJECT_ROOT / "outputs"

# --- API Keys ---
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
HF_TOKEN = os.getenv("HF_TOKEN", "")

# --- Whisper (Kişi A) ---
WHISPER_MODEL_SIZE = os.getenv("WHISPER_MODEL_SIZE", "base")

# --- Redis (Shared Context) ---
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")

# --- Genel ---
SAMPLE_RATE = 16000  # Whisper ve Pyannote 16kHz bekler
