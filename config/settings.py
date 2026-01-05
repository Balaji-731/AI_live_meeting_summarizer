# config/settings.py
from pathlib import Path
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Base directory
BASE_DIR = Path(__file__).resolve().parent.parent

# Audio settings
AUDIO_SETTINGS = {
    'SAMPLE_RATE': 16000,
    'CHANNELS': 1,
    'CHUNK_SIZE': 1024,
    'RECORDINGS_DIR': os.path.join(BASE_DIR, 'recordings')
}

# Create necessary directories
os.makedirs(AUDIO_SETTINGS['RECORDINGS_DIR'], exist_ok=True)

# Email settings (update these in .env)
EMAIL_SETTINGS = {
    'SMTP_SERVER': os.getenv('SMTP_SERVER', 'smtp.gmail.com'),
    'SMTP_PORT': int(os.getenv('SMTP_PORT', 587)),
    'EMAIL_SENDER': os.getenv('EMAIL_SENDER'),
    'EMAIL_PASSWORD': os.getenv('EMAIL_PASSWORD')
}

# Whisper settings
WHISPER_SETTINGS = {
    'MODEL_SIZE': 'base',  # base, small, medium, large
    'DEVICE': 'cuda' if os.getenv('CUDA_AVAILABLE') == 'true' else 'cpu'
}