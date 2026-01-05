# stt/whisper_stt.py
import os
import torch
import numpy as np
import logging
from config.settings import WHISPER_SETTINGS

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global model variable
_model = None

def load_model():
    """Load Whisper model with caching."""
    global _model
    if _model is None:
        import whisper
        logger.info(f"Loading Whisper {WHISPER_SETTINGS['MODEL_SIZE']} model...")
        _model = whisper.load_model(WHISPER_SETTINGS['MODEL_SIZE'], device=WHISPER_SETTINGS['DEVICE'])
        logger.info("Model loaded successfully")
    return _model

def transcribe(audio_np, language="en"):
    """Transcribe audio using Whisper."""
    try:
        logger.info("Starting transcription...")
        model = load_model()
        
        # Convert audio to float32 if needed
        if audio_np.dtype != np.float32:
            audio_np = audio_np.astype(np.float32)
            
        # Transcribe
        result = model.transcribe(
            audio_np,
            language=language,
            fp16=torch.cuda.is_available(),
            temperature=0.2,
            best_of=5,
            beam_size=5,
            initial_prompt="This is a spoken recording of a meeting or conversation."
        )
        
        transcript = result.get("text", "").strip()
        logger.info(f"Transcription complete. Length: {len(transcript)} characters")
        
        if not transcript:
            logger.warning("Whisper returned an empty transcription")
            
        return transcript
        
    except Exception as e:
        logger.error(f"Error in transcription: {e}")
        return ""