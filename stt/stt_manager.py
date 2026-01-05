# stt/stt_manager.py
import os
import numpy as np
import soundfile as sf
import librosa
from stt.whisper_stt import transcribe
from config.settings import AUDIO_SETTINGS

def read_audio_file(file_path):
    """Read audio file and return numpy array."""
    try:
        # Try reading with soundfile first
        try:
            audio, sr = sf.read(file_path, dtype='float32')
            # Convert to mono if stereo
            if len(audio.shape) > 1:
                audio = np.mean(audio, axis=1)
            # Resample if needed
            if sr != AUDIO_SETTINGS['SAMPLE_RATE']:
                audio = librosa.resample(audio, orig_sr=sr, target_sr=AUDIO_SETTINGS['SAMPLE_RATE'])
            return audio
        except Exception as e:
            print(f"Error reading with soundfile: {e}, trying with librosa...")
            audio, sr = librosa.load(file_path, sr=AUDIO_SETTINGS['SAMPLE_RATE'], mono=True)
            return audio
    except Exception as e:
        print(f"Error reading audio file: {e}")
        raise

def get_full_transcript(audio_source, is_file=True):
    """Get full transcript from either file or numpy array."""
    try:
        if is_file:
            audio_data = read_audio_file(audio_source)
        else:
            audio_data = audio_source
            
        if len(audio_data) == 0:
            return "Error: Empty audio data"
            
        # Transcribe the audio
        return transcribe(audio_data)
    except Exception as e:
        print(f"Error in get_full_transcript: {e}")
        return f"Error: {str(e)}"