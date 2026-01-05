# audio/recorder.py
import sounddevice as sd
import numpy as np
import soundfile as sf
from datetime import datetime
import os
from queue import Queue
import threading
from config.settings import AUDIO_SETTINGS

class AudioRecorder:
    def __init__(self, sample_rate=AUDIO_SETTINGS['SAMPLE_RATE'], channels=AUDIO_SETTINGS['CHANNELS']):
        self.sample_rate = sample_rate
        self.channels = channels
        self.recording = False
        self.audio_queue = Queue()
        self.audio_thread = None
        self.stream = None
        self.audio_data = []
        
    def callback(self, indata, frames, time, status):
        """This is called for each audio block from the sounddevice."""
        if self.recording:
            self.audio_data.append(indata.copy())
            rms = np.sqrt(np.mean(indata**2))
            db = 20 * np.log10(max(1e-10, rms))
            print(f"Audio level: {db:.1f} dB", end='\r')
    
    def record(self):
        """Start recording in a separate thread."""
        self.recording = True
        self.audio_data = []
        self.stream = sd.InputStream(
            samplerate=self.sample_rate,
            channels=self.channels,
            callback=self.callback,
            dtype='float32'
        )
        self.stream.start()
    
    def stop(self):
        """Stop recording and return the audio data."""
        if self.stream:
            self.stream.stop()
            self.stream.close()
            self.stream = None
        self.recording = False
        
        if self.audio_data:
            return np.concatenate(self.audio_data, axis=0)
        return np.array([])
    
    @staticmethod
    def save_audio(audio_data, sample_rate, filename=None):
        """Save audio data to a file."""
        os.makedirs(AUDIO_SETTINGS['RECORDINGS_DIR'], exist_ok=True)
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"recording_{timestamp}.wav"
        elif not filename.endswith('.wav'):
            filename += '.wav'
            
        filepath = os.path.join(AUDIO_SETTINGS['RECORDINGS_DIR'], filename)
        sf.write(filepath, audio_data, sample_rate, subtype='PCM_16')
        return filepath

# Global recorder instance
recorder = AudioRecorder()

def start_recording():
    """Start recording audio."""
    global recorder
    recorder = AudioRecorder()
    recorder.record()
    return recorder

def stop_and_save(recorder_instance, filename=None):
    """Stop recording and save the audio."""
    audio_data = recorder_instance.stop()
    if len(audio_data) > 0:
        return AudioRecorder.save_audio(audio_data, recorder_instance.sample_rate, filename)
    return None