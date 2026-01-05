"""Thread-safe queue for audio data chunks."""
import queue

# Create a thread-safe queue for audio chunks
audio_queue = queue.Queue()

def clear():
    """Clear all items from the audio queue."""
    while not audio_queue.empty():
        try:
            audio_queue.get_nowait()
        except queue.Empty:
            break

def get_all():
    """Get all items from the queue and clear it.
    
    Returns:
        list: List of all audio chunks in the queue
    """
    items = []
    while not audio_queue.empty():
        try:
            items.append(audio_queue.get_nowait())
        except queue.Empty:
            break
    return items

def put(chunk):
    """Put a chunk of audio data into the queue.
    
    Args:
        chunk: Audio data chunk to add to the queue
    """
    audio_queue.put(chunk)

def is_empty():
    """Check if the queue is empty.
    
    Returns:
        bool: True if the queue is empty, False otherwise
    """
    return audio_queue.empty()
