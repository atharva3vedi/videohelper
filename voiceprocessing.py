import pyaudio
import wave
import os
import tempfile
from groq import Groq
from pydub import AudioSegment
import io
from queue import Queue

# Initialize the Groq client
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

def record_audio(stream, frames, audio):
    """Record audio from the microphone."""
    data = stream.read(1024)
    frames.append(data)

def process_voice_input(text_queue):
    """Process voice input and convert to text using Groq API."""
    CHUNK = 1024
    FORMAT = pyaudio.paInt16
    CHANNELS = 1
    RATE = 16000

    p = pyaudio.PyAudio()
    frames = []

    stream = p.open(format=FORMAT,
                    channels=CHANNELS,
                    rate=RATE,
                    input=True,
                    frames_per_buffer=CHUNK)

    print("Recording...")

    try:
        while True:
            record_audio(stream, frames, p)
            
            # Process audio every 5 seconds
            if len(frames) >= int(RATE / CHUNK * 5):
                audio_data = b''.join(frames)
                audio_segment = AudioSegment(
                    data=audio_data,
                    sample_width=p.get_sample_size(FORMAT),
                    frame_rate=RATE,
                    channels=CHANNELS
                )

                # Convert to WAV
                buffer = io.BytesIO()
                audio_segment.export(buffer, format="wav")
                buffer.seek(0)

                # Transcribe using Groq API
                transcription = client.audio.transcriptions.create(
                    file=("audio.wav", buffer),
                    model="whisper-large-v3-turbo"
                )

                # Put the transcribed text in the queue
                text_queue.put(transcription.text)

                # Clear frames for the next chunk
                frames = []

    except KeyboardInterrupt:
        print("Stopped recording.")
    finally:
        stream.stop_stream()
        stream.close()
        p.terminate()

if __name__ == "__main__":
    def print_transcription(text):
        print(f"Transcribed: {text}")

    test_queue = Queue()
    process_voice_input(test_queue)
    while not test_queue.empty():
        print(test_queue.get())