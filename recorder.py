import sounddevice as sd
from scipy.io import wavfile
import numpy as np

SAMPLERATE = 44100
DURATION = 10  # Längden på inspelningen i sekunder
OUTPUT_FILE = 'recording.wav' 

def record_audio(duration, samplerate):
    print("Recording...")

    recording = sd.rec(int(duration * samplerate), samplerate=samplerate, channels=2)
    sd.wait()  
    print("Done!")
    return recording

audio_data = record_audio(DURATION, SAMPLERATE)
audio_data_int16 = np.int16(audio_data * 32767)
wavfile.write(OUTPUT_FILE, SAMPLERATE, audio_data_int16)
print(f"Recording saved in: {OUTPUT_FILE}")