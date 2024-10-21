import sounddevice as sd
from scipy.io import wavfile
import numpy as np
 
SAMPLERATE = 44100
DURATION = 16
recording = None
is_recording = False
 
def record_audio(samplerate, duration):
    global recording, is_recording
    print("Recording... Will stop automatically after {} seconds.".format(duration))
    recording = sd.rec(int(duration * samplerate), samplerate=samplerate, channels=2, dtype='float64')  
    is_recording = True
    sd.wait()  
    is_recording = False
    print("Done!")
 
record_audio(SAMPLERATE, DURATION)
OUTPUT_FILE = f'signal.wav'
audio_data_int16 = np.int16(recording * 32767)
wavfile.write(OUTPUT_FILE, SAMPLERATE, audio_data_int16)
print(f"Recording saved in: {OUTPUT_FILE}")