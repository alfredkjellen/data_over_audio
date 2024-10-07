import numpy as np
import sounddevice as sd
import scipy.io.wavfile as wavfile
import matplotlib.pyplot as plt
from translator import text_to_binary

# KONSTANTER
FREKVENS = 2000  # Hz
SAMPLINGSFREKVENS = 44100  # Hz
BIT_DURATION = 0.1  # sekunder
AMPLITUD = 0.5
CHUNK_SIZE = int(SAMPLINGSFREKVENS * BIT_DURATION)
DATA_FILE = text_to_binary("test_text.txt")

def generera_sinusvåg(duration):
    t = np.linspace(0, duration, int(SAMPLINGSFREKVENS * duration), endpoint=False)
    return AMPLITUD * np.sin(2 * np.pi * FREKVENS * t)

def skapa_fsk_signal(data):
    signal = np.array([])
    for bit in data:
        if bit == '1':
            chunk_signal = generera_sinusvåg(BIT_DURATION)
        else:
            chunk_signal = np.zeros(CHUNK_SIZE)
        signal = np.concatenate((signal, chunk_signal))
    plt.plot(signal)
    plt.show()
    return signal

if __name__ == "__main__":
    print(f"Original data:\n{DATA_FILE}\n")
    
    full_data = f'{DATA_FILE}'
    print(f"Data med pilotsekvens:\n{full_data}\n")
    
    fsk_signal = skapa_fsk_signal(full_data)

    output_filnamn = "signal.wav"
    wavfile.write(output_filnamn, SAMPLINGSFREKVENS, (fsk_signal * 32767).astype(np.int16))
    print(f"Ljudfil sparad som {output_filnamn}\n")

    print("Spelar upp signal...")
    sd.play(fsk_signal, SAMPLINGSFREKVENS)
    sd.wait()
    print("Ljuduppspelning klar!")