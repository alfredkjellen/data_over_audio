import numpy as np
import sounddevice as sd
import scipy.io.wavfile as wavfile
import matplotlib.pyplot as plt
from translator import text_to_binary

# KONSTANTER
FREKVENS = 2000  # Hz
PILOT_FREKVENS = 2000
SAMPLINGSFREKVENS = 44100  # Hz
BIT_DURATION = 0.1  # sekunder
PILOT_DURATION = 0.1
AMPLITUD = 0.5
CHUNK_SIZE = int(SAMPLINGSFREKVENS * BIT_DURATION)
DATA_FILE = text_to_binary("text.txt")

def generera_sinusvåg(duration, frekvens):
    t = np.linspace(0, duration, int(SAMPLINGSFREKVENS * duration), endpoint=False)
    return AMPLITUD * np.sin(2 * np.pi * frekvens * t)

def skapa_moduleringsvåg(duration, frekvens):
    t = np.linspace(0, duration, int(SAMPLINGSFREKVENS * duration), endpoint=False)
    return np.abs(np.sin((frekvens * 2 * np.pi * t)))  # Moduleringsvågen med en period på 0.2 sekunder

def skapa_fsk_signal(data):
    pilot_moduleringsvåg = skapa_moduleringsvåg(PILOT_DURATION, 5)
    clear_pilot_signal = generera_sinusvåg(PILOT_DURATION, PILOT_FREKVENS)

    pilot_signal = pilot_moduleringsvåg * clear_pilot_signal
    
    signal = np.array([])
    moduleringsvåg = skapa_moduleringsvåg(BIT_DURATION, 5)  # Skapa moduleringsvåg för en bit
    
    for bit in data:
        if bit == '1':
            chunk_signal = generera_sinusvåg(BIT_DURATION, FREKVENS)
        else:
            chunk_signal = np.zeros(CHUNK_SIZE)
        
        # Applicera moduleringsvågen (mjuka övergångar) på varje bit
        chunk_signal = chunk_signal * moduleringsvåg
        signal = np.concatenate((signal, chunk_signal))

    silence = np.zeros(int(SAMPLINGSFREKVENS * 2))

    # Lägg till tystnad, pilotsignal och datasignal
    signal = np.concatenate((silence, pilot_signal, signal))
    return signal

if __name__ == "__main__":
    print(f"Binär data:\n{DATA_FILE}\n")    
    fsk_signal = skapa_fsk_signal(DATA_FILE)

    output_filnamn = "signal.wav"
    wavfile.write(output_filnamn, SAMPLINGSFREKVENS, (fsk_signal * 32767).astype(np.int16))
    print(f"Ljudfil sparad som {output_filnamn}\n")

    print("Spelar upp signal...")
    sd.play(fsk_signal, SAMPLINGSFREKVENS)
    plt.plot(fsk_signal)
    plt.show()
    sd.wait()
    print("Ljuduppspelning klar!")
