import numpy as np
import sounddevice as sd
import scipy.io.wavfile as wavfile
import matplotlib.pyplot as plt
from translator import text_to_binary

# KONSTANTER
FREKVENSER = [1511, 2099]  # Två frekvenser för två bitar
PILOT_FREKVENS = 750
SAMPLINGSFREKVENS = 44100  # Hz
BIT_DURATION = 0.2  # sekunder
PILOT_DURATION = 0.1
AMPLITUD = 0.5
CHUNK_SIZE = int(SAMPLINGSFREKVENS * BIT_DURATION)
DATA_FILE = text_to_binary("text.txt")

def generera_sinusvåg(frekvens, duration):
    t = np.linspace(0, duration, int(SAMPLINGSFREKVENS * duration), endpoint=False)
    return AMPLITUD * np.sin(2 * np.pi * frekvens * t)

def skapa_moduleringsvåg(duration, frekvens):
    t = np.linspace(0, duration, int(SAMPLINGSFREKVENS * duration), endpoint=False)
    return np.abs((np.sin((frekvens * 2 * np.pi * t))))

def ease_out(duration, k=9):
    t = np.linspace(0, duration, int(SAMPLINGSFREKVENS * duration), endpoint=False)
    return np.exp(-k * t / duration)

def skapa_fsk_signal(data):
    pilot_moduleringsvåg = skapa_moduleringsvåg(PILOT_DURATION, 5)
    clear_pilot_signal = generera_sinusvåg(PILOT_FREKVENS, PILOT_DURATION)

    pilot_signal = pilot_moduleringsvåg * clear_pilot_signal * 0.2
    
    signal = np.array([])

    moduleringsvåg = skapa_moduleringsvåg(BIT_DURATION, 5)
    
    for x in range(0, len(data), 2):
        chunk = data[x:x+2]
        chunk_signal = np.zeros(CHUNK_SIZE)

        for y, bit in enumerate(chunk):
            if bit == '1':
                sinusvåg = generera_sinusvåg(FREKVENSER[y], BIT_DURATION)
                ease_out_vector = ease_out(BIT_DURATION)
                sinusvåg = sinusvåg * ease_out_vector  
                chunk_signal += sinusvåg  

        chunk_signal = chunk_signal * moduleringsvåg
        signal = np.concatenate((signal, chunk_signal))

    paus = generera_sinusvåg(100, 0.02) * 0

    signal = np.concatenate((pilot_signal, paus, signal))
    return signal

if __name__ == "__main__":
    print(f"Binär data:\n{DATA_FILE}\n")    
    fsk_signal = skapa_fsk_signal(DATA_FILE)

    output_filnamn = "signal_2frekvenser.wav"
    wavfile.write(output_filnamn, SAMPLINGSFREKVENS, (fsk_signal * 32767).astype(np.int16))
    print(f"Ljudfil sparad som {output_filnamn}\n")

    print("Spelar upp signal...")
    sd.play(fsk_signal, SAMPLINGSFREKVENS)
    plt.plot(fsk_signal)
    plt.show()
    sd.wait()
    print("Ljuduppspelning klar!")
