import numpy as np
import sounddevice as sd
import scipy.io.wavfile as wavfile
import matplotlib.pyplot as plt
from translate import file_to_binary

FREQUENCIES = [730, 950, 1300, 1800, 2400, 3100, 3800, 4700]
PILOT_FREQUENCY = 520
SAMPLING_FREQUENCY = 44100
BIT_DURATION = 0.2
PILOT_DURATION = 0.1
AMPLITUDE = 0.5
CHUNK_SIZE = int(SAMPLING_FREQUENCY * BIT_DURATION)
DATA_FILE = file_to_binary("text.txt")

def generate_sine_wave(frequency, duration):
    t = np.linspace(0, duration, int(SAMPLING_FREQUENCY * duration), endpoint=False)
    return AMPLITUDE * np.sin(2 * np.pi * frequency * t)

def create_modulation_wave(duration, frequency):
    t = np.linspace(0, duration, int(SAMPLING_FREQUENCY * duration), endpoint=False)
    return np.abs((np.sin((frequency * 2 * np.pi * t))))

def ease_out(duration, k=9):
    t = np.linspace(0, duration, int(SAMPLING_FREQUENCY * duration), endpoint=False)
    return np.exp(-k * t / duration)

def create_fsk_signal(data):
    pilot_modulation_wave = create_modulation_wave(PILOT_DURATION, 5)
    clear_pilot_signal = generate_sine_wave(PILOT_FREQUENCY, PILOT_DURATION)

    pilot_signal = pilot_modulation_wave * clear_pilot_signal * 0.2
   
    signal = np.array([])

    modulation_wave = create_modulation_wave(BIT_DURATION, 2.5)
   
    for x in range(0, len(data), 8):
        chunk = data[x:x+8]
        chunk_signal = np.zeros(CHUNK_SIZE)

        for y, bit in enumerate(chunk):
            if bit == '1':
                sine_wave = generate_sine_wave(FREQUENCIES[y], BIT_DURATION)
                ease_out_vector = ease_out(BIT_DURATION)
                sine_wave = sine_wave * ease_out_vector  
                chunk_signal += sine_wave  

        chunk_signal = chunk_signal * modulation_wave
        signal = np.concatenate((signal, chunk_signal))

    pause = generate_sine_wave(100, 0.02) * 0

    signal = np.concatenate((pilot_signal, pause, signal))
    signal = signal[:10*SAMPLING_FREQUENCY]
    return signal

if __name__ == "__main__":
    print(f"Binary data:\n{DATA_FILE}\n")    
    fsk_signal = create_fsk_signal(DATA_FILE)

    output_filename = "8f_signal.wav"
    wavfile.write(output_filename, SAMPLING_FREQUENCY, (fsk_signal * 32767).astype(np.int16))
    print(f"Audio file saved as {output_filename}\n")

    print("Playing signal...")
    sd.play(fsk_signal, SAMPLING_FREQUENCY)
    plt.plot(fsk_signal)
    plt.show()
    sd.wait()
    print("Audio playback complete!")