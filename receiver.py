import numpy as np
from scipy.io import wavfile
from scipy.fft import fft
import matplotlib.pyplot as plt
from translator import binary_to_text

# KONSTANTER
FREKVENS = 2000  
SAMPLE_RATE = 44100  
BIT_DURATION = 0.1  
CHUNK_SIZE = int(SAMPLE_RATE * BIT_DURATION)
AUDIO_FILE = "signal.wav"

def detect_frequency(chunk):
    fft_result = fft(chunk)
    magnitude = np.abs(fft_result[:len(chunk)//2])
    freq_bins = np.fft.fftfreq(len(chunk), 1/SAMPLE_RATE)[:len(chunk)//2]
    
    index = np.argmin(np.abs(freq_bins - FREKVENS))
    threshold = np.mean(magnitude) * 3
    return 1 if magnitude[index] > threshold else 0

def decode_fsk_signal(filename):
    print(f"Läser in ljudfilen: {filename}\n")
    sample_rate, audio_data = wavfile.read(filename)
    print(f"Samplingsfrekvens: {sample_rate} Hz")
    print(f"Ljuddatans längd: {len(audio_data)} samples")
    
    binary_data = []
    
    for x in range(0, len(audio_data), CHUNK_SIZE):
        chunk = audio_data[x:x + CHUNK_SIZE]
        detected = detect_frequency(chunk)
        binary_data.append(detected)
    
    return binary_data, audio_data, sample_rate

def plot_time_domain(audio_data, sample_rate):
    time = np.linspace(0, len(audio_data) / sample_rate, num=len(audio_data))
    plt.plot(time, audio_data)
    plt.show()

def plot_frequency_domain(audio_data, sample_rate):
    fft_result = fft(audio_data)
    magnitude = np.abs(fft_result[:len(audio_data)//2])
    freq_bins = np.fft.fftfreq(len(audio_data), 1/sample_rate)[:len(audio_data)//2]

    plt.plot(freq_bins, magnitude)
    plt.xlim(0, 5000) 
    plt.show()

if __name__ == "__main__":
    # Dekoda signalen och få tillbaka audio_data
    binary_data, audio_data, sample_rate = decode_fsk_signal(AUDIO_FILE)
    
    if binary_data:
        print("\nAvkodad binär data:")
        print(''.join(map(str, binary_data)))
        
        decoded_text = binary_to_text(binary_data)
        print("\nAvkodad text:")
        print(decoded_text)
    else:
        print("Ingen data kunde avkodas")
    
    plot_time_domain(audio_data, sample_rate)
    plot_frequency_domain(audio_data, sample_rate)