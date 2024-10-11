import numpy as np
from scipy.io import wavfile
from scipy.fft import fft
import matplotlib.pyplot as plt
from pathlib import Path
from translator import binary_to_text

# KONSTANTER
FREQUENCY = 2000
PILOT_FREQUENCY = 1500
SAMPLE_RATE = 44100
BIT_DURATION = 0.1
PILOT_DURATION = 0.1
CHUNK_SIZE = int(SAMPLE_RATE * BIT_DURATION)
PILOT_CHUNK_SIZE = int(SAMPLE_RATE * PILOT_DURATION)
PILOT_SEARCH_STEP = 50
AUDIO_FOLDER = Path("1500_hz")

# Thresholds
ABSOLUTE_MAGNITUDE_THRESHOLD = 75000
RELATIVE_MAGNITUDE_THRESHOLD = 1.5

def get_frequency_magnitude(chunk, freq):
    fft_result = fft(chunk)
    magnitude = np.abs(fft_result[:len(chunk)//2])
    freq_bins = np.fft.fftfreq(len(chunk), 1/SAMPLE_RATE)[:len(chunk)//2]
    index = np.argmin(np.abs(freq_bins - freq))
    return magnitude[index]

def find_pilot_signal(audio_data):
    max_magnitude = 0
    pilot_start = 0
    
    for i in range(0, len(audio_data) - PILOT_CHUNK_SIZE, PILOT_SEARCH_STEP):
        chunk = audio_data[i:i + PILOT_CHUNK_SIZE]
        magnitude = get_frequency_magnitude(chunk, PILOT_FREQUENCY)
        
        if magnitude > max_magnitude:
            max_magnitude = magnitude
            pilot_start = i
    
    print(f'Pilot signal starts at sample {pilot_start}')
    return pilot_start

def decode_fsk_signal(filename):
    print(f"Reading audio file: {filename}\n")
    sample_rate, audio_data = wavfile.read(filename)
    print(f"Sample rate: {sample_rate} Hz")
    print(f"Audio data length: {len(audio_data)} samples")
    
    if len(audio_data.shape) == 2:
        print("Audio is stereo, converting to mono...")
        audio_data = np.mean(audio_data, axis=1).astype(audio_data.dtype)

    start_pos = find_pilot_signal(audio_data) + PILOT_CHUNK_SIZE
    
    if start_pos is None:
        print("Pilot signal not found")
        return None, audio_data, sample_rate

    print(f'Data signal starts at sample {start_pos}')
    
    return audio_data, sample_rate


def plot_magnitude_over_time(audio_data, sample_rate, pilot_freq, data_freq, start_pos):
    time = np.linspace(0, len(audio_data) / sample_rate, num=len(audio_data))
    pilot_magnitudes = []
    data_magnitudes = []

    for x in range(0, len(audio_data) - CHUNK_SIZE, PILOT_SEARCH_STEP):
        chunk = audio_data[x:x + CHUNK_SIZE]
        pilot_magnitude = get_frequency_magnitude(chunk, pilot_freq)
        data_magnitude = get_frequency_magnitude(chunk, data_freq)
        pilot_magnitudes.append(pilot_magnitude)
        data_magnitudes.append(data_magnitude)

    chunk_times = time[::PILOT_SEARCH_STEP]

    plt.figure(figsize=(12, 6))
    plt.plot(chunk_times[:len(pilot_magnitudes)], pilot_magnitudes, label=f'Magnitude of {pilot_freq} Hz (Pilot)', color='blue')
    plt.title(f'Magnitude of Pilot ({pilot_freq} Hz) frequency Over Time')
    plt.xlabel("Time (seconds)")
    plt.ylabel("Magnitude")

    plt.xlim(0, ((start_pos + 44100) / SAMPLE_RATE))
    plt.grid(True)

    plt.axvline(x=(start_pos) / SAMPLE_RATE, color='green', linestyle='-', linewidth=1.5, label="Pilot Signal Start")
    plt.legend()
    plt.show()

if __name__ == "__main__":
    audio_files = list(AUDIO_FOLDER.glob("*.wav"))
    actual_binary = "01001000011001010110101000100000011010000110010101101010001011100010000001011001011001010010000100111111001011100010111000101110"
    
    for audio_file in audio_files:
        print(f"Processing file: {audio_file}")
        
        audio_data, sample_rate = decode_fsk_signal(audio_file)
        
        pilot_start = find_pilot_signal(audio_data) 
        if pilot_start is not None:
                plot_magnitude_over_time(audio_data, sample_rate, PILOT_FREQUENCY, FREQUENCY, pilot_start)
        else:
                print("Could not plot graphs: pilot signal not found")