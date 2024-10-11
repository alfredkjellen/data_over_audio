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

def detect_bit(chunk, previous_chunk, freq):
    current_magnitude = get_frequency_magnitude(chunk, freq)
    previous_magnitude = get_frequency_magnitude(previous_chunk, freq)
    
    if (current_magnitude > ABSOLUTE_MAGNITUDE_THRESHOLD and 
        current_magnitude > previous_magnitude * RELATIVE_MAGNITUDE_THRESHOLD):
        return 1
    elif (current_magnitude < ABSOLUTE_MAGNITUDE_THRESHOLD or 
          current_magnitude * RELATIVE_MAGNITUDE_THRESHOLD < previous_magnitude):
        return 0
    else:
        return detect_bit(previous_chunk, np.zeros_like(previous_chunk), freq)

def decode_fsk_signal(filename):
    print(f"Reading audio file: {filename}\n")
    sample_rate, audio_data = wavfile.read(filename)
    print(f"Sample rate: {sample_rate} Hz")
    print(f"Audio data length: {len(audio_data)} samples")
    
    if len(audio_data.shape) == 2:
        print("Audio is stereo, converting to mono...")
        audio_data = np.mean(audio_data, axis=1).astype(audio_data.dtype)

    start_pos = int(find_pilot_signal(audio_data) + PILOT_CHUNK_SIZE)
    
    if start_pos is None:
        print("Pilot signal not found")
        return None, audio_data, sample_rate

    end_pos = start_pos + 10 * sample_rate

    print(f'Data signal starts at sample {start_pos}')
    binary_data = []
    
    previous_chunk = audio_data[start_pos - CHUNK_SIZE: start_pos]
    for x in range(start_pos, end_pos, CHUNK_SIZE):
        chunk = audio_data[x:x + CHUNK_SIZE]
        detected = detect_bit(chunk, previous_chunk, FREQUENCY)
        binary_data.append(detected)
        previous_chunk = chunk
    
    return binary_data, audio_data, sample_rate

def plot_time_domain(audio_data, sample_rate, binary_data, start_pos, actual_binary):
    time = np.linspace(0, len(audio_data) / sample_rate, num=len(audio_data))
    
    actual_binary = actual_binary[:len(binary_data)]
    
    plt.figure(figsize=(12, 10))
    
    plt.subplot(2, 1, 1)
    pilot_pos = start_pos - PILOT_CHUNK_SIZE
    pilot_time = time[:start_pos]
    plt.plot(pilot_time, audio_data[:start_pos], color='blue', label="Pilot Signal")
    
    for i, bit in enumerate(binary_data):
        bit_start = int(start_pos + i * CHUNK_SIZE)
        bit_end = bit_start + CHUNK_SIZE
        bit_time = time[bit_start:bit_end]
        bit_color = 'green' if bit == 1 else 'red'
        plt.plot(bit_time, audio_data[bit_start:bit_end], color=bit_color)
    
    for i in range(len(binary_data) + 1):
        bit_start_time = (start_pos + i * CHUNK_SIZE) / sample_rate
        plt.axvline(x=bit_start_time, color='black', linestyle='--', linewidth=0.5)

    pilot_start_time = pilot_pos / sample_rate
    plt.axvline(x=pilot_start_time, color='blue', linestyle='-', linewidth=1.5, label="Pilot Signal Start")
    
    plt.title("Decoded Audio Signal with Bit Markings")
    plt.xlabel("Time (seconds)")
    plt.ylabel("Amplitude")
    plt.legend()

    plt.subplot(2, 1, 2)
    plt.plot(pilot_time, audio_data[:start_pos], color='blue', label="Pilot Signal")
    
    for i, bit in enumerate(actual_binary):
        bit_start = int(start_pos + i * CHUNK_SIZE)
        bit_end = bit_start + CHUNK_SIZE
        bit_time = time[bit_start:bit_end]
        bit_color = 'green' if bit == '1' else 'red'
        plt.plot(bit_time, audio_data[bit_start:bit_end], color=bit_color)
    
    for i in range(len(actual_binary) + 1):
        bit_start_time = (start_pos + i * CHUNK_SIZE) / sample_rate
        plt.axvline(x=bit_start_time, color='black', linestyle='--', linewidth=0.5)

    plt.axvline(x=pilot_start_time, color='blue', linestyle='-', linewidth=1.5, label="Pilot Signal Start")
    
    plt.title("Actual Audio Signal with Bit Markings (Ground Truth)")
    plt.xlabel("Time (seconds)")
    plt.ylabel("Amplitude")
    plt.legend()

    plt.tight_layout()
    plt.show()

def plot_frequency_domain(audio_data, sample_rate):
    fft_result = fft(audio_data)
    magnitude = np.abs(fft_result[:len(audio_data)//2])
    freq_bins = np.fft.fftfreq(len(audio_data), 1/sample_rate)[:len(audio_data)//2]

    plt.figure(figsize=(12, 6))
    plt.plot(freq_bins, magnitude)
    plt.xlim(0, 5000)
    plt.title("Frequency Domain Representation")
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Magnitude")
    plt.show()

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

    plt.axvline(x=(start_pos - PILOT_CHUNK_SIZE) / SAMPLE_RATE, color='green', linestyle='-', linewidth=1.5, label="Pilot Signal Start")
    plt.legend()
    plt.show()

if __name__ == "__main__":
    audio_files = list(AUDIO_FOLDER.glob("*.wav"))
    actual_binary = "01001000011001010110101000100000011010000110010101101010001011100010000001011001011001010010000100111111001011100010111000101110"
    
    for audio_file in audio_files:
        print(f"Processing file: {audio_file}")
        
        binary_data, audio_data, sample_rate = decode_fsk_signal(audio_file)
        
        if binary_data:
            print("\nDecoded binary data:")
            print(''.join(map(str, binary_data)))
            
            decoded_text = binary_to_text(binary_data)
            print("\nDecoded text:")
            print(decoded_text)
            
            start_pos = int(find_pilot_signal(audio_data) + PILOT_CHUNK_SIZE )
            if start_pos is not None:
                plot_time_domain(audio_data, sample_rate, binary_data, start_pos, actual_binary)
                plot_frequency_domain(audio_data, sample_rate)
                plot_magnitude_over_time(audio_data, sample_rate, PILOT_FREQUENCY, FREQUENCY, start_pos)
            else:
                print("Could not plot graphs: pilot signal not found")
        else:
            print("No data could be decoded from:", audio_file)