import numpy as np
from scipy.io import wavfile
from scipy.fft import fft
import matplotlib.pyplot as plt
from translator import binary_to_text
from pathlib import Path
 

# CONSTANTS
FREQUENCY = 2000
PILOT_FREQUENCY = 2000
SAMPLE_RATE = 44100
BIT_DURATION = 0.1
PILOT_DURATION = 0.1
CHUNK_SIZE = int(SAMPLE_RATE * BIT_DURATION)
PILOT_CHUNK_SIZE = int(SAMPLE_RATE * PILOT_DURATION)
PILOT_CHUNK_STEP = 50
AUDIO_FOLDER = Path("recordings")

# Thresholds
ABSOLUTE_MAGNITUDE_THRESHOLD = 50000
RELATIVE_MAGNITUDE_THRESHOLD = 1.5

def get_frequency_magnitude(chunk, freq):
    fft_result = fft(chunk)
    magnitude = np.abs(fft_result[:len(chunk)//2])
    freq_bins = np.fft.fftfreq(len(chunk), 1/SAMPLE_RATE)[:len(chunk)//2]
    index = np.argmin(np.abs(freq_bins - freq))
    return magnitude[index]

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

def find_pilot_signal_fft(audio_data):
    for x in range(0, len(audio_data) - PILOT_CHUNK_SIZE, PILOT_CHUNK_STEP):
        chunk = audio_data[x:x + PILOT_CHUNK_SIZE]
        if get_frequency_magnitude(chunk, PILOT_FREQUENCY) > np.mean(np.abs(fft(chunk))) * 10:
            print(f'Pilot signal (FFT method) starts at {x + PILOT_CHUNK_SIZE}')
            return x + 2 * PILOT_CHUNK_SIZE
    return None

def decode_fsk_signal(filename):
    print(f"Reading audio file: {filename}\n")
    sample_rate, audio_data = wavfile.read(filename)
    print(f"Sample rate: {sample_rate} Hz")
    print(f"Audio data length: {len(audio_data)} samples")
    
    if len(audio_data.shape) == 2:
        print("Audio is stereo, converting to mono...")
        audio_data = np.mean(audio_data, axis=1).astype(audio_data.dtype)

    start_pos_fft = find_pilot_signal_fft(audio_data)
    
    if start_pos_fft is not None:
        start_pos = start_pos_fft
        print("Using FFT method for pilot signal detection")
    else:
        print("Pilot signal not found")
        return None, audio_data, sample_rate

    end_pos = start_pos + 10 * sample_rate                              


    print(f'Data signal starts at sample {start_pos}')
    binary_data = []
    
    previous_chunk = np.zeros(CHUNK_SIZE)
    for x in range(start_pos, end_pos, CHUNK_SIZE):
        chunk = audio_data[x:x + CHUNK_SIZE]
        detected = detect_bit(chunk, previous_chunk, FREQUENCY)
        binary_data.append(detected)
        previous_chunk = chunk
    
    return binary_data, audio_data, sample_rate

def plot_time_domain(audio_data, sample_rate, binary_data, start_pos):
    time = np.linspace(0, len(audio_data) / sample_rate, num=len(audio_data))
    
    plt.figure(figsize=(12, 6))
    
    pilot_pos = start_pos - int(PILOT_DURATION * sample_rate)
    pilot_time = time[:int(start_pos)]
    plt.plot(pilot_time, audio_data[:int(start_pos)], color='blue', label="Pilot Signal")
    
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
    
    plt.title("Audio Signal with Bit and Pilot Signal Markings")
    plt.xlabel("Time (seconds)")
    plt.ylabel("Amplitude")
    plt.legend()
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

if __name__ == "__main__":
    audio_files = list(AUDIO_FOLDER.glob("*.wav"))  # Läser in alla .wav-filer, ändra filtillägg om nödvändigt
   
    for audio_file in audio_files:
        print(f"Processing file: {audio_file}")
       
        # Ladda och bearbeta varje ljudfil
        binary_data, audio_data, sample_rate = decode_fsk_signal(audio_file)
       
        if binary_data:
            print("\nDecoded binary data:")
            print(''.join(map(str, binary_data)))
           
            decoded_text = binary_to_text(binary_data)
            print("\nDecoded text:")
            print(decoded_text)
           
            start_pos = find_pilot_signal_fft(audio_data)
            if start_pos:
                plot_time_domain(audio_data, sample_rate, binary_data, start_pos)
                plot_frequency_domain(audio_data, sample_rate)
            else:
                print("Could not plot graphs: pilot signal not found")
        else:
            print("No data could be decoded from:", audio_file)