import numpy as np
from scipy.io import wavfile
from scipy.fft import fft
import matplotlib.pyplot as plt
from pathlib import Path
from translator import binary_to_text

# KONSTANTER
FREQUENCIES = [1511, 2000]
PILOT_FREQUENCY = 750
SAMPLE_RATE = 44100
BIT_DURATION = 0.2
PILOT_DURATION = 0.1
CHUNK_SIZE = int(SAMPLE_RATE * BIT_DURATION)
PILOT_CHUNK_SIZE = int(SAMPLE_RATE * PILOT_DURATION)
PILOT_SEARCH_STEP = 50
AUDIO_FOLDER = Path("2f")

# Thresholds
ABSOLUTE_MAGNITUDE_THRESHOLD = 25000
RELATIVE_MAGNITUDE_THRESHOLD = 1.5

def get_frequency_magnitude(chunk, target_freq, bandwidth=50):
    if len(chunk) == 0:
        return 0
    
    fft_result = fft(chunk)
    magnitude = np.abs(fft_result[:len(chunk)//2])
    freq_bins = np.fft.fftfreq(len(chunk), 1/SAMPLE_RATE)[:len(chunk)//2]
    
    mask = np.abs(freq_bins - target_freq) <= bandwidth
    
    if not np.any(mask):
        return 0
    
    return np.max(magnitude[mask])


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
        if current_magnitude < ABSOLUTE_MAGNITUDE_THRESHOLD:
            print ('för låg')
            return 0
        else:
            print('skillnad')
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
        for freq in FREQUENCIES:
            detected = detect_bit(chunk, previous_chunk, freq)
            binary_data.append(detected)
            previous_chunk = chunk
    
    return binary_data, audio_data, sample_rate

def plot_time_domain(audio_data, sample_rate, binary_data, start_pos, actual_binary):
    time = np.linspace(0, len(audio_data) / sample_rate, num=len(audio_data))
    
    actual_binary = actual_binary[:len(binary_data)]

    plt.figure(figsize=(12, 10))

    # Första subplot för detekterad binär data
    plt.subplot(2, 1, 1)
    pilot_pos = start_pos - PILOT_CHUNK_SIZE
    pilot_time = time[:start_pos]
    plt.plot(pilot_time, audio_data[:start_pos], color='blue', label="Pilot Signal")

    for i in range(0, len(binary_data), 2):  # Stega med 2 för att läsa två bitar åt gången
        bit_start = int(start_pos + (i // 2) * CHUNK_SIZE)
        bit_end = bit_start + CHUNK_SIZE
        bit_time = time[bit_start:bit_end]
        
        # Ta ut de två bitarna (om det finns två)
        bit_pair = binary_data[i:i+2]
        if len(bit_pair) < 2:
            break  # Om det inte finns två bitar, avbryt
        
        # Kombinera till en 2-bits sträng
        bit_string = ''.join(map(str, bit_pair))

        # Bestäm färgen baserat på bitkombinationen
        if bit_string == '00':
            bit_color = 'red'
        elif bit_string == '01':
            bit_color = 'orange'
        elif bit_string == '10':
            bit_color = 'yellow'
        elif bit_string == '11':
            bit_color = 'green'

        plt.plot(bit_time, audio_data[bit_start:bit_end], color=bit_color)

    for i in range(0, len(binary_data) // 2 + 1):
        bit_start_time = (start_pos + i * CHUNK_SIZE) / sample_rate
        plt.axvline(x=bit_start_time, color='black', linestyle='--', linewidth=0.5)

    pilot_start_time = pilot_pos / sample_rate
    plt.axvline(x=pilot_start_time, color='blue', linestyle='-', linewidth=1.5, label="Pilot Signal Start")


    plt.title("Decoded Audio Signal with Bit Markings")
    plt.xlabel("Time (seconds)")
    plt.ylabel("Amplitude")
    plt.legend()

    # Andra subplot för den faktiska binära sekvensen (ground truth)
    plt.subplot(2, 1, 2)
    plt.plot(pilot_time, audio_data[:start_pos], color='blue', label="Pilot Signal")

    for i in range(0, len(actual_binary), 2):  # Stega med 2 för att läsa två bitar åt gången
        bit_start = int(start_pos + (i // 2) * CHUNK_SIZE)
        bit_end = bit_start + CHUNK_SIZE
        bit_time = time[bit_start:bit_end]
        
        # Ta ut de två bitarna från actual_binary
        actual_bit_pair = actual_binary[i:i+2]
        if len(actual_bit_pair) < 2:
            break  # Om det inte finns två bitar, avbryt
        
        # Kombinera till en 2-bits sträng
        actual_bit_string = ''.join(actual_bit_pair)

        # Bestäm färgen baserat på actual_bit_pair
        if actual_bit_string == '00':
            bit_color = 'red'
        elif actual_bit_string == '01':
            bit_color = 'orange'
        elif actual_bit_string == '10':
            bit_color = 'yellow'
        elif actual_bit_string == '11':
            bit_color = 'green'

        plt.plot(bit_time, audio_data[bit_start:bit_end], color=bit_color)

    for i in range(0, len(actual_binary) // 2 + 1):
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


def plot_magnitude_over_time(audio_data, sample_rate, pilot_freq, pilot_start, chunk_size):
    time = np.linspace(0, len(audio_data) / sample_rate, num=len(audio_data))
    pilot_magnitudes = []

    for x in range(0, len(audio_data) - CHUNK_SIZE, PILOT_SEARCH_STEP):
        chunk = audio_data[x:x + chunk_size]
        pilot_magnitude = get_frequency_magnitude(chunk, pilot_freq)
        pilot_magnitudes.append(pilot_magnitude)

    chunk_times = time[::PILOT_SEARCH_STEP]

    plt.figure(figsize=(12, 6))
    plt.plot(chunk_times[:len(pilot_magnitudes)], pilot_magnitudes, label=f'Magnitude of {pilot_freq} Hz (Pilot)', color='blue')
    plt.title(f'Magnitude of Pilot ({pilot_freq} Hz) frequency Over Time')
    plt.xlabel("Time (seconds)")
    plt.ylabel("Magnitude")

    plt.grid(True)

    plt.axvline(x=(pilot_start) / SAMPLE_RATE, color='green', linestyle='-', linewidth=1.5, label="Pilot Signal Start")
    plt.legend()
    plt.show()

if __name__ == "__main__":
    audio_files = list(AUDIO_FOLDER.glob("*.wav"))
    actual_binary = "01000001011000010010000001000010011000100010000001000011011000110010000001000100011001000010000001000101011001010010000001000110011001100010000001000111011001110010000001001000011010000010000001001001"
    
    for audio_file in audio_files:
        print(f"Processing file: {audio_file}")
        
        binary_data, audio_data, sample_rate = decode_fsk_signal(audio_file)
        
        if binary_data:
            print("\nDecoded binary data:")
            print(''.join(map(str, binary_data)))
            
            decoded_text = binary_to_text(binary_data)
            print("\nDecoded text:")
            print(decoded_text)
            
            pilot_start = int(find_pilot_signal(audio_data))
            start_pos = pilot_start + PILOT_CHUNK_SIZE
            if pilot_start is not None:
                plot_time_domain(audio_data, sample_rate, binary_data, start_pos, actual_binary)
                plot_frequency_domain(audio_data, sample_rate)
                plot_magnitude_over_time(audio_data, sample_rate, PILOT_FREQUENCY, pilot_start, 400)
                plot_magnitude_over_time(audio_data, sample_rate, PILOT_FREQUENCY, pilot_start, PILOT_CHUNK_SIZE)

        else:
            print("No data could be decoded from:", audio_file)