import numpy as np
from scipy.io import wavfile
from scipy.fft import fft
import matplotlib.pyplot as plt
from translator import binary_to_text

# KONSTANTER
FREKVENS = 2000  
PILOT_FREKVENS = 2000
SAMPLE_RATE = 44100  
BIT_DURATION = 0.1  
PILOT_DURATION = 0.1
CHUNK_SIZE = int(SAMPLE_RATE * BIT_DURATION)
PILOT_CHUNK_SIZE = int(SAMPLE_RATE * PILOT_DURATION)
PILOT_CHUNK_STEP = 50
AUDIO_FILE = "recording.wav"

def detect_frequency(chunk, freq):
    fft_result = fft(chunk)
    magnitude = np.abs(fft_result[:len(chunk)//2])
    freq_bins = np.fft.fftfreq(len(chunk), 1/SAMPLE_RATE)[:len(chunk)//2]
    
    
    index = np.argmin(np.abs(freq_bins - freq))
    threshold = np.mean(magnitude) * 100
    return 1 if magnitude[index] > threshold else 0

def find_pilot_signal(audio_data):
    for x in range(0, len(audio_data), PILOT_CHUNK_STEP):
        chunk = audio_data[x:x + PILOT_CHUNK_SIZE]
        if detect_frequency (chunk, PILOT_FREKVENS) == 1:
            print(f'pilotsignal börjar vid {x + PILOT_CHUNK_SIZE}')
            return x + 2 * PILOT_CHUNK_SIZE
    return None

def decode_fsk_signal(filename):
    print(f"Läser in ljudfilen: {filename}\n")
    sample_rate, audio_data = wavfile.read(filename)
    print(f"Samplingsfrekvens: {sample_rate} Hz")
    print(f"Ljuddatans längd: {len(audio_data)} samples")
    
    # Kontrollera om ljudet är stereo
    if len(audio_data.shape) == 2:  # Om det är fler än en kanal (stereo)
        print("Ljudet är stereo, konverterar till mono...")
        # Ta medelvärdet av de två kanalerna (stereo till mono)
        audio_data = np.mean(audio_data, axis=1).astype(audio_data.dtype)

    start_pos = find_pilot_signal(audio_data)
    if not start_pos:
        print("pilotsignal khittades inte")
        return 
    
    print(f'Datasignal börjar vid sample {start_pos}')
    binary_data = []
    
    for x in range(start_pos, len(audio_data), CHUNK_SIZE):
        chunk = audio_data[x:x + CHUNK_SIZE]
        detected = detect_frequency(chunk, FREKVENS)
        binary_data.append(detected)
    
    return binary_data, audio_data, sample_rate


def plot_time_domain(audio_data, sample_rate, binary_data, start_pos):
    # Skapa tidsvektorn
    time = np.linspace(0, len(audio_data) / sample_rate, num=len(audio_data))
    
    # Plotta ljudsignalen
    plt.figure(figsize=(12, 6))
    
    # Pilotsignalen före datasignalen (start_pos)
    pilot_pos = start_pos - int(PILOT_DURATION * sample_rate)
    pilot_time = time[:int(start_pos)]  # Tid för pilotsignalen
    plt.plot(pilot_time, audio_data[:int(start_pos)], color='blue', label="Pilotsignal")
    
    # Plotta ljudsignalen för varje bit baserat på binärdata (1 eller 0)
    for i, bit in enumerate(binary_data):
        bit_start = int(start_pos + i * CHUNK_SIZE)
        bit_end = bit_start + CHUNK_SIZE
        bit_time = time[bit_start:bit_end]  # Tid för denna bit

        # Välj färg baserat på om det är en etta eller nolla
        bit_color = 'green' if bit == 1 else 'red'
        
        # Plotta segmentet med rätt färg
        plt.plot(bit_time, audio_data[bit_start:bit_end], color=bit_color)
    
    # Plotta vertikala linjer vid varje bitgräns för att tydliggöra bitgränser
    for i, bit in enumerate(binary_data):
        bit_start_time = (start_pos + i * CHUNK_SIZE) / sample_rate
        plt.axvline(x=bit_start_time, color='black', linestyle='--', linewidth=0.5)

    # Plotta en vertikal linje för pilotsignalens start
    pilot_start_time = pilot_pos / sample_rate
    plt.axvline(x=pilot_start_time, color='blue', linestyle='-', linewidth=1.5, label="Pilotsignal start")
    
    # Inställningar för grafen
    plt.title("Ljudsignal med färgmarkeringar för bitar och pilotsignal")
    plt.xlabel("Tid (sekunder)")
    plt.ylabel("Amplitud")
    plt.legend()
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
        
        # Hitta var datasignalen börjar (dvs efter pilotsignalen)
        start_pos = find_pilot_signal(audio_data)
        
        # Plotta ljudsignalen med bitgränser och pilotsignalens start
        plot_time_domain(audio_data, sample_rate, binary_data, start_pos)
        plot_frequency_domain(audio_data, sample_rate)
    else:
        print("Ingen data kunde avkodas")

#01001000011001010110101000100000011010000110010101101010001011100010000001011001011001010010000100111111001011100010111000101110