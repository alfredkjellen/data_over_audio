import numpy as np
from scipy.io import wavfile
from scipy.fft import fft
import matplotlib.pyplot as plt
from translator import binary_to_text

# KONSTANTER
FREKVENS = 2000  
PILOT_FREKVENS = 1009
SAMPLE_RATE = 44100  
BIT_DURATION = 0.1  
PILOT_DURATION = 0.5
CHUNK_SIZE = int(SAMPLE_RATE * BIT_DURATION)
PILOT_CHUNK_SIZE = int(SAMPLE_RATE * PILOT_DURATION)
PILOT_CHUNK_STEP = 50
AUDIO_FILE = "recording.wav"
MAX_DURATION = SAMPLE_RATE * 10

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
            return x  + PILOT_CHUNK_SIZE -750
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

    pilot_start_pos = find_pilot_signal(audio_data)
    data_start_pos = pilot_start_pos + PILOT_CHUNK_SIZE
    if not data_start_pos:
        print("pilotsignal khittades inte")
        return 
    
    print(f'Datasignal börjar vid sample {data_start_pos}')
    binary_data = []
    audio_data = audio_data[pilot_start_pos:MAX_DURATION]
    print(len(audio_data))
    print(len(audio_data)/SAMPLE_RATE)
    
    for x in range(data_start_pos, len(audio_data), CHUNK_SIZE):
        chunk = audio_data[x:x + CHUNK_SIZE]
        detected = detect_frequency(chunk, FREKVENS)
        binary_data.append(detected)
    

    return binary_data, audio_data, sample_rate


def plot_time_domain(audio_data, sample_rate, binary_data, start_pos):
    # Skapa tidsvektorn
    time = np.linspace(0, len(audio_data) / sample_rate, num=len(audio_data))
    
    # Plotta ljudsignalen
    plt.figure(figsize=(12, 6))
    plt.plot(time, audio_data, label="Ljudsignal")
    
    # Beräkna pilotens starttid (0.5 sekunder innan start_pos)
    pilot_pos = start_pos - int(PILOT_DURATION * sample_rate)
    pilot_start_time = pilot_pos / sample_rate
    plt.axvline(x=pilot_start_time, color='green', linestyle='-', linewidth=1.5, label="Pilotsignal start")
    
    # Plotta vertikala linjer vid varje bitgräns
    for i, bit in enumerate(binary_data):
        bit_start_time = (start_pos + i * CHUNK_SIZE) / sample_rate
        plt.axvline(x=bit_start_time, color='red', linestyle='--', linewidth=0.8, label="Bitgräns" if i == 0 else "")
    
    plt.title("Ljudsignal med markerade bitgränser och pilotsignal")
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
