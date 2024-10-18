import numpy as np
from scipy.io import wavfile
from scipy.fft import fft


FREQUENCIES = [730, 950, 1300, 1800, 2400, 3100, 3800, 4700]
PILOT_FREQUENCY = 520
SAMPLE_RATE = 44100
BIT_DURATION = 0.2
PILOT_DURATION = 0.1
CHUNK_SIZE = int(SAMPLE_RATE * BIT_DURATION)
PILOT_CHUNK_SIZE = int(SAMPLE_RATE * PILOT_DURATION)
PILOT_SEARCH_STEP = 50

ABSOLUTE_MAGNITUDE_THRESHOLD = 50000
RELATIVE_MAGNITUDE_THRESHOLD = 5
MAGNITUDE_THRESHOLDS = {730:ABSOLUTE_MAGNITUDE_THRESHOLD, 950:ABSOLUTE_MAGNITUDE_THRESHOLD, 1300:ABSOLUTE_MAGNITUDE_THRESHOLD, 1800:ABSOLUTE_MAGNITUDE_THRESHOLD, 2400:ABSOLUTE_MAGNITUDE_THRESHOLD, 3100:ABSOLUTE_MAGNITUDE_THRESHOLD, 3800:ABSOLUTE_MAGNITUDE_THRESHOLD, 4700:25000}

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
   
    return pilot_start
 
def detect_bit(chunk, previous_chunk, freq, threshold):
    current_magnitude = get_frequency_magnitude(chunk, freq)
    previous_magnitude = get_frequency_magnitude(previous_chunk, freq)
   
    if (current_magnitude > threshold and
        current_magnitude > previous_magnitude * RELATIVE_MAGNITUDE_THRESHOLD):
        return 1
   
    elif (current_magnitude < threshold or
          current_magnitude * RELATIVE_MAGNITUDE_THRESHOLD < previous_magnitude):
        return 0
    else:
        return detect_bit(previous_chunk, np.zeros_like(previous_chunk), freq, threshold)


def decode_fsk_signal(filename):
    sample_rate, audio_data = wavfile.read(filename)
   
    if len(audio_data.shape) == 2:
        audio_data = np.mean(audio_data, axis=1).astype(audio_data.dtype)
 
    start_pos = int(find_pilot_signal(audio_data) + PILOT_CHUNK_SIZE)
   
    if start_pos is None:
        print("Pilot signal not found")
        return None, audio_data, sample_rate
 
    end_pos = start_pos + 10 * sample_rate
 
    binary_data = []
   
    previous_chunk = audio_data[start_pos - CHUNK_SIZE: start_pos]
    for x in range(start_pos, end_pos, CHUNK_SIZE):
        chunk = audio_data[x:x + CHUNK_SIZE]
        for freq in FREQUENCIES:
            detected = detect_bit(chunk, previous_chunk, freq, MAGNITUDE_THRESHOLDS[freq])
            binary_data.append(detected)
            previous_chunk = chunk
   
    return binary_data, audio_data, sample_rate