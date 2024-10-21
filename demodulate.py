import numpy as np
from scipy.io import wavfile
from scipy.fft import fft

FREQUENCIES = [730, 950, 1300, 1800, 2400, 3100, 3800, 4700]
PILOT_FREQUENCY = 520
BIT_DURATION = 0.2
PILOT_DURATION = 0.1
PILOT_SEARCH_STEP = 50
ABSOLUTE_MAGNITUDE_THRESHOLD = 75000
RELATIVE_MAGNITUDE_THRESHOLD = 5
MAGNITUDE_THRESHOLDS = {freq: ABSOLUTE_MAGNITUDE_THRESHOLD if freq != 4700 else 25000 for freq in FREQUENCIES}


def get_frequency_magnitude(chunk, target_freq, sample_rate, bandwidth=50):
    if len(chunk) == 0:
        return 0
   
    fft_result = fft(chunk)
    magnitude = np.abs(fft_result[:len(chunk)//2])
    freq_bins = np.fft.fftfreq(len(chunk), 1/sample_rate)[:len(chunk)//2]
   
    mask = np.abs(freq_bins - target_freq) <= bandwidth
   
    if not np.any(mask):
        return 0
   
    return np.max(magnitude[mask])
 
 
def get_pilot_start(audio_data, pilot_chunk_size, sample_rate):
    max_magnitude = 0
    pilot_start = 0
   
    for i in range(0, len(audio_data) - pilot_chunk_size, PILOT_SEARCH_STEP):
        chunk = audio_data[i:i + pilot_chunk_size]
        magnitude = get_frequency_magnitude(chunk, PILOT_FREQUENCY, sample_rate)
       
        if magnitude > max_magnitude:
            max_magnitude = magnitude
            pilot_start = i
   
    return pilot_start
 
def detect_bit(chunk, previous_chunk, freq, threshold, sample_rate):
    current_magnitude = get_frequency_magnitude(chunk, freq, sample_rate)
    previous_magnitude = get_frequency_magnitude(previous_chunk, freq, sample_rate)
   
    if (current_magnitude > threshold and
        current_magnitude > previous_magnitude * RELATIVE_MAGNITUDE_THRESHOLD):
        return 1
   
    elif (current_magnitude < threshold or
          current_magnitude * RELATIVE_MAGNITUDE_THRESHOLD < previous_magnitude):
        return 0
    else:
        return detect_bit(previous_chunk, np.zeros_like(previous_chunk), freq, threshold, sample_rate)


def decode_fsk_signal(filename):
    sample_rate, audio_data = wavfile.read(filename)
    chunk_size = int(sample_rate * BIT_DURATION)
    pilot_chunk_size = int(sample_rate * PILOT_DURATION)
   
    if len(audio_data.shape) == 2:
        audio_data = np.mean(audio_data, axis=1).astype(audio_data.dtype)
 
    start_pos = get_pilot_start(audio_data, pilot_chunk_size, sample_rate) + pilot_chunk_size
   
    if start_pos is None:
        print("Pilot signal not found")
        return None, audio_data, sample_rate
 
    
    end_pos = start_pos + 10 * sample_rate # Signal is 10 seconds
 
    bits = []
   
    previous_chunk = audio_data[start_pos - chunk_size: start_pos]
    for i in range(start_pos, end_pos, chunk_size):
        chunk = audio_data[i:i + chunk_size]
        for freq in FREQUENCIES:
            detected = detect_bit(chunk, previous_chunk, freq, MAGNITUDE_THRESHOLDS[freq], sample_rate)
            bits.append(detected)
            previous_chunk = chunk
   
    return bits, audio_data, sample_rate

