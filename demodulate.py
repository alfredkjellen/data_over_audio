import numpy as np
import matplotlib.pyplot as plt
from scipy.io.wavfile import read

def remove_silence(signal, threshold=250, min_duration=0.1):
    abs_signal = np.abs(signal)
    
    start = 0
    while start < len(signal) and abs_signal[start] < threshold:
        start += 1
    
    end = len(signal) - 1
    while end > start and abs_signal[end] < threshold:
        end -= 1
    
    min_samples = int(min_duration * sampling_rate)
    if end - start < min_samples:
        return signal
    
    return signal[start:end+1]

carrier_freq = 1000
sampling_rate = 44100
num_bits = 4

filename = 'am_modulated_signal.wav'
sampling_rate, modulated_signal = read(filename)

modulated_signal = remove_silence(modulated_signal)

duration = len(modulated_signal) / sampling_rate
print(f"Signal duration after silence removal: {duration:.2f} seconds")
t = np.linspace(0, duration, len(modulated_signal), endpoint=False)

bit_duration = duration / num_bits
samples_per_bit = int(sampling_rate * bit_duration)

start_index = np.argmax(np.abs(modulated_signal) > np.max(np.abs(modulated_signal)) * 0.1)
start_index = start_index - (start_index % samples_per_bit)

decoded_bits = []
amplitudes = []

previous_avg_peak_amplitude = None
for i in range(start_index, len(modulated_signal), samples_per_bit):
    bit_segment = modulated_signal[i:i + samples_per_bit]
    
    if len(bit_segment) == samples_per_bit:
        sorted_amplitudes = np.sort(np.abs(bit_segment))
        avg_peak_amplitude = np.mean(sorted_amplitudes[-1000:])
        amplitudes.append(avg_peak_amplitude)
        
        if previous_avg_peak_amplitude is None:
            decoded_bits.append(1)
        else:
            decoded_bits.append(1 if avg_peak_amplitude >= previous_avg_peak_amplitude else 0)
        
        previous_avg_peak_amplitude = avg_peak_amplitude

print("Demodulated bits:", decoded_bits)

plt.figure(figsize=(12, 6))
plt.plot(t, modulated_signal)
plt.title('Signal with borders')
plt.xlabel('Time (seconds)')
plt.ylabel('Amplitude')
plt.grid(True)

for i in range(len(decoded_bits)):
    bit_start = start_index + i * samples_per_bit
    plt.axvline(x=bit_start / sampling_rate, color='r', linestyle='--', alpha=0.5)

plt.show()

plt.figure(figsize=(12, 6))
plt.plot(range(len(amplitudes)), amplitudes, 'bo-')
plt.title('Average of Top 100 Amplitudes per Bit')
plt.xlabel('Bit Index')
plt.ylabel('Average Peak Amplitude')
plt.grid(True)
plt.show()