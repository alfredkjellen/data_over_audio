import numpy as np
import matplotlib.pyplot as plt
from scipy.io.wavfile import write

carrier_freq = 1000
sampling_rate = 44100
duration = 20
bits = [1, 1, 1, 1]

t = np.linspace(0, duration, int(sampling_rate * duration), endpoint=False)

carrier_wave = np.sin(2 * np.pi * carrier_freq * t)

bit_duration = duration / len(bits)
samples_per_bit = int(sampling_rate * bit_duration)

modulated_signal = np.zeros_like(t)
for i, bit in enumerate(bits):
    start_idx = i * samples_per_bit
    end_idx = start_idx + samples_per_bit
    modulated_signal[start_idx:end_idx] = (1 + bit) * carrier_wave[start_idx:end_idx]

if end_idx < len(modulated_signal):
    modulated_signal[end_idx:] = (1 + bits[-1]) * carrier_wave[end_idx:]

write('am_modulated_signal.wav', sampling_rate, modulated_signal.astype(np.float32))

plt.figure(figsize=(12, 6))
plt.plot(t, modulated_signal)
plt.title(f'AM-modulated Signal with {len(bits)} bits over 10 seconds')
plt.xlabel('Time (seconds)')
plt.ylabel('Amplitude')
plt.grid(True)
plt.show()

print("AM-modulated signal saved as 'am_modulated_signal.wav'")