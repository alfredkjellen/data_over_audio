import numpy as np
import sounddevice as sd
import scipy.io.wavfile as wavfile
import matplotlib.pyplot as plt
from translate import text_to_binary

# Constants
FREQUENCIES = [730, 950, 1300, 1800, 2400, 3100, 3800, 4700]  # Frequency for each bit position
PILOT_FREQUENCY = 520  # Pilot signal frequency in Hz
SAMPLING_FREQUENCY = 44100  # Sampling rate for signal generation (Hz)
BIT_DURATION = 0.2  # Duration of each bit in seconds
PILOT_DURATION = 0.1  # Duration of pilot signal in seconds
SIGNAL_DURATION = 10 # Duration of entire signal in seconds
AMPLITUDE = 0.5  # Amplitude of the signal (scaling factor)
CHUNK_SIZE = int(SAMPLING_FREQUENCY * BIT_DURATION)  # Number of samples per chunk
BIT_MODULATION_FREQUENCY = 1 / BIT_DURATION  # Modulation frequency for bit transitions
PILOT_MODULATION_FREQUENCY = 1 / PILOT_DURATION  # Modulation frequency for pilot signal
EASE_OUT_EXPONENT = 9
DATA_FILE = text_to_binary("text.txt")  # Convert text file to binary data

def generate_sine_wave(frequency, duration):
    """
    Generates a sine wave at a given frequency and duration.
    
    Input:
        frequency (float): Frequency of the sine wave in Hz.
        duration (float): Duration of the sine wave in seconds.
    
    Output:
        numpy array: The sine wave values.
    """
    t = np.linspace(0, duration, int(SAMPLING_FREQUENCY * duration), endpoint=False)
    return AMPLITUDE * np.sin(2 * np.pi * frequency * t)

def create_modulation_wave(duration, modulation_frequency):
    """
    Creates a modulation wave to modulate the signal amplitude.
    
    Input:
        duration (float): Duration for which the modulation wave is generated (seconds).
        modulation_frequency (float): Frequency for modulating signal transitions (Hz).
    
    Output:
        numpy array: Modulation wave for the given duration.
    """
    t = np.linspace(0, duration, int(SAMPLING_FREQUENCY * duration), endpoint=False)
    return np.abs(np.sin(np.pi * modulation_frequency * t))

def ease_out(duration):
    """
    Generates an ease-out curve to reduce echo.
    
    Input:
        duration (float): The length of the easing function in seconds.
    
    Output:
        numpy array: A vector of values that decay exponentially.
    """
    t = np.linspace(0, duration, int(SAMPLING_FREQUENCY * duration), endpoint=False)
    return np.exp(-EASE_OUT_EXPONENT * t / duration)

def create_signal(data):
    """
    Converts binary data into a modulated audio signal.

    Input:
        data (str): A binary string (composed of '0's and '1's).
    
    Output:
        numpy array: A modulated signal based on the input binary data.
    """
    # Create pilot modulation arrays
    pilot_modulation_wave = create_modulation_wave(PILOT_DURATION, PILOT_MODULATION_FREQUENCY)
    modulation_wave = create_modulation_wave(BIT_DURATION, BIT_MODULATION_FREQUENCY)
    ease_out_vector = ease_out(BIT_DURATION)

    # Create pilot signal
    clear_pilot_signal = generate_sine_wave(PILOT_FREQUENCY, PILOT_DURATION)
    pilot_signal = pilot_modulation_wave * clear_pilot_signal * AMPLITUDE

    # Creates an empty signal
    signal = np.array([])

    # Process data in chunks of 8 bits
    for x in range(0, len(data), 8):
        chunk = data[x:x+8]  # Take 8 bits at a time
        chunk_signal = np.zeros(CHUNK_SIZE)  # Empty signal chunk
        
        # For each bit, modulate the corresponding frequency
        for y, bit in enumerate(chunk):
            if bit == '1':
                sine_wave = generate_sine_wave(FREQUENCIES[y], BIT_DURATION)
                sine_wave = sine_wave * ease_out_vector  # Apply easing
                chunk_signal += sine_wave 

        # Apply modulation wave for smooth transitions between chunks
        chunk_signal = chunk_signal * modulation_wave

        # Add chunk to excisting signal
        signal = np.concatenate((signal, chunk_signal))

    # Concatenate pilot signal with the main signal
    signal = np.concatenate((pilot_signal, signal))
    
    # Limit the signal to 10 seconds
    signal = signal[:SIGNAL_DURATION * SAMPLING_FREQUENCY]

    return signal

# Main program execution
if __name__ == "__main__":
    # Display the binary data
    print(f"Binary data:\n{DATA_FILE}\n")

    # Generate the signal from binary data
    signal = create_signal(DATA_FILE)

    # Save the signal as a WAV file
    output_filename = "8f_signal.wav"
    wavfile.write(output_filename, SAMPLING_FREQUENCY, (signal * 32767).astype(np.int16))
    print(f"Audio file saved as {output_filename}\n")

    # Play the signal using the sound device
    print("Playing signal...")
    sd.play(signal, SAMPLING_FREQUENCY)

    # Plot the signal for visualization
    plt.plot(signal)
    plt.show()

    # Wait for playback to finish
    sd.wait()
    print("Done!")