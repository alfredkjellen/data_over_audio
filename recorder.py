import sounddevice as sd  # Import library for audio recording
from scipy.io import wavfile  # Import library for saving audio files
import numpy as np  # Import NumPy for array handling

# Constants for sampling rate and recording duration
SAMPLERATE = 44100  # Number of samples per second (44.1 kHz, standard for audio)
DURATION = 16  # Duration of the recording in seconds
CHANNELS = 2  # Number of audio channels (2 for stereo)

recording = None  # Variable to store the recording
is_recording = False  # Flag to indicate whether recording is in progress

def record_audio(samplerate, duration, channels):
    """
    Record audio for a specified duration.

    Input:
        samplerate (int): The number of samples per second (sampling frequency).
        duration (int/float): Duration of the recording in seconds.
        channels (int): Number of audio channels (1 for mono, 2 for stereo).

    Output:
        recording (numpy.ndarray): The recorded audio signal as a NumPy array.
    """
    global recording, is_recording
    print(f"Recording... Will stop automatically after {duration} seconds.")

    # Start recording: using 'float64' format for high-quality audio
    recording = sd.rec(int(duration * samplerate), samplerate=samplerate, channels=channels, dtype='float64')
    is_recording = True

    # Wait until the recording is finished
    sd.wait()
    is_recording = False
    print("Recording done!")

    return recording

def save_audio(file_name, data, samplerate):
    """
    Save the recorded audio signal as a WAV file.

    Input:
        file_name (str): The name of the output file.
        data (numpy.ndarray): The audio data to be saved.
        samplerate (int): The sampling frequency for the audio file.
    """
    # Convert the recording from 'float64' to 'int16' as WAV format requires it
    audio_data_int16 = np.int16(data * 32767)  # 32767 is the max value for 16-bit audio
    wavfile.write(file_name, samplerate, audio_data_int16)  # Save as WAV file
    print(f"Recording saved as: {file_name}")

# Main section to execute the recording and save it
if __name__ == "__main__":
    # Step 1: Record the audio
    audio_data = record_audio(SAMPLERATE, DURATION, CHANNELS)

    # Step 2: Save the recording to a WAV file
    OUTPUT_FILE = 'signal.wav'  # Name of the output file
    save_audio(OUTPUT_FILE, audio_data, SAMPLERATE)
