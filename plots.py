import numpy as np
from scipy.fft import fft
import plotly.graph_objs as go
from demodulate import get_frequency_magnitude, PILOT_SEARCH_STEP, BIT_DURATION

def plot_time_domain(audio_data, sample_rate, duration, binary_data, start_pos, actual_binary):
    chunk_size = int(sample_rate * duration)
    time = np.linspace(0, len(audio_data) / sample_rate, num=len(audio_data))
    
    trace1 = go.Scatter(x=time[:start_pos], y=audio_data[:start_pos], mode='lines', name="Pilot Signal")
    traces = [trace1]
    
    for i in range(len(binary_data) // 2 + 1):
        bit_start_time = (start_pos + i * chunk_size) / sample_rate
        trace = go.Scatter(x=[bit_start_time, bit_start_time], y=[min(audio_data), max(audio_data)], 
                           mode='lines', line=dict(color='gray', dash='dash'), showlegend=False)
        traces.append(trace)

    for i in range(0, len(binary_data), 2):
        bit_start = start_pos + (i // 2) * chunk_size
        bit_time = time[bit_start:bit_start + chunk_size]
        bit_pair = binary_data[i:i+2]
        actual_bit_pair = actual_binary[i:i+2]
        bit_string = ''.join(map(str, bit_pair))
        correct_pair = ''.join(actual_bit_pair)
        
        if bit_string == correct_pair:
            color = "limegreen"
        else:
            if bit_string == '00':
                color = 'red'
            elif bit_string == '01':
                color = 'darkorange'
            elif bit_string == '10':
                color = 'yellow'
            else:
                color = 'green'

        trace = go.Scatter(x=bit_time, y=audio_data[bit_start:bit_start + chunk_size], mode='lines', 
                           line=dict(color=color), showlegend=False)
        traces.append(trace)
    
    layout = go.Layout(title="Time Domain Representation with Bit Boundaries", 
                       xaxis_title="Time (s)", yaxis_title="Amplitude")
    
    return go.Figure(data=traces, layout=layout)


def plot_frequency_domain(audio_data, sample_rate):
    fft_result = fft(audio_data)
    magnitude = np.abs(fft_result[:len(audio_data)//2])
    freq_bins = np.fft.fftfreq(len(audio_data), 1/sample_rate)[:len(audio_data)//2]
    trace = go.Scatter(x=freq_bins, y=magnitude, mode='lines', name="Frequency Magnitude")
    layout = go.Layout(title="Frequency Domain", xaxis_title="Frequency (Hz)", yaxis_title="Magnitude")
    return go.Figure(data=[trace], layout=layout)

def plot_magnitude_over_time(audio_data, sample_rate, freq, threshold):
    chunk_size = int(BIT_DURATION, sample_rate)
    time = np.linspace(0, len(audio_data) / sample_rate, num=len(audio_data))
    pilot_magnitudes = [get_frequency_magnitude(audio_data[i:i + chunk_size], freq) for i in range(0, len(audio_data) - chunk_size, PILOT_SEARCH_STEP, sample_rate)]
    chunk_times = time[::PILOT_SEARCH_STEP]
    
    trace = go.Scatter(x=chunk_times[:len(pilot_magnitudes)], y=pilot_magnitudes, mode='lines', name="Magnitude")
    
    threshold_trace = go.Scatter(
        x=chunk_times[:len(pilot_magnitudes)],
        y=[threshold] * len(pilot_magnitudes),
        mode='lines',
        name="Magnitude Threshold",
        line=dict(color='red', dash='dash') 
    )
    
    layout = go.Layout(
        title=f"{freq} Hz Magnitude",
        xaxis_title="Time (s)",
        yaxis_title="Magnitude"
    )
    
    return go.Figure(data=[trace, threshold_trace], layout=layout)



def get_highlighted_bits(binary_data, correct_binary):
    highlighted = ""
    for d_bit, c_bit in zip(binary_data, correct_binary):
        if d_bit == c_bit:
            highlighted += f"<span style='background-color: limegreen; color: white;'>{d_bit}</span>"
        else:
            highlighted += f"<span style='background-color: red; color: white;'>{d_bit}</span>"
    return highlighted
