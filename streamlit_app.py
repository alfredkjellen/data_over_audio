import streamlit as st
from pathlib import Path
from translate import binary_to_text, file_to_binary
from demodulate import get_pilot_start, decode_fsk_signal, FREQUENCIES, MAGNITUDE_THRESHOLDS, BIT_DURATION, PILOT_DURATION
from plots import get_highlighted_bits, plot_frequency_domain, plot_magnitude_over_time, plot_time_domain

if __name__ == "__main__":
    audio_folder = Path("8f")

    st.title("Data Over Audio Demodulation")
    audio_files = list(audio_folder.glob("*.wav"))
    correct_bits = file_to_binary("text.txt")

    selected_audio = st.selectbox("Select an audio file", audio_files)

    if selected_audio:
        bits, audio_data, sample_rate = decode_fsk_signal(selected_audio)
        if bits:
            
            bits_str = ''.join(map(str, bits))
            highlighted_bits = get_highlighted_bits(bits_str, correct_bits)
            st.markdown(f"Decoded binary data: {highlighted_bits}", unsafe_allow_html=True)

            decoded_text = binary_to_text(bits)
            st.write(decoded_text)
            
            pilot_start = int(get_pilot_start(audio_data))
            pilot_chunk_size = int(PILOT_DURATION * sample_rate)
            start_pos = pilot_start + pilot_chunk_size

            st.plotly_chart(plot_time_domain(audio_data, sample_rate, BIT_DURATION, bits, start_pos, correct_bits))
            st.plotly_chart(plot_frequency_domain(audio_data, sample_rate))
            
            for freq in FREQUENCIES:
                st.plotly_chart(plot_magnitude_over_time(audio_data, sample_rate, freq, MAGNITUDE_THRESHOLDS[freq]))
        else:
            st.write("No data could be decoded from the selected file.")



