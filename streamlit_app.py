import streamlit as st
from pathlib import Path
from translate import binary_to_text, file_to_binary
from demodulate import find_pilot_signal, decode_fsk_signal, PILOT_CHUNK_SIZE, FREQUENCIES, MAGNITUDE_THRESHOLDS
from plots import highlight_binary_data, plot_frequency_domain, plot_magnitude_over_time, plot_time_domain

if __name__ == "__main__":
    audio_folder = Path("signal")

    st.title("Data Over Audio Demodulation")
    audio_files = list(audio_folder.glob("*.wav"))
    actual_binary = file_to_binary("text.txt")

    selected_audio = st.selectbox("Select an audio file", audio_files)

    if selected_audio:
        binary_data, audio_data, sample_rate = decode_fsk_signal(selected_audio)
        if binary_data:
            
            binary_data_str = ''.join(map(str, binary_data))
            highlighted_binary_data = highlight_binary_data(binary_data_str, actual_binary)
            st.markdown(f"Decoded binary data: {highlighted_binary_data}", unsafe_allow_html=True)

            decoded_text = binary_to_text(binary_data)
            st.write(decoded_text)
            
            pilot_start = int(find_pilot_signal(audio_data))
            start_pos = pilot_start + PILOT_CHUNK_SIZE

            st.plotly_chart(plot_time_domain(audio_data, sample_rate, binary_data, start_pos, actual_binary))
            st.plotly_chart(plot_frequency_domain(audio_data, sample_rate))
            
            for freq in FREQUENCIES:
                st.plotly_chart(plot_magnitude_over_time(audio_data, sample_rate, freq, MAGNITUDE_THRESHOLDS[freq]))
        else:
            st.write("No data could be decoded from the selected file.")



