import streamlit as st
from transformers import pipeline
import soundfile as sf
import io
import numpy as np

@st.cache_resource
def load_tts_model():
    return pipeline("text-to-speech", model="facebook/mms-tts-eng")

def generate_audio(text, model):
    audio = model(text)
    return audio["audio"]

def main():
    st.title("TTS Generator")
    text = st.text_area("Enter text:")
    
    if st.button("Generate"):
        tts = load_tts_model()
        audio = generate_audio(text, tts)
        
        # Convert to proper format
        audio_array = np.array(audio).squeeze()
        audio_array = (audio_array * 32767).astype(np.int16)
        
        # Save to buffer
        audio_buffer = io.BytesIO()
        sf.write(
            audio_buffer,
            audio_array,
            samplerate=22050,
            subtype='PCM_16',
            format='WAV'
        )
        audio_buffer.seek(0)
        
        st.audio(audio_buffer, format="audio/wav")

if __name__ == "__main__":
    main()