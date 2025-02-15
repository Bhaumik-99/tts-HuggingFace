import streamlit as st
from transformers import pipeline
import soundfile as sf
import io
import numpy as np

@st.cache_resource
def load_tts_model():
    return pipeline("text-to-speech", model="facebook/mms-tts-eng")

def generate_audio(text, model):
    audio = model(text)  # Restore default speech speed
    return audio["audio"]

def main():
    st.set_page_config(page_title="AI Text-to-Speech", page_icon="🔊", layout="wide")
    
    st.markdown("""
        <h1 style='text-align: center;'>🔊 AI Text-to-Speech Generator</h1>
        <p style='text-align: center;'>🎤 Enter text to convert into speech. Note: Numbers are not allowed.</p>
    """, unsafe_allow_html=True)
    
    st.markdown("""
        <div style='text-align: center;'>
            <b>Features:</b><br>
            🎙️ Converts text into high-quality speech<br>
            🚫 Numbers are not allowed<br>
        </div>
    """, unsafe_allow_html=True)
    
    text = st.text_area("✍️ Enter your text here:", height=150)
    
    if any(char.isdigit() for char in text):
        st.error("❌ Numbers are not allowed. Please enter text without numbers.")
        return
    
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        if st.button("🎶 Generate Speech", use_container_width=True):
            with st.spinner("🎛️ Processing your speech..."):
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
                st.success("✅ Speech generation complete! 🎉")
                
                st.download_button(
                    label="📥 Download Audio",
                    data=audio_buffer,
                    file_name="speech.wav",
                    mime="audio/wav"
                )

if __name__ == "__main__":
    main()

