import streamlit as st
from transformers import pipeline
import soundfile as sf
import io
import numpy as np

# Set page config
st.set_page_config(page_title="🎙️ AI Voice Generator", layout="centered")

# Custom CSS for better UI
st.markdown(
    """
    <style>
    .stTextArea textarea {font-size: 16px !important;}
    .stButton button {width: 100%; font-size: 18px; padding: 10px; border-radius: 10px;}
    .stAudio audio {width: 100% !important;}
    </style>
    """,
    unsafe_allow_html=True
)

@st.cache_resource
def load_tts_model():
    return pipeline("text-to-speech", model="facebook/mms-tts-eng")

def generate_audio(text, model):
    audio = model(text)
    return audio["audio"]

# App UI
st.title("🎙️ AI Text-to-Speech Generator")
st.write("🔊 Convert text into natural-sounding speech using AI.")

# User input
text = st.text_area("📝 Enter text:", placeholder="Type something...")

# Generate Button
if st.button("🎤 Generate Speech"):
    if text.strip():
        with st.spinner("Generating audio... 🎧"):
            tts = load_tts_model()
            audio = generate_audio(text, tts)

            # Convert to proper format
            audio_array = np.array(audio).squeeze()
            audio_array = (audio_array * 32767).astype(np.int16)

            # Save to buffer
            audio_buffer = io.BytesIO()
            sf.write(audio_buffer, audio_array, samplerate=22050, subtype='PCM_16', format='WAV')
            audio_buffer.seek(0)

            # Display Audio Player
            st.success("✅ Speech generated successfully!")
            st.audio(audio_buffer, format="audio/wav")

            # Provide download link
            st.download_button(
                label="⬇️ Download Audio",
                data=audio_buffer,
                file_name="generated_speech.wav",
                mime="audio/wav"
            )
    else:
        st.warning("⚠️ Please enter some text before generating.")

# Footer
st.markdown("---")
st.caption("🚀 Powered by Bhaumik Senwal 🤗")
