# 🎙️ AI Text-to-Speech (TTS) Generator

## 🚀 Overview
This is a **Text-to-Speech (TTS) Generator** powered by **Facebook MMS-TTS** and **Hugging Face Transformers**. The app converts text into **natural-sounding speech** and allows users to **listen** and **download** the generated audio.

## ✨ Features
- 🎤 **Convert text to speech** using AI
- 🎧 **Audio player** to listen to the generated speech
- ⬇️ **Download speech as WAV file**
- 🖥 **Beautiful Streamlit UI** with interactive elements
- ⚡ **Fast processing** using pre-trained models



## 🛠️ Installation

### 1️⃣ Clone the Repository
```bash
[git clone https://github.com/Bhaumik-99/tts-HuggingFace/
cd tts-generator
```

### 2️⃣ Create Virtual Environment (Optional)
```bash
python -m venv myenv
source myenv/bin/activate  # On Windows: myenv\Scripts\activate
```

### 3️⃣ Install Dependencies
```bash
pip install -r requirements.txt
```

## ▶️ Run the App
```bash
streamlit run app.py
```

## 🛠️ Technologies Used
- **Python** 🐍
- **Streamlit** 🎨 (for UI)
- **Hugging Face Transformers** 🤗 (for AI model)
- **Facebook MMS-TTS** 🗣 (for speech generation)
- **NumPy & SoundFile** 🔊 (for audio processing)

## 📜 Usage Guide
1. Enter text in the input box
2. Click **"Generate Speech"**
3. Listen to the generated speech
4. Download the **WAV** file if needed

## 📌 Notes
- Ensure you have a stable internet connection for model loading.
- You can modify the model in `load_tts_model()` if needed.

## 🤝 Contributing
Want to improve this project? Feel free to **fork** the repo and submit a **pull request**! 🚀

## 📜 License
This project is licensed under the **MIT License**.

---
🚀 Built with ❤️ using AI & Streamlit!

