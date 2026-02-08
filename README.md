# 🎙️ Audio Transcription & Quality Check Tool

A comprehensive local audio transcription and NLP analysis tool built for the **Natural Language Processing (CS438)** course project.

## ✨ Features

### 📥 Input Sources
- **YouTube URL** - Download audio directly from YouTube
- **Audio Upload** - MP3, WAV, M4A, FLAC, OGG
- **Video Upload** - MP4, AVI, MOV, MKV, WEBM (auto-extracts audio)

### 📝 Transcription
- **Whisper AI** - OpenAI's state-of-the-art speech recognition
- **Multiple Models** - tiny, base, small, medium, large, large-v3
- **GPU Support** - Automatic CUDA detection
- **Export Formats** - TXT, SRT subtitles, JSON

### 📊 Quality Check
- **Text Statistics** - Word count, character count, sentence count
- **Profanity Detection** - Identifies inappropriate language
- **Filler Word Analysis** - Detects um, uh, like, you know, etc.
- **Pass/Fail Status** - Based on profanity and filler word ratio

### 🔬 Advanced NLP Analysis
- **😊 Sentiment Analysis** - Polarity and subjectivity scoring (TextBlob)
- **⚠️ Toxicity Detection** - Rule-based toxic content detection
- **🏷️ Named Entity Recognition** - Identifies people, organizations, places (spaCy)
- **🔑 Keyword Extraction** - TF-IDF based key phrase extraction
- **📝 Text Summarization** - Extractive summarization
- **📊 Readability Scores** - Flesch Reading Ease, Flesch-Kincaid Grade, Gunning Fog, SMOG

### 💰 AI Monetization Advice (Gemini)
- **2-Line Summary** - Quick overview of content
- **Words to Avoid** - Demonetization-risk words with safe alternatives
- **Monetization Tips** - Actionable advice for YouTube monetization

---

## 🛠️ Technology Stack

| Component | Technology |
|-----------|------------|
| UI Framework | Streamlit |
| Speech-to-Text | Faster-Whisper (OpenAI Whisper) |
| YouTube Download | yt-dlp |
| Audio Processing | FFmpeg |
| Sentiment Analysis | TextBlob |
| Named Entity Recognition | spaCy (en_core_web_sm) |
| Keyword Extraction | scikit-learn (TF-IDF) |
| AI Advice | Google Gemini API |

---

## 📁 Project Structure

```
audio-transcription-local/
├── app.py                 # Main Streamlit UI (entry point)
├── transcriber.py         # Whisper transcription + quality check
├── audio_utils.py         # YouTube download + audio extraction
├── nlp_analysis.py        # All NLP analysis functions
├── model_manager.py       # Whisper model caching
├── download_models.py     # Pre-download models script
├── requirements.txt       # Python dependencies
├── downloads/             # Downloaded YouTube videos
└── .venv/                 # Python virtual environment
```

---

## 📋 System Requirements

- Python 3.10+
- FFmpeg (for audio extraction)
- 8GB+ RAM (16GB+ recommended for larger models)
- GPU optional but recommended

---

## 🔧 Installation

### 1. Install FFmpeg

**Windows:**
```bash
# Download from https://ffmpeg.org/download.html
# Extract to C:\ffmpeg\ and add to PATH
# Or install via Chocolatey:
choco install ffmpeg
```

**macOS:**
```bash
brew install ffmpeg
```

**Linux:**
```bash
sudo apt-get install ffmpeg
```

### 2. Create Virtual Environment

```bash
# Create virtual environment
python -m venv .venv

# Activate it
# Windows:
.venv\Scripts\activate

# macOS/Linux:
source .venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt

# Download spaCy English model
python -m spacy download en_core_web_sm
```

### 4. Pre-download Whisper Models (Optional)

```bash
python download_models.py
```

---

## 🚀 Running the Application

```bash
streamlit run app.py
```

Open in browser: `http://localhost:8501`

---

## 📖 Usage Guide

### Tab 1: Input
1. Choose input method (YouTube URL / Audio Upload / Video Upload)
2. Click **Transcribe** to process

### Tab 2: Transcription
- View full transcription text
- Download as TXT, SRT, or JSON

### Tab 3: Quality Check
1. View text statistics and quality assessment
2. Click **▶️ Run Analysis** for advanced NLP features
3. Click **🤖 Get AI Advice** for monetization tips

---

## 🔄 Pipeline Flow

```
Audio/Video Input → FFmpeg → MP3 → Whisper → Text
                                              ↓
                                    ┌─────────┴─────────┐
                                    ↓                   ↓
                            Quality Check         NLP Analysis
                            (profanity/filler)    (sentiment/NER/etc)
                                    ↓                   ↓
                                    └─────────┬─────────┘
                                              ↓
                                    Gemini AI Advice
                                              ↓
                                    Final Report/Downloads
```

---

## 👨‍🎓 Course Information

**Course:** Natural Language Processing (CS438)  
**Semester:** VII  
**Institution:** UMT, Lahore

---

## 📄 License

This project is for educational purposes only.
