# Audio Transcription & Quality Check Tool
## Project Report - Natural Language Processing (CS438)

---

**Course:** Natural Language Processing (CS438)  
**Semester:** VII  
**Institution:** Aspire Group of Colleges  
**Project Title:** Audio Transcription & Quality Check Tool with Advanced NLP Analysis  

---

# Table of Contents

1. [Introduction](#1-introduction)
2. [System Architecture & Pipeline](#2-system-architecture--pipeline)
3. [Audio Processing Module](#3-audio-processing-module)
4. [Speech-to-Text Transcription (Whisper)](#4-speech-to-text-transcription-whisper)
5. [SRT Subtitle Generation](#5-srt-subtitle-generation)
6. [Quality Check Module](#6-quality-check-module)
7. [Natural Language Processing Techniques](#7-natural-language-processing-techniques)
8. [AI Integration (Google Gemini API)](#8-ai-integration-google-gemini-api)
9. [User Interface (Streamlit)](#9-user-interface-streamlit)
10. [Conclusion & Future Work](#10-conclusion--future-work)

---

# 1. Introduction

## 1.1 Project Overview

This project is a web-based tool that helps users convert speech from audio or video files into written text. Users can upload audio files, video files, or paste a YouTube link, and the system will automatically extract the audio and convert it to text using OpenAI's Whisper model. Once the text is ready, the tool checks the content for bad words and filler words like "um" and "like" to see if the content is suitable for platforms like YouTube. The tool also runs several Natural Language Processing (NLP) techniques on the text, including sentiment analysis to detect if the tone is positive or negative, named entity recognition to find names of people and places, keyword extraction to identify important topics, and readability scoring to check how easy the text is to understand. Finally, the tool connects to Google's Gemini AI to give users smart advice on how to make their videos more suitable for monetization. The entire application runs in a simple web interface built with Streamlit, making it easy to use without any technical knowledge.

## 1.2 Theoretical Context: The Evolution of Speech and Language Technology

### 1.2.1 Historical Perspective

The journey from speech to text processing spans decades of research:

| Era | Technology | Capability |
|-----|------------|------------|
| 1950s | Rule-based systems | 10 words, single speaker |
| 1970s | Hidden Markov Models | Limited vocabulary |
| 1990s | Statistical models | Dictation software |
| 2010s | Deep Learning | Consumer virtual assistants |
| 2020s | Transformers (Whisper) | Human-level accuracy |

### 1.2.2 The Convergence of Technologies

Modern content analysis requires integration of multiple AI disciplines:

```
                    CONTENT ANALYSIS
                          │
    ┌─────────────────────┼─────────────────────┐
    │                     │                     │
    ▼                     ▼                     ▼
┌─────────┐         ┌─────────┐         ┌─────────┐
│  ASR    │         │   NLP   │         │   LLM   │
│ Speech  │    →    │  Text   │    →    │  Smart  │
│ to Text │         │ Analysis│         │ Advice  │
└─────────┘         └─────────┘         └─────────┘
```

### 1.2.3 Why This Project Matters

Content creation has exploded with platforms like YouTube, TikTok, and podcasts. Creators need:
- **Accessibility**: Subtitles for hearing-impaired audiences
- **Discoverability**: Text for search engine indexing
- **Quality Control**: Automated content moderation
- **Monetization**: Platform compliance for ad revenue

## 1.3 Problem Statement

Content creators and researchers often need to:
1. Transcribe audio/video content accurately
2. Identify inappropriate or problematic language
3. Analyze the sentiment and tone of their content
4. Understand key topics and entities mentioned
5. Get actionable advice for content improvement

## 1.3 Objectives

| Objective | Description |
|-----------|-------------|
| Accurate Transcription | Use Whisper AI for high-quality speech-to-text |
| Quality Analysis | Detect profanity, filler words, and quality issues |
| NLP Analysis | Apply sentiment, NER, keyword extraction, readability scoring |
| AI Advice | Use Gemini API for intelligent content recommendations |
| User-Friendly Interface | Build intuitive Streamlit web application |

## 1.4 Technologies Used

| Component | Technology | Purpose |
|-----------|------------|---------|
| UI Framework | Streamlit | Web interface |
| Speech-to-Text | Faster-Whisper | Audio transcription |
| Audio Processing | FFmpeg, yt-dlp | Audio extraction/download |
| Sentiment Analysis | TextBlob | Polarity/subjectivity scoring |
| Named Entity Recognition | spaCy | Entity extraction |
| Keyword Extraction | scikit-learn | TF-IDF analysis |
| AI Advice | Google Gemini | Content recommendations |

---

# 2. System Architecture & Pipeline

## 2.0 Theoretical Background: Software Architecture

### 2.0.1 What is Software Architecture?

**Software Architecture** defines the high-level structure of a system, including:
- Components and their responsibilities
- Interactions between components
- Design decisions and trade-offs

### 2.0.2 Layered Architecture Pattern

Our system uses a **Layered Architecture** (also called N-tier):

```
┌─────────────────────────────────────────┐
│         PRESENTATION LAYER              │  ← User Interface
├─────────────────────────────────────────┤
│         APPLICATION LAYER               │  ← Business Logic
├─────────────────────────────────────────┤
│           SERVICE LAYER                 │  ← Reusable Services
├─────────────────────────────────────────┤
│          DATA ACCESS LAYER              │  ← File/API Access
└─────────────────────────────────────────┘
```

**Benefits:**
- **Separation of Concerns**: Each layer has one responsibility
- **Maintainability**: Changes in one layer don't affect others
- **Testability**: Layers can be tested independently
- **Reusability**: Lower layers can be reused

### 2.0.3 Pipeline Architecture Pattern

For data processing, we use a **Pipeline Architecture**:

```
Input → Stage 1 → Stage 2 → Stage 3 → ... → Output
         ↓          ↓          ↓
       Filter    Transform   Enrich
```

**Characteristics:**
- Each stage transforms data
- Output of one stage is input to next
- Stages are independent and composable
- Easy to add/remove stages

### 2.0.4 Modular Design Principles

**Single Responsibility Principle:**
- Each module does one thing well
- `audio_utils.py` → Audio processing only
- `transcriber.py` → Transcription only
- `nlp_analysis.py` → NLP only

**Loose Coupling:**
- Modules communicate through defined interfaces
- Changes in one module don't break others
- Easy to replace implementations

**High Cohesion:**
- Related functionality stays together
- All sentiment functions in `nlp_analysis.py`
- All audio functions in `audio_utils.py`

### 2.0.5 Data Flow Design

**ETL Pattern** (Extract, Transform, Load):
```
EXTRACT          TRANSFORM              LOAD
───────────      ──────────────         ─────────
Audio File  →    Whisper ASR      →     Text Store
YouTube URL →    NLP Analysis     →     Results Display
Video File  →    AI Enhancement   →     Download Files
```

## 2.1 High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                         USER INPUT LAYER                            │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐              │
│  │ YouTube URL  │  │ Audio File   │  │ Video File   │              │
│  └──────────────┘  └──────────────┘  └──────────────┘              │
└─────────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────────┐
│                     AUDIO PROCESSING LAYER                          │
│  • yt-dlp (YouTube download)                                        │
│  • FFmpeg (audio extraction from video)                             │
│  • Audio normalization and format conversion                        │
└─────────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────────┐
│                   TRANSCRIPTION LAYER (Whisper)                     │
│  • Faster-Whisper model (tiny/base/small/medium/large)             │
│  • Voice Activity Detection (VAD) filter                           │
│  • Word-level timestamps                                            │
│  • Language detection                                               │
└─────────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────────┐
│                     NLP ANALYSIS LAYER                              │
│  ┌─────────────┐ ┌─────────────┐ ┌─────────────┐ ┌─────────────┐   │
│  │ Sentiment   │ │ Toxicity    │ │ NER         │ │ Keywords    │   │
│  │ (TextBlob)  │ │ Detection   │ │ (spaCy)     │ │ (TF-IDF)    │   │
│  └─────────────┘ └─────────────┘ └─────────────┘ └─────────────┘   │
│  ┌─────────────┐ ┌─────────────┐ ┌─────────────┐                   │
│  │ Readability │ │ Summary     │ │ Quality     │                   │
│  │ Scores      │ │ Extraction  │ │ Check       │                   │
│  └─────────────┘ └─────────────┘ └─────────────┘                   │
└─────────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────────┐
│                     AI ADVISORY LAYER (Gemini)                      │
│  • Content summarization                                            │
│  • Word replacement suggestions                                     │
│  • Monetization advice                                              │
└─────────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────────┐
│                        OUTPUT LAYER                                 │
│  • TXT (plain text)                                                 │
│  • SRT (subtitles with timestamps)                                  │
│  • JSON (structured metadata)                                       │
│  • Quality Report                                                   │
└─────────────────────────────────────────────────────────────────────┘
```

## 2.2 Data Flow

1. **Input Stage**: User provides YouTube URL, audio file, or video file
2. **Extraction Stage**: Audio is extracted/downloaded in MP3 format
3. **Transcription Stage**: Whisper model converts speech to text with timestamps
4. **Analysis Stage**: Multiple NLP techniques analyze the transcribed text
5. **AI Stage**: Gemini API provides intelligent recommendations
6. **Output Stage**: Results displayed in UI and available for download

## 2.3 Module Dependencies

```python
# Core dependencies and their roles
streamlit          # Web UI framework
faster-whisper     # Speech-to-text (Whisper implementation)
yt-dlp            # YouTube audio download
pandas            # Data manipulation
numpy             # Numerical operations
textblob          # Sentiment analysis
spacy             # Named Entity Recognition
scikit-learn      # TF-IDF keyword extraction
google-generativeai  # Gemini API integration
```

---

# 3. Audio Processing Module

## 3.1 Overview

The audio processing module (`audio_utils.py`) handles all audio-related operations including YouTube downloads and video-to-audio conversion.

## 3.2 Theoretical Background: Digital Audio

### 3.2.1 What is Digital Audio?

Digital audio is the representation of sound as a sequence of discrete numerical values. When sound waves (analog signals) are converted to digital format, two key processes occur:

1. **Sampling**: The continuous audio wave is measured at regular intervals (sample rate). Common sample rates include:
   - 44.1 kHz (CD quality) - 44,100 samples per second
   - 48 kHz (Professional video)
   - 16 kHz (Speech recognition - sufficient for human voice)

2. **Quantization**: Each sample's amplitude is converted to a discrete value (bit depth):
   - 16-bit: 65,536 possible values
   - 24-bit: 16.7 million possible values

### 3.2.2 Audio Codecs and Compression

Audio files use codecs (coder-decoder) to compress data:

| Codec | Type | Quality | File Size | Use Case |
|-------|------|---------|-----------|----------|
| WAV | Uncompressed | Highest | Large | Professional editing |
| FLAC | Lossless | High | Medium | Archival |
| MP3 | Lossy | Good | Small | General use |
| AAC | Lossy | Better | Small | Streaming |

**Why MP3 for Transcription?**
- Smaller file size (faster upload/processing)
- 192 kbps preserves speech clarity
- Universally supported
- Whisper optimized for compressed audio

### 3.2.3 Audio Extraction Theory

Video files are **containers** (MP4, AVI, MKV) that hold multiple streams:
- Video stream (H.264, H.265)
- Audio stream (AAC, MP3)
- Subtitle stream
- Metadata

Extraction involves:
1. **Demuxing**: Separating streams from the container
2. **Decoding**: Converting compressed audio to raw PCM
3. **Re-encoding**: Converting to target format (MP3)

## 3.3 FFmpeg Integration

FFmpeg is an open-source multimedia framework for handling audio/video. The system searches for FFmpeg in multiple locations:

```python
def find_ffmpeg():
    """Find ffmpeg in common Windows locations or system PATH"""
    paths = [
        'C:\\ffmpeg\\ffmpeg-8.0.1-essentials_build\\bin\\ffmpeg.exe',
        'C:\\ffmpeg\\bin\\ffmpeg.exe',
    ]
    for p in paths:
        if Path(p).exists():
            return str(p)
    
    # Fall back to system PATH
    ffmpeg = shutil.which('ffmpeg')
    if ffmpeg:
        return ffmpeg
    
    return 'ffmpeg'  # Default
```

## 3.3 Audio Extraction from Video

When a user uploads a video file, audio is extracted using FFmpeg:

```python
def extract_audio(video_path, output_dir=None, bitrate_k='160'):
    """Extract audio from video file"""
    output_path = output_dir / f"{video_path.stem}.mp3"
    
    cmd = [
        FFMPEG, '-y',           # Overwrite output
        '-i', str(video_path),  # Input video
        '-vn',                  # No video
        '-acodec', 'libmp3lame', # MP3 codec
        '-ab', f'{bitrate_k}k', # Bitrate
        '-ac', '2',             # Stereo
        str(output_path)
    ]
    
    subprocess.run(cmd, check=True, capture_output=True)
    return output_path
```

**Parameters Explained:**
- `-y`: Automatically overwrite output files
- `-i`: Input file path
- `-vn`: Disable video stream (extract audio only)
- `-acodec libmp3lame`: Use LAME MP3 encoder
- `-ab 160k`: Audio bitrate of 160 kbps
- `-ac 2`: Stereo audio (2 channels)

## 3.4 YouTube Download

YouTube audio is downloaded directly as MP3 using yt-dlp:

```python
def download_youtube_video(url, output_dir=None):
    """Download audio (MP3) from YouTube URL"""
    import yt_dlp
    
    ydl_opts = {
        'format': 'bestaudio/best',  # Best audio quality
        'outtmpl': str(output_dir / '%(title)s.%(ext)s'),
        'postprocessors': [{
            'key': 'FFmpegExtractAudio',
            'preferredcodec': 'mp3',
            'preferredquality': '192',
        }],
    }
    
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(url, download=True)
        audio_path = output_dir / f"{info['title']}.mp3"
        return audio_path
```

**How yt-dlp Works:**
1. Extracts video metadata from YouTube
2. Downloads the best available audio stream
3. Uses FFmpeg postprocessor to convert to MP3
4. Returns the path to the downloaded audio file

---

# 4. Speech-to-Text Transcription (Whisper)

## 4.1 Theoretical Background: Automatic Speech Recognition (ASR)

### 4.1.1 What is Speech Recognition?

Automatic Speech Recognition (ASR) is the technology that converts spoken language into written text. This is a complex problem because:

1. **Acoustic Variability**: Same words sound different due to:
   - Speaker differences (age, gender, accent)
   - Speaking speed and emotion
   - Background noise and recording quality

2. **Linguistic Ambiguity**: 
   - Homophones: "their" vs "there" vs "they're"
   - Word boundaries: "ice cream" vs "I scream"
   - Context-dependent meaning

### 4.1.2 Traditional ASR Pipeline

Classical speech recognition used a multi-stage pipeline:

```
Audio → Feature Extraction → Acoustic Model → Language Model → Text
         (MFCC)              (HMM/GMM)         (N-gram)
```

1. **Feature Extraction (MFCC)**: 
   - Mel-Frequency Cepstral Coefficients
   - Converts audio into frequency features mimicking human ear perception
   - 13-40 coefficients per 25ms frame

2. **Acoustic Model**:
   - Maps acoustic features to phonemes (smallest sound units)
   - Traditional: Hidden Markov Models (HMM) + Gaussian Mixture Models (GMM)
   - Modern: Deep Neural Networks

3. **Language Model**:
   - Determines likely word sequences
   - N-gram models: P(word | previous N-1 words)
   - Helps disambiguate similar-sounding phrases

### 4.1.3 Modern End-to-End ASR (Whisper's Approach)

Whisper uses an **end-to-end Transformer architecture** that directly maps audio to text without separate components:

```
Audio Waveform → Mel Spectrogram → Encoder → Decoder → Text
                    (80 channels)   (Transformer)  (Autoregressive)
```

**Key Advantages:**
- Single model learns all components jointly
- No handcrafted features needed
- Better generalization across languages
- Handles punctuation and formatting

### 4.1.4 Transformer Architecture in Whisper

The Transformer architecture (Vaswani et al., 2017) revolutionized NLP:

**Encoder** (processes audio):
- Converts mel spectrogram to hidden representations
- Uses **self-attention** to capture long-range dependencies
- Multiple layers with feed-forward networks

**Decoder** (generates text):
- Autoregressive: generates one token at a time
- Uses **cross-attention** to focus on relevant audio parts
- Predicts probability distribution over vocabulary

**Self-Attention Mechanism:**
```
Attention(Q, K, V) = softmax(QK^T / √d_k) × V
```
- Q (Query): What am I looking for?
- K (Key): What do I contain?
- V (Value): What information do I provide?

### 4.1.5 Whisper's Training Data

Whisper was trained on 680,000 hours of labeled audio:
- Web-scraped audio with transcripts
- 96+ languages
- Diverse accents and recording conditions
- Weak supervision (not perfectly labeled)

This massive dataset enables robust performance without fine-tuning.

## 4.2 Whisper Model Overview

OpenAI's Whisper is a state-of-the-art automatic speech recognition (ASR) system trained on 680,000 hours of multilingual data. We use `faster-whisper`, an optimized implementation using CTranslate2.

## 4.2 Model Sizes and Characteristics

| Model | Parameters | VRAM | Speed | Accuracy |
|-------|------------|------|-------|----------|
| tiny | 39M | ~1GB | Fastest | Lower |
| base | 74M | ~1GB | Fast | Good |
| small | 244M | ~2GB | Medium | Better |
| medium | 769M | ~5GB | Slow | High |
| large | 1.5B | ~10GB | Slowest | Highest |
| large-v3 | 1.5B | ~10GB | Slowest | Best |

## 4.3 Transcription Implementation

```python
def transcribe_audio_local(
    audio_path: Path,
    model_size: str = "tiny",
    device: str = "auto",
    compute_type: str = "float16",
    vad_filter: bool = True,
) -> Dict[str, Any]:
    """Transcribe audio locally using faster-whisper"""
    
    from faster_whisper import WhisperModel
    
    # Device selection (GPU if available)
    if device == "auto":
        import torch
        dev = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Load model with caching
    model = WhisperModel(
        model_size, 
        device=dev, 
        compute_type=compute_type,
        download_root=str(MODEL_CACHE_DIR)
    )
    
    # Run transcription
    segments, info = model.transcribe(
        str(audio_path),
        vad_filter=vad_filter,      # Voice Activity Detection
        word_timestamps=True,        # Word-level timing
        language=None,              # Auto-detect language
    )
    
    return process_segments(segments)
```

## 4.4 Voice Activity Detection (VAD)

VAD filter removes silence and non-speech segments:

```python
# Transcription with VAD
segments, info = model.transcribe(
    audio_path,
    vad_filter=True,  # Enable VAD
    vad_parameters=dict(
        min_silence_duration_ms=500,  # Min silence to split
        speech_pad_ms=200,            # Padding around speech
    )
)
```

**Benefits of VAD:**
- Removes silent segments (faster processing)
- Improves transcription accuracy
- Reduces processing time by up to 50%

## 4.5 Word-Level Timestamps

Whisper provides precise timing for each word:

```python
for segment in segments:
    for word in segment.words:
        print(f"Word: {word.word}")
        print(f"Start: {word.start}s")
        print(f"End: {word.end}s")
        print(f"Confidence: {word.probability}")
```

---

# 5. SRT Subtitle Generation

## 5.1 Theoretical Background: Subtitle Formats

### 5.1.1 What are Subtitles?

Subtitles are textual representations of spoken dialogue synchronized with video/audio. They serve multiple purposes:

1. **Accessibility**: For deaf/hard-of-hearing viewers
2. **Translation**: Foreign language content
3. **Clarity**: Noisy environments, unclear audio
4. **SEO**: Search engines can index subtitle text

### 5.1.2 Types of Subtitle Formats

| Format | Extension | Features | Use Case |
|--------|-----------|----------|----------|
| SRT | .srt | Simple, widely supported | Universal |
| VTT | .vtt | Web standard, styling | HTML5 video |
| ASS/SSA | .ass | Advanced styling, effects | Anime |
| TTML | .xml | Broadcast standard | TV/Streaming |

### 5.1.3 Time Synchronization Theory

Subtitle timing requires precise synchronization with audio:

**Key Concepts:**
- **In-point**: When subtitle appears (start time)
- **Out-point**: When subtitle disappears (end time)
- **Duration**: Time subtitle is visible
- **Reading Speed**: ~150-200 words per minute

**Timing Best Practices:**
- Minimum duration: 1 second
- Maximum duration: 7 seconds
- Gap between subtitles: 0.1-0.5 seconds
- Maximum 2 lines, 42 characters per line

### 5.1.4 Timestamp Precision

Timestamps use hierarchical time units:
```
Hours : Minutes : Seconds , Milliseconds
  HH  :   MM    :   SS    ,    mmm
```

**Why Milliseconds Matter:**
- Human perception: ~50ms difference is noticeable
- Speech rate: ~3-4 words per second
- Single word duration: 250-500ms average

## 5.2 SRT Format Specification

SRT (SubRip Subtitle) is a standard subtitle format:

```
1
00:00:00,000 --> 00:00:02,500
Hello, welcome to this video.

2
00:00:02,500 --> 00:00:05,000
Today we will discuss NLP.
```

**Structure:**
1. **Index**: Sequential number starting from 1
2. **Timestamp**: Start --> End in HH:MM:SS,mmm format
3. **Text**: The subtitle content
4. **Blank line**: Separator between entries

## 5.2 SRT Generation Implementation

```python
def _format_ts(t: float) -> str:
    """Format timestamp for SRT format"""
    h = int(t // 3600)           # Hours
    m = int((t % 3600) // 60)    # Minutes
    s = int(t % 60)              # Seconds
    ms = int((t - math.floor(t)) * 1000)  # Milliseconds
    return f"{h:02}:{m:02}:{s:02},{ms:03}"

# Generate SRT from segments
srt_lines = []
for idx, segment in enumerate(segments, 1):
    start_ts = _format_ts(segment.start)
    end_ts = _format_ts(segment.end)
    
    srt_lines.append(f"{idx}")
    srt_lines.append(f"{start_ts} --> {end_ts}")
    srt_lines.append(segment.text.strip())
    srt_lines.append("")  # Blank line separator

srt_content = "\n".join(srt_lines)
```

## 5.3 Timestamp Calculation

Converting float seconds to SRT format:

```python
# Example: 3661.5 seconds
t = 3661.5

hours = int(3661.5 // 3600)      # = 1 hour
remaining = 3661.5 % 3600        # = 61.5 seconds
minutes = int(61.5 // 60)        # = 1 minute
seconds = int(61.5 % 60)         # = 1 second
milliseconds = int(0.5 * 1000)   # = 500 ms

# Result: "01:01:01,500"
```

---

# 6. Quality Check Module

## 6.1 Theoretical Background: Text Quality Analysis

### 6.1.1 Why Quality Checking Matters

Content quality analysis is essential for:
1. **Platform Compliance**: YouTube, TikTok have content policies
2. **Monetization**: Inappropriate content affects ad revenue
3. **Audience Appropriateness**: Age-appropriate content
4. **Professional Standards**: Business/educational content

### 6.1.2 Types of Quality Issues

| Category | Examples | Impact |
|----------|----------|--------|
| Profanity | Curse words, slurs | Demonetization |
| Filler Words | "um", "like", "basically" | Reduced clarity |
| Repetition | Repeated phrases | Poor engagement |
| Grammar | Incorrect transcription | Misunderstanding |

### 6.1.3 Text Pattern Matching Theory

Pattern matching is fundamental to quality checking:

**String Matching Approaches:**
1. **Exact Match**: `word in text` - Simple but catches partial matches
2. **Regular Expressions**: Pattern-based matching with boundaries
3. **Fuzzy Matching**: Handles typos, variations (Levenshtein distance)
4. **Phonetic Matching**: Catches intentional misspellings (Soundex)

### 6.1.4 Regular Expressions (Regex) Theory

Regular expressions define search patterns using special characters:

| Pattern | Meaning | Example |
|---------|---------|---------|
| `\b` | Word boundary | `\bcat\b` matches "cat" not "category" |
| `\w` | Word character | `[a-zA-Z0-9_]` |
| `+` | One or more | `a+` matches "a", "aa", "aaa" |
| `*` | Zero or more | `ab*` matches "a", "ab", "abb" |
| `[]` | Character class | `[aeiou]` matches any vowel |
| `\s` | Whitespace | Space, tab, newline |

**Word Boundary (`\b`) Deep Dive:**
```
Text: "class is in session"
Pattern: \bass\b

Position analysis:
- "cl|ass" - \b matches between 'l' and 'a'? No, both are \w
- "ass| is" - \b matches between 's' and ' '? Yes!
- But "ass" is part of "class", not standalone

Result: No match (correct behavior!)
```

### 6.1.5 Filler Word Linguistics

Filler words (discourse markers) serve linguistic functions:

**Types of Fillers:**
1. **Hesitation Markers**: "um", "uh" - thinking pause
2. **Discourse Markers**: "like", "you know" - maintaining conversation
3. **Hedges**: "basically", "actually" - softening statements
4. **Intensifiers**: "really", "literally" - emphasis (often overused)

**Why They're Problematic in Content:**
- Reduce perceived expertise
- Lower engagement metrics
- Indicate unscripted/unprepared content
- Affect accessibility (harder to follow)

**Acceptable Thresholds:**
- Professional content: <5% filler ratio
- Casual content: <10% filler ratio
- Conversation: 10-15% (natural)

## 6.2 Overview

The quality check module analyzes transcribed text for various quality metrics and potential issues.

## 6.2 Text Statistics

```python
def check_transcription_quality(transcription: str) -> Dict:
    """Analyze transcription quality"""
    
    # Word count
    words = transcription.split()
    word_count = len(words)
    
    # Character count
    char_count = len(transcription)
    
    # Sentence count (using punctuation)
    sentences = re.split(r'[.!?]+', transcription)
    sentence_count = len([s for s in sentences if s.strip()])
    
    # Average word length
    avg_word_length = char_count / max(word_count, 1)
    
    return {
        "word_count": word_count,
        "char_count": char_count,
        "sentence_count": sentence_count,
        "avg_word_length": avg_word_length
    }
```

## 6.3 Profanity Detection

Profanity is detected using pattern matching:

```python
# Profanity word list
profanity_list = [
    "shit", "fuck", "fucking", "damn", 
    "hell", "ass", "bitch", "bastard"
]

text_lower = transcription.lower()
found_profanity = []

for word in profanity_list:
    # Use regex for word boundary matching
    pattern = r'\b' + re.escape(word) + r'\b'
    matches = re.findall(pattern, text_lower)
    
    if matches:
        count = len(matches)
        found_profanity.append(f"{word} ({count}x)")
```

**Word Boundary Matching (`\b`):**
- `\b` matches the position between a word character and non-word character
- Prevents matching "class" when searching for "ass"
- Example: `\bass\b` matches "ass" but not "class" or "assume"

## 6.4 Filler Word Detection

Filler words indicate speech disfluency:

```python
filler_words = [
    "like", "just", "really", "actually", 
    "basically", "literally", "you know", "i mean"
]

filler_count = 0
for filler in filler_words:
    if ' ' in filler:  # Multi-word filler
        count = text_lower.count(filler)
    else:
        count = len(re.findall(r'\b' + re.escape(filler) + r'\b', text_lower))
    filler_count += count

# Calculate filler ratio
filler_ratio = (filler_count / word_count) * 100
```

## 6.5 Quality Pass/Fail Logic

```python
# Only fail if:
# 1. Profanity detected, OR
# 2. Filler ratio > 10%

has_critical_issues = (
    quality["profanity_found"] or 
    quality["filler_ratio_percent"] > 10
)

if has_critical_issues:
    status = "FAILED"
else:
    status = "PASSED"
```

---

# 7. Natural Language Processing Techniques

## 7.0 Theoretical Foundation: What is NLP?

### 7.0.1 Definition and Scope

**Natural Language Processing (NLP)** is a subfield of artificial intelligence that focuses on the interaction between computers and human language. It combines:

- **Linguistics**: Understanding language structure and meaning
- **Computer Science**: Algorithms and data structures
- **Machine Learning**: Learning patterns from data
- **Statistics**: Probabilistic models

### 7.0.2 Levels of Language Analysis

NLP operates at multiple linguistic levels:

| Level | Focus | Example Task |
|-------|-------|--------------|
| **Phonology** | Sound patterns | Speech recognition |
| **Morphology** | Word structure | Stemming, lemmatization |
| **Syntax** | Sentence structure | Parsing, POS tagging |
| **Semantics** | Meaning | Word sense disambiguation |
| **Pragmatics** | Context/intent | Sentiment, sarcasm detection |
| **Discourse** | Multi-sentence | Summarization, coherence |

### 7.0.3 Key NLP Tasks

```
                    ┌─────────────────────────────────────┐
                    │         NLP TASK TAXONOMY           │
                    └─────────────────────────────────────┘
                                    │
        ┌───────────────────────────┼───────────────────────────┐
        ▼                           ▼                           ▼
┌───────────────┐          ┌───────────────┐          ┌───────────────┐
│  SEQUENCE     │          │   SEQUENCE    │          │   SEQUENCE    │
│  LABELING     │          │   TO SEQUENCE │          │   CLASSIF.    │
├───────────────┤          ├───────────────┤          ├───────────────┤
│ • POS Tagging │          │ • Translation │          │ • Sentiment   │
│ • NER         │          │ • Summarize   │          │ • Topic Class │
│ • Chunking    │          │ • Q&A         │          │ • Spam Detect │
└───────────────┘          └───────────────┘          └───────────────┘
```

### 7.0.4 Text Preprocessing Pipeline

Before NLP analysis, text undergoes preprocessing:

1. **Tokenization**: Splitting text into tokens (words, subwords)
   - "I'm happy" → ["I", "'m", "happy"] or ["I'm", "happy"]

2. **Lowercasing**: Normalizing case
   - "Hello WORLD" → "hello world"

3. **Stop Word Removal**: Removing common words
   - "the", "is", "at", "which", "on"

4. **Stemming**: Reducing to root form (crude)
   - "running", "runs", "ran" → "run"

5. **Lemmatization**: Dictionary-based root form
   - "better" → "good" (stems would fail here)

### 7.0.5 Text Representation

Computers need numerical representations of text:

| Method | Description | Dimension |
|--------|-------------|-----------|
| **One-Hot** | Binary vector per word | Vocabulary size |
| **Bag of Words** | Word frequency counts | Vocabulary size |
| **TF-IDF** | Weighted word importance | Vocabulary size |
| **Word2Vec** | Dense semantic vectors | 100-300 |
| **Transformers** | Contextual embeddings | 768-1024 |

## 7.1 Sentiment Analysis (TextBlob)

### 7.1.1 Theoretical Background: Sentiment Analysis

**Sentiment Analysis** (Opinion Mining) determines the emotional tone of text. It answers: "Is this text positive, negative, or neutral?"

**Applications:**
- Customer review analysis
- Social media monitoring
- Brand reputation management
- Political opinion tracking

**Approaches to Sentiment Analysis:**

| Approach | Method | Pros | Cons |
|----------|--------|------|------|
| **Lexicon-Based** | Dictionary lookup | No training needed | Limited vocabulary |
| **Machine Learning** | Trained classifier | Learns patterns | Needs labeled data |
| **Deep Learning** | Neural networks | Best accuracy | Compute intensive |
| **Hybrid** | Combined approach | Balanced | Complex |

### 7.1.2 Lexicon-Based Sentiment (TextBlob's Approach)

TextBlob uses a **lexicon-based** approach with the Pattern library's sentiment dictionary.

**How the Lexicon Works:**

Each word has pre-assigned scores:
```
Word        Polarity    Subjectivity
─────────────────────────────────────
"love"      +0.5        0.6
"hate"      -0.8        0.9
"good"      +0.7        0.6
"bad"       -0.7        0.6
"amazing"   +0.8        1.0
"terrible"  -1.0        1.0
```

**Polarity Calculation:**
```
Polarity = Σ(word_polarity × modifier) / N

Where:
- word_polarity: Lexicon score for each word
- modifier: Adjustments for negation, intensifiers
- N: Number of sentiment-bearing words
```

**Negation Handling:**
- "not good" → polarity of "good" is flipped
- "never happy" → "happy" becomes negative
- Window of 3 words typically checked

**Intensifier Handling:**
- "very good" → "good" polarity amplified by ~1.3x
- "extremely bad" → "bad" polarity amplified
- "somewhat happy" → "happy" polarity reduced

### 7.1.3 Subjectivity vs Objectivity

**Subjectivity** measures how opinion-based the text is:
- 0.0 = Completely objective (facts)
- 1.0 = Completely subjective (opinions)

**Examples:**
```
"The movie is 2 hours long" → Subjectivity: 0.0 (fact)
"The movie is amazing" → Subjectivity: 1.0 (opinion)
"The movie is somewhat good" → Subjectivity: 0.7 (mixed)
```

### 7.1.4 How TextBlob Works

TextBlob uses a lexicon-based approach with a pre-built sentiment dictionary. Each word has pre-assigned polarity and subjectivity scores.

```python
from textblob import TextBlob

def analyze_sentiment(text: str) -> Dict:
    blob = TextBlob(text)
    
    # Polarity: -1 (negative) to +1 (positive)
    polarity = blob.sentiment.polarity
    
    # Subjectivity: 0 (objective) to 1 (subjective)
    subjectivity = blob.sentiment.subjectivity
    
    # Determine label
    if polarity > 0.1:
        label = "Positive"
    elif polarity < -0.1:
        label = "Negative"
    else:
        label = "Neutral"
    
    return {
        "polarity": polarity,
        "subjectivity": subjectivity,
        "label": label
    }
```

### 7.1.2 Sentiment Calculation Process

1. **Tokenization**: Text is split into words
2. **Lookup**: Each word's sentiment score is retrieved from lexicon
3. **Aggregation**: Individual scores are averaged
4. **Negation Handling**: Words like "not" flip nearby word sentiment

**Example:**
```
"I love this amazing product"
- love: +0.5
- amazing: +0.8
Average polarity: +0.65 → Positive
```

### 7.1.3 Per-Sentence Analysis

```python
from textblob import TextBlob

text = "I love NLP. But sometimes it's confusing."
blob = TextBlob(text)

for sentence in blob.sentences:
    print(f"Sentence: {sentence}")
    print(f"Polarity: {sentence.sentiment.polarity}")
    print(f"Subjectivity: {sentence.sentiment.subjectivity}")
```

**Output:**
```
Sentence: I love NLP.
Polarity: 0.5
Subjectivity: 0.6

Sentence: But sometimes it's confusing.
Polarity: -0.2
Subjectivity: 0.4
```

## 7.2 Named Entity Recognition (spaCy)

### 7.2.1 Theoretical Background: Named Entity Recognition

**Named Entity Recognition (NER)** is the task of identifying and classifying named entities in text into predefined categories such as person names, organizations, locations, dates, etc.

**Why NER is Important:**
1. **Information Extraction**: Finding key facts in documents
2. **Knowledge Graphs**: Building structured databases
3. **Question Answering**: Identifying relevant entities
4. **Content Analysis**: Understanding what/who is discussed

**NER Challenges:**
- **Ambiguity**: "Apple" - company or fruit?
- **Unknown Entities**: New names not in training data
- **Boundary Detection**: "New York City" - one or three entities?
- **Nested Entities**: "Bank of America" contains "America"

### 7.2.2 NER Approaches

| Approach | Method | Example |
|----------|--------|---------|
| **Rule-Based** | Patterns, gazetteers | Regex for dates |
| **Statistical** | CRF, HMM | Sequence labeling |
| **Neural** | BiLSTM-CRF | Deep learning |
| **Transformer** | BERT, spaCy | Contextual |

### 7.2.3 BIO Tagging Scheme

NER uses the BIO (Beginning-Inside-Outside) scheme:

```
Text: "Barack Obama visited New York City"

Token        Tag
─────────────────────
Barack       B-PERSON    (Beginning of PERSON)
Obama        I-PERSON    (Inside PERSON)
visited      O           (Outside any entity)
New          B-GPE       (Beginning of GPE)
York         I-GPE       (Inside GPE)
City         I-GPE       (Inside GPE)
```

**Why BIO?**
- Handles multi-word entities
- Clear boundaries between adjacent entities
- Standard format for training data

### 7.2.4 spaCy's NER Architecture

spaCy uses a **Transition-Based** NER system with neural networks:

```
Word Embeddings → CNN/Transformer → Transition System → Entities
```

1. **Word Embeddings**: Convert words to vectors
2. **Context Encoding**: Capture surrounding context
3. **Transition System**: Decide entity boundaries
4. **Output**: Entity spans with labels

### 7.2.5 How NER Works

NER identifies and classifies named entities in text:

```python
import spacy

def extract_named_entities(text: str) -> Dict:
    # Load English model
    nlp = spacy.load("en_core_web_sm")
    
    # Process text
    doc = nlp(text)
    
    entities = {}
    for ent in doc.ents:
        label = ent.label_  # Entity type (PERSON, ORG, etc.)
        text = ent.text     # Entity text
        
        if label not in entities:
            entities[label] = []
        entities[label].append(text)
    
    return entities
```

### 7.2.2 Entity Types

| Label | Description | Example |
|-------|-------------|---------|
| PERSON | People's names | "John Smith" |
| ORG | Organizations | "Google", "UN" |
| GPE | Countries, cities | "Pakistan", "Karachi" |
| DATE | Dates | "January 2026" |
| TIME | Times | "3:00 PM" |
| MONEY | Monetary values | "$100" |
| PERCENT | Percentages | "50%" |
| PRODUCT | Products | "iPhone" |
| EVENT | Events | "Olympics" |

### 7.2.3 spaCy Pipeline

```
Text → Tokenizer → Tagger → Parser → NER → Doc
         ↓          ↓        ↓       ↓
       Tokens     POS Tags  Deps   Entities
```

1. **Tokenizer**: Splits text into tokens
2. **Tagger**: Assigns part-of-speech tags
3. **Parser**: Determines syntactic dependencies
4. **NER**: Identifies named entities

## 7.3 Keyword Extraction (TF-IDF)

### 7.3.1 Theoretical Background: Keyword Extraction

**Keyword Extraction** identifies the most important words or phrases that represent the main topics of a document.

**Use Cases:**
- Search engine indexing
- Document summarization
- Content tagging/categorization
- SEO optimization

**Approaches to Keyword Extraction:**

| Method | Type | Description |
|--------|------|-------------|
| **TF-IDF** | Statistical | Word importance based on frequency |
| **RAKE** | Statistical | Rapid Automatic Keyword Extraction |
| **TextRank** | Graph-based | PageRank for text |
| **YAKE** | Unsupervised | Yet Another Keyword Extractor |
| **KeyBERT** | Neural | BERT embeddings + cosine similarity |

### 7.3.2 Term Frequency (TF)

**Term Frequency** measures how often a word appears in a document:

```
TF(t, d) = (Number of times term t appears in document d)
           ─────────────────────────────────────────────────
           (Total number of terms in document d)
```

**Example:**
```
Document: "the cat sat on the mat"
Total terms: 6

TF("the") = 2/6 = 0.33
TF("cat") = 1/6 = 0.17
TF("sat") = 1/6 = 0.17
```

**Problem with TF alone:**
- Common words like "the", "is", "and" get high scores
- But they don't represent document content!

### 7.3.3 Inverse Document Frequency (IDF)

**IDF** measures how unique a word is across all documents:

```
IDF(t) = log( Total number of documents )
             ─────────────────────────────────
             Number of documents containing term t
```

**Intuition:**
- Words appearing in many documents → Low IDF (common)
- Words appearing in few documents → High IDF (rare/specific)

**Example (5 documents):**
```
"the" appears in 5/5 documents → IDF = log(5/5) = 0
"algorithm" appears in 1/5 documents → IDF = log(5/1) = 0.7
```

### 7.3.4 TF-IDF Combined

**TF-IDF** combines both metrics:

```
TF-IDF(t, d) = TF(t, d) × IDF(t)
```

**Why This Works:**
- High TF × High IDF = Important keyword (frequent + unique)
- High TF × Low IDF = Common word (ignored)
- Low TF × High IDF = Rare mention (less important)

**Mathematical Properties:**
- Range: 0 to ~1 (normalized)
- Sparse representation (most values are 0)
- Cosine similarity for document comparison

### 7.3.5 N-grams: Beyond Single Words

**N-grams** capture phrases, not just individual words:

| N | Name | Example |
|---|------|---------|
| 1 | Unigram | "machine", "learning" |
| 2 | Bigram | "machine learning" |
| 3 | Trigram | "natural language processing" |

**Why N-grams Matter:**
- "hot dog" ≠ "hot" + "dog"
- "New York" is one entity, not two words
- Phrases carry more specific meaning

### 7.3.6 TF-IDF Explained

**TF (Term Frequency):**
```
TF(t,d) = Number of times term t appears in document d
          ─────────────────────────────────────────────
          Total number of terms in document d
```

**IDF (Inverse Document Frequency):**
```
IDF(t) = log( Total number of documents )
             ──────────────────────────────
             Number of documents containing term t
```

**TF-IDF Score:**
```
TF-IDF(t,d) = TF(t,d) × IDF(t)
```

### 7.3.2 Implementation

```python
from sklearn.feature_extraction.text import TfidfVectorizer

def extract_keywords_tfidf(text: str, top_n: int = 10) -> Dict:
    # Split into sentences (documents)
    sentences = re.split(r'(?<=[.!?])\s+', text)
    
    # TF-IDF vectorizer
    vectorizer = TfidfVectorizer(
        max_features=100,       # Max vocabulary size
        stop_words='english',   # Remove common words
        ngram_range=(1, 2),     # Unigrams and bigrams
    )
    
    # Fit and transform
    tfidf_matrix = vectorizer.fit_transform(sentences)
    feature_names = vectorizer.get_feature_names_out()
    
    # Sum scores across all sentences
    tfidf_scores = tfidf_matrix.sum(axis=0).A1
    
    # Get top keywords
    top_indices = tfidf_scores.argsort()[-top_n:][::-1]
    keywords = [
        (feature_names[i], tfidf_scores[i]) 
        for i in top_indices
    ]
    
    return {"keywords": keywords}
```

### 7.3.3 Why TF-IDF Works

- **Common words** (the, is, and) have low IDF scores
- **Rare but relevant words** have high TF-IDF scores
- **Ngrams (1,2)** capture phrases like "machine learning"

## 7.4 Readability Scores

### 7.4.1 Theoretical Background: Readability Analysis

**Readability** measures how easy or difficult a text is to read and understand. It's based on linguistic features that correlate with comprehension difficulty.

**Why Readability Matters:**
1. **Education**: Match content to student reading level
2. **Healthcare**: Patient-friendly medical documents
3. **Legal**: Plain language requirements
4. **Marketing**: Accessible advertising copy
5. **Content Creation**: Audience-appropriate content

### 7.4.2 Factors Affecting Readability

| Factor | Easy | Difficult |
|--------|------|-----------|
| **Word Length** | Short words | Long words |
| **Sentence Length** | Short sentences | Long sentences |
| **Syllables** | Few syllables | Many syllables |
| **Vocabulary** | Common words | Rare/technical words |
| **Sentence Structure** | Simple | Complex/nested |

### 7.4.3 History of Readability Formulas

| Year | Formula | Creator | Focus |
|------|---------|---------|-------|
| 1948 | Flesch Reading Ease | Rudolf Flesch | General readability |
| 1952 | Fog Index | Robert Gunning | Business writing |
| 1975 | Flesch-Kincaid | Kincaid et al. | US Grade level |
| 1969 | SMOG | G. Harry McLaughlin | Healthcare materials |

### 7.4.4 Syllable Theory

**Syllables** are units of pronunciation. Longer syllables = harder words.

**Syllable Counting Rules:**
1. Count vowel groups (a, e, i, o, u, y)
2. Subtract silent 'e' at end
3. Minimum 1 syllable per word

**Examples:**
```
"cat" → 1 syllable (a)
"table" → 2 syllables (a-le, silent e ignored)
"beautiful" → 4 syllables (beau-ti-ful)
"algorithm" → 4 syllables (al-go-ri-thm)
```

### 7.4.5 Grade Level Interpretation

US Grade levels map to ages and education:

| Grade | Age | Reading Level |
|-------|-----|---------------|
| 1-5 | 6-10 | Elementary |
| 6-8 | 11-13 | Middle School |
| 9-12 | 14-17 | High School |
| 13-16 | 18-22 | College |
| 17+ | 22+ | Graduate/Professional |

**Target Levels:**
- General audience: 6th-8th grade
- Academic papers: 12th-16th grade
- Children's content: 3rd-5th grade

### 7.4.6 Flesch Reading Ease

```python
def flesch_reading_ease(text):
    words = count_words(text)
    sentences = count_sentences(text)
    syllables = count_syllables(text)
    
    score = 206.835 - 1.015 * (words / sentences) 
                    - 84.6 * (syllables / words)
    return score
```

**Score Interpretation:**
| Score | Level |
|-------|-------|
| 90-100 | Very Easy (5th grade) |
| 80-90 | Easy (6th grade) |
| 70-80 | Fairly Easy (7th grade) |
| 60-70 | Standard (8th-9th grade) |
| 50-60 | Fairly Difficult (10th-12th grade) |
| 30-50 | Difficult (College) |
| 0-30 | Very Difficult (Graduate) |

### 7.4.2 Flesch-Kincaid Grade Level

```python
def flesch_kincaid_grade(text):
    words = count_words(text)
    sentences = count_sentences(text)
    syllables = count_syllables(text)
    
    grade = 0.39 * (words / sentences) 
          + 11.8 * (syllables / words) - 15.59
    return grade
```

### 7.4.3 Gunning Fog Index

```python
def gunning_fog_index(text):
    words = count_words(text)
    sentences = count_sentences(text)
    complex_words = count_complex_words(text)  # 3+ syllables
    
    fog = 0.4 * ((words / sentences) 
               + 100 * (complex_words / words))
    return fog
```

### 7.4.4 Syllable Counting

```python
def count_syllables(word):
    """Count syllables in a word"""
    word = word.lower()
    count = 0
    vowels = "aeiouy"
    
    if word[0] in vowels:
        count += 1
    
    for i in range(1, len(word)):
        if word[i] in vowels and word[i-1] not in vowels:
            count += 1
    
    # Handle silent 'e'
    if word.endswith('e'):
        count -= 1
    
    return max(count, 1)
```

## 7.5 Toxicity Detection

### 7.5.1 Theoretical Background: Toxicity Detection

**Toxicity Detection** identifies harmful, offensive, or inappropriate content in text. This is crucial for content moderation and platform safety.

**Types of Toxic Content:**

| Category | Description | Examples |
|----------|-------------|----------|
| **Profanity** | Curse words, vulgar language | Swear words |
| **Hate Speech** | Targeting protected groups | Racism, sexism |
| **Threats** | Violence, harm | Death threats |
| **Harassment** | Personal attacks | Bullying, insults |
| **Self-harm** | Suicide, self-injury content | - |
| **Sexual Content** | Explicit material | Adult content |

### 7.5.2 Approaches to Toxicity Detection

| Approach | Method | Pros | Cons |
|----------|--------|------|------|
| **Word Lists** | Blacklist matching | Fast, simple | Easily bypassed |
| **Pattern Matching** | Regex patterns | Catches variations | Complex rules |
| **ML Classification** | Trained models | Learns context | Needs training data |
| **Deep Learning** | BERT, Transformers | Best accuracy | Compute intensive |

### 7.5.3 Challenges in Toxicity Detection

**1. Context Dependence:**
```
"I'll kill you" - Threat or playful banter?
"That's sick!" - Negative or positive (slang)?
```

**2. Obfuscation Techniques:**
- Letter substitution: "sh!t", "f*ck"
- Spacing: "f u c k"
- Homoglyphs: "shіt" (Cyrillic 'і')
- Leetspeak: "sh1t", "h8"

**3. Implicit Toxicity:**
- Sarcasm: "Oh, what a *brilliant* idea"
- Dog whistles: Coded language
- Microaggressions: Subtle discrimination

**4. False Positives:**
- Medical terms: "cancer", "suicide prevention"
- Quotes/reporting: Discussing toxicity
- Reclaimed terms: Within-group usage

### 7.5.4 Rule-Based vs ML Approaches

**Rule-Based (Our Implementation):**
```
Advantages:
✓ No training required
✓ Explainable decisions
✓ Easy to update word lists
✓ Fast execution

Limitations:
✗ Misses context
✗ Easily bypassed with misspellings
✗ High false positive rate
✗ Doesn't understand intent
```

**ML-Based (Advanced):**
```
Advantages:
✓ Understands context
✓ Catches paraphrased toxicity
✓ Handles variations
✓ Lower false positives

Limitations:
✗ Needs labeled training data
✗ Computational cost
✗ "Black box" decisions
✗ Can learn biases
```

### 7.5.5 Rule-Based Approach

```python
def detect_toxicity_rulebased(text: str) -> Dict:
    """Detect toxic content using word lists"""
    
    toxic_categories = {
        "profanity": ["fuck", "shit", "damn", "ass", "bitch"],
        "hate_speech": ["racist", "sexist", "homophobic"],
        "threats": ["kill", "murder", "destroy", "attack"],
        "insults": ["stupid", "idiot", "moron", "dumb"]
    }
    
    text_lower = text.lower()
    toxic_found = {}
    total_count = 0
    
    for category, words in toxic_categories.items():
        found = []
        for word in words:
            if re.search(r'\b' + word + r'\b', text_lower):
                found.append(word)
                total_count += 1
        if found:
            toxic_found[category] = found
    
    # Calculate toxicity score
    word_count = len(text.split())
    toxicity_score = total_count / max(word_count, 1)
    
    return {
        "is_toxic": total_count > 0,
        "toxic_words_found": toxic_found,
        "toxicity_score": toxicity_score
    }
```

## 7.6 Text Summarization

### 7.6.1 Theoretical Background: Text Summarization

**Text Summarization** automatically creates a shorter version of a document while preserving key information.

**Types of Summarization:**

| Type | Method | Output |
|------|--------|--------|
| **Extractive** | Select existing sentences | Original sentences |
| **Abstractive** | Generate new text | Paraphrased content |
| **Hybrid** | Combine both | Mixed approach |

### 7.6.2 Extractive vs Abstractive Summarization

**Extractive Summarization:**
```
Input: "The cat sat on the mat. The cat was orange. It was a sunny day."
Process: Score each sentence, pick top ones
Output: "The cat sat on the mat. It was a sunny day."
```

**Abstractive Summarization:**
```
Input: "The cat sat on the mat. The cat was orange. It was a sunny day."
Process: Understand meaning, generate new text
Output: "An orange cat rested on a mat during a sunny day."
```

### 7.6.3 Extractive Methods

| Method | Description | Technique |
|--------|-------------|-----------|
| **Frequency-Based** | Important words appear often | TF-IDF |
| **Position-Based** | First/last sentences important | Heuristics |
| **Graph-Based** | Sentence similarity graphs | TextRank |
| **ML-Based** | Trained sentence selector | Classification |

**Our Approach: TF-IDF Based Extraction**

1. Split text into sentences
2. Calculate TF-IDF for each sentence
3. Score sentences by TF-IDF sum
4. Select top-scoring sentences
5. Maintain original order

### 7.6.4 TextRank Algorithm (Graph-Based)

TextRank (Mihalcea & Tarau, 2004) adapts PageRank for text:

```
Sentences as Nodes → Similarity as Edges → PageRank → Top Sentences
```

**Similarity Calculation:**
```
Similarity(Si, Sj) = |words in common| / (log|Si| + log|Sj|)
```

**PageRank for Sentences:**
```
Score(Si) = (1-d) + d × Σ (Similarity(Si,Sj) × Score(Sj))
                       j∈neighbors

Where d = 0.85 (damping factor)
```

### 7.6.5 Compression Ratio

**Compression Ratio** measures summary conciseness:

```
Compression Ratio = Length of Summary / Length of Original
```

**Typical Ratios:**
- News headlines: 5-10%
- Executive summaries: 10-20%
- Academic abstracts: 5-15%
- Content previews: 20-30%

### 7.6.6 Extractive Summarization

Our system uses extractive summarization - selecting the most important existing sentences:

```python
def summarize_text_extractive(text: str, num_sentences: int = 3) -> Dict:
    """Extract most important sentences using TF-IDF"""
    
    sentences = re.split(r'(?<=[.!?])\s+', text)
    
    if len(sentences) <= num_sentences:
        return {"summary": text}
    
    # Calculate TF-IDF for each sentence
    vectorizer = TfidfVectorizer(stop_words='english')
    tfidf_matrix = vectorizer.fit_transform(sentences)
    
    # Score each sentence by sum of TF-IDF values
    sentence_scores = tfidf_matrix.sum(axis=1).A1
    
    # Get top sentences (maintain original order)
    top_indices = sentence_scores.argsort()[-num_sentences:]
    top_indices = sorted(top_indices)
    
    summary = ' '.join([sentences[i] for i in top_indices])
    
    return {
        "summary": summary,
        "original_sentences": len(sentences),
        "compression_ratio": num_sentences / len(sentences)
    }
```

---

# 8. AI Integration (Google Gemini API)

## 8.1 Theoretical Background: Large Language Models

### 8.1.1 What are Large Language Models (LLMs)?

**Large Language Models** are neural networks trained on massive text corpora to understand and generate human language. They represent the current state-of-the-art in NLP.

**Key Characteristics:**
- **Scale**: Billions of parameters (GPT-4: ~1.7T, Gemini: ~1T)
- **Training Data**: Trillions of tokens from internet text
- **Capabilities**: Text generation, summarization, reasoning, coding

### 8.1.2 How LLMs Work

**Training Process:**
```
Text Corpus → Tokenization → Neural Network → Predict Next Token
     ↓              ↓              ↓              ↓
  "The cat"    [123, 456]    Transformer    P(sat) = 0.8
```

**Self-Supervised Learning:**
- No labeled data needed
- Predict next word given context
- Billions of training examples from raw text

**Emergent Capabilities:**
- Few-shot learning: Learn from examples in prompt
- Chain-of-thought: Step-by-step reasoning
- Instruction following: Understand natural language commands

### 8.1.3 Transformer Architecture (Foundation)

LLMs are built on the **Transformer** architecture:

```
Input Tokens → Embeddings → [Attention + FFN] × N → Output
                                    ↑
                           Multi-Head Attention
```

**Key Components:**

1. **Token Embeddings**: Words → Vectors
2. **Positional Encoding**: Sequence order information
3. **Multi-Head Attention**: Focus on relevant context
4. **Feed-Forward Networks**: Non-linear transformations
5. **Layer Normalization**: Stable training

### 8.1.4 Google Gemini Specifics

**Gemini** is Google's multimodal AI family:

| Model | Capabilities | Speed | Use Case |
|-------|--------------|-------|----------|
| Gemini Flash | Text, fast | Fastest | Real-time apps |
| Gemini Pro | Text, reasoning | Fast | General purpose |
| Gemini Ultra | Multimodal, complex | Slower | Advanced tasks |

**Multimodal Capabilities:**
- Text understanding and generation
- Image and video analysis
- Code generation and debugging
- Audio processing

### 8.1.5 API-Based AI Integration

**Why Use APIs Instead of Local Models?**

| Factor | API | Local |
|--------|-----|-------|
| Setup | Minutes | Hours/Days |
| Cost | Pay per use | Hardware cost |
| Quality | Cutting-edge | May be older |
| Privacy | Data leaves device | Stays local |
| Compute | Cloud servers | Your hardware |
| Scaling | Automatic | Manual |

### 8.1.6 Prompt Engineering

**Prompt Engineering** is the art of crafting inputs to get desired outputs from LLMs.

**Key Techniques:**

1. **Clear Instructions**: Specify exactly what you want
2. **Role Assignment**: "You are an expert in..."
3. **Format Specification**: "Return as JSON/bullet points"
4. **Examples (Few-shot)**: Show desired input-output pairs
5. **Chain of Thought**: "Think step by step"
6. **Constraints**: "Do not include...", "Limit to..."

**Our Monetization Prompt Structure:**
```
[Context]: What the text is (video transcription)
[Task 1]: 2-line summary
[Task 2]: Words to avoid + alternatives
[Task 3]: Monetization advice
[Format]: Specify headers and structure
```

## 8.2 Gemini API Overview

Google Gemini is a multimodal AI model that can understand and generate text, code, and more.

## 8.2 API Configuration

```python
import google.generativeai as genai

# Configure API key
genai.configure(api_key="YOUR_API_KEY")

# Initialize model
model = genai.GenerativeModel("models/gemini-2.5-flash")
```

## 8.3 Prompt Engineering

Our system uses a carefully crafted prompt for monetization advice:

```python
prompt = f"""Analyze this video transcription and provide:

1. **2-LINE SUMMARY**: A brief 2-line summary of the content.

2. **WORDS TO AVOID**: List any profanity, inappropriate language, 
   or demonetization-risk words found. For each word, suggest a 
   safe alternative.

3. **MONETIZATION ADVICE**: Give 3-5 specific tips to make this 
   video more YouTube-monetization friendly.

Transcription:
{transcript[:3000]}

Format your response clearly with headers."""
```

## 8.4 API Response Handling

```python
try:
    response = model.generate_content(prompt)
    advice = response.text
except Exception as e:
    advice = f"Error: {str(e)}"
```

## 8.5 Rate Limiting and Error Handling

```python
# Gemini has rate limits:
# - Free tier: 60 requests per minute
# - 1 million tokens per minute

if "429" in str(error):
    # Rate limit exceeded
    st.warning("API rate limit exceeded. Please wait.")
elif "401" in str(error):
    # Invalid API key
    st.error("Invalid API key.")
```

---

# 9. User Interface (Streamlit)

## 9.1 Theoretical Background: Web Application Development

### 9.1.1 Traditional vs Modern Web Development

**Traditional Web Development:**
```
HTML (Structure) + CSS (Styling) + JavaScript (Interactivity)
           ↓
    Backend (Python/Java/Node.js)
           ↓
    Database (MySQL/MongoDB)
```

**Modern Python Web Frameworks:**

| Framework | Type | Learning Curve | Use Case |
|-----------|------|----------------|----------|
| Django | Full-stack | High | Complex apps |
| Flask | Micro | Medium | APIs, small apps |
| FastAPI | API-focused | Medium | REST APIs |
| Streamlit | Data apps | **Low** | ML demos, dashboards |
| Gradio | ML interfaces | Low | Model demos |

### 9.1.2 Why Streamlit for Data Applications?

**Streamlit's Philosophy:**
- Python-only (no HTML/CSS/JS needed)
- Reactive programming model
- Built for data scientists
- Rapid prototyping

**Comparison:**

| Feature | Flask | Django | Streamlit |
|---------|-------|--------|-----------|
| Lines of code for basic app | 50+ | 100+ | 10 |
| HTML/CSS needed | Yes | Yes | No |
| Built-in widgets | No | No | Yes |
| Data visualization | Manual | Manual | Built-in |
| Hot reload | Plugin | Plugin | Automatic |

### 9.1.3 Streamlit's Reactive Model

**How Traditional Apps Work:**
```
User Action → Event Handler → Update State → Re-render Component
```

**How Streamlit Works:**
```
User Action → Re-run Entire Script → Display New State
```

**Key Insight:**
- Streamlit re-runs the entire script on every interaction
- State is managed through `st.session_state`
- Widgets automatically update when data changes

### 9.1.4 Streamlit Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     STREAMLIT APP                           │
├─────────────────────────────────────────────────────────────┤
│  Python Script (app.py)                                     │
│     ↓                                                       │
│  Streamlit Library                                          │
│     ↓                                                       │
│  WebSocket Connection                                       │
│     ↓                                                       │
│  React Frontend (Browser)                                   │
└─────────────────────────────────────────────────────────────┘
```

**Components:**
1. **Python Script**: Your application logic
2. **Streamlit Library**: Translates Python to web components
3. **WebSocket**: Real-time communication
4. **React Frontend**: Renders UI in browser

### 9.1.5 Session State Theory

**The Problem:**
- Streamlit re-runs script on every interaction
- Local variables are reset
- How to preserve data between runs?

**The Solution: Session State**
```python
# Without session state - resets to 0 every time
counter = 0

# With session state - persists across runs
if 'counter' not in st.session_state:
    st.session_state.counter = 0
```

**Session State Scope:**
- Per-user: Each user has their own state
- Per-session: Cleared when browser closes
- In-memory: Not persisted to disk

### 9.1.6 Layout and Components

**Streamlit Layout System:**
```
┌────────────────────────────────────────────────────┐
│                    HEADER                          │
├───────────┬────────────────────────────────────────┤
│           │                                        │
│  SIDEBAR  │           MAIN CONTENT                 │
│           │                                        │
│  - Input  │   ┌─────────┬─────────┬─────────┐     │
│  - Filter │   │  TAB 1  │  TAB 2  │  TAB 3  │     │
│  - Config │   ├─────────┴─────────┴─────────┤     │
│           │   │                             │     │
│           │   │    COLUMNS / EXPANDERS      │     │
│           │   │                             │     │
│           │   └─────────────────────────────┘     │
└───────────┴────────────────────────────────────────┘
```

## 9.2 Streamlit Overview

Streamlit is a Python framework for building data apps with minimal code.

## 9.2 Page Configuration

```python
st.set_page_config(
    page_title="Audio Transcription Tool",
    layout="wide",
    initial_sidebar_state="collapsed"
)
```

## 9.3 Tab-Based Layout

```python
# Create tabs
tab1, tab2, tab3 = st.tabs([
    "📥 Input", 
    "📝 Transcription", 
    "✔️ Quality Check"
])

with tab1:
    # Input handling code
    
with tab2:
    # Transcription display
    
with tab3:
    # Quality analysis
```

## 9.4 Session State Management

```python
# Store transcription result in session
if "transcription_result" not in st.session_state:
    st.session_state.transcription_result = None

# After transcription
st.session_state.transcription_result = result

# Access in other tabs
if st.session_state.transcription_result:
    transcript = st.session_state.transcription_result["text"]
```

## 9.5 Interactive Components

```python
# File uploader
uploaded_file = st.file_uploader(
    "Upload audio file",
    type=["mp3", "wav", "m4a", "flac", "ogg"]
)

# Selectbox
model_size = st.selectbox(
    "Whisper Model Size",
    ["tiny", "base", "small", "medium", "large"]
)

# Button with loading state
if st.button("Transcribe", type="primary"):
    with st.spinner("Transcribing..."):
        result = transcribe_audio(audio_path)
```

---

# 10. Conclusion & Future Work

## 10.0 Theoretical Reflection: NLP in Practice

### 10.0.1 Integration of Multiple NLP Techniques

This project demonstrates how multiple NLP techniques work together in a real application:

```
┌─────────────────────────────────────────────────────────────────┐
│                    NLP TECHNIQUE SYNERGY                        │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ASR (Speech→Text)                                              │
│       ↓                                                         │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │  Lexical Analysis: Tokenization, Word Statistics        │   │
│  └─────────────────────────────────────────────────────────┘   │
│       ↓                                                         │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │  Semantic Analysis: Sentiment, NER, Keywords            │   │
│  └─────────────────────────────────────────────────────────┘   │
│       ↓                                                         │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │  Discourse Analysis: Summarization, Readability         │   │
│  └─────────────────────────────────────────────────────────┘   │
│       ↓                                                         │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │  Pragmatic Analysis: Toxicity, Quality, LLM Advice      │   │
│  └─────────────────────────────────────────────────────────┘   │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### 10.0.2 Trade-offs in NLP System Design

| Aspect | Our Choice | Alternative | Trade-off |
|--------|------------|-------------|-----------|
| Sentiment | Lexicon (TextBlob) | BERT | Speed vs Accuracy |
| NER | spaCy (small) | Transformer NER | Memory vs Coverage |
| Keywords | TF-IDF | KeyBERT | Simplicity vs Quality |
| Toxicity | Rule-based | Perspective API | Privacy vs Accuracy |
| Summary | Extractive | Abstractive | Faithfulness vs Fluency |

### 10.0.3 Lessons from NLP Theory to Practice

**1. No Single Best Approach:**
Different techniques excel in different contexts. Lexicon sentiment works well for general content but struggles with sarcasm. Transformer models are more accurate but slower.

**2. Preprocessing Matters:**
The quality of NLP output depends heavily on input quality. ASR errors propagate through the pipeline.

**3. Explainability vs Performance:**
Rule-based systems (toxicity, fillers) are explainable but limited. ML systems perform better but are "black boxes."

**4. Domain Adaptation:**
Pre-trained models work well for general content but may need fine-tuning for specialized domains (medical, legal, technical).

## 10.1 Summary of Achievements

| Feature | Status | Technology |
|---------|--------|------------|
| Audio Transcription | ✅ Complete | Faster-Whisper |
| YouTube Download | ✅ Complete | yt-dlp |
| Quality Check | ✅ Complete | Regex matching |
| Sentiment Analysis | ✅ Complete | TextBlob |
| Named Entity Recognition | ✅ Complete | spaCy |
| Keyword Extraction | ✅ Complete | TF-IDF |
| Readability Scoring | ✅ Complete | Custom formulas |
| Toxicity Detection | ✅ Complete | Rule-based |
| AI Advice | ✅ Complete | Gemini API |

## 10.2 Lessons Learned

1. **NLP Libraries**: Different libraries excel at different tasks
2. **Pipeline Design**: Modular architecture enables easy maintenance
3. **API Integration**: Proper error handling is essential
4. **User Experience**: Real-time feedback improves usability

## 10.3 Future Enhancements

| Enhancement | Description |
|-------------|-------------|
| Speaker Diarization | Identify different speakers |
| Multi-language Support | Support for Urdu, Arabic, etc. |
| Real-time Transcription | Live audio transcription |
| Transformer Models | Use BERT for better sentiment |
| Custom Word Lists | User-defined profanity lists |
| Cloud Deployment | Deploy to Streamlit Cloud |

## 10.4 Technical Challenges Faced

1. **OpenMP Conflict**: Resolved with `KMP_DUPLICATE_LIB_OK=TRUE`
2. **Model Caching**: Implemented custom cache path management
3. **API Rate Limits**: Handled with proper error messages
4. **FFmpeg Integration**: Searched multiple paths for compatibility

---

# Appendix A: File Structure

```
audio-transcription-local/
├── app.py              (526 lines) - Main UI
├── transcriber.py      (204 lines) - Whisper integration
├── audio_utils.py      (85 lines)  - Audio processing
├── nlp_analysis.py     (670 lines) - NLP functions
├── model_manager.py    (157 lines) - Model caching
├── download_models.py  (75 lines)  - Model download script
├── requirements.txt    (14 lines)  - Dependencies
└── README.md           - Documentation
```

---

# Appendix B: Dependencies

```
streamlit          - Web UI framework
faster-whisper     - Speech-to-text
yt-dlp            - YouTube download
pandas            - Data handling
numpy             - Numerical operations
textblob          - Sentiment analysis
spacy             - NER
scikit-learn      - TF-IDF
google-generativeai - Gemini API
```

---

**End of Report**

*Total Pages: 10+*
*Word Count: ~4000*
