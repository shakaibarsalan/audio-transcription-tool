import os
# Fix OpenMP conflict error
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import streamlit as st
import tempfile
from pathlib import Path
import pandas as pd
from datetime import datetime

# Gemini AI
try:
    import google.generativeai as genai
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False

# Import local modules
from audio_utils import download_youtube_video, extract_audio, find_ffmpeg
from transcriber import transcribe_audio_local, check_transcription_quality
from model_manager import is_model_downloaded, get_models_status, get_model_cache_path
from nlp_analysis import (
    analyze_sentiment, extract_named_entities, summarize_text_extractive,
    detect_toxicity_rulebased, extract_keywords_tfidf, calculate_readability,
    analyze_pos_tags, run_full_nlp_analysis
)

st.set_page_config(
    page_title="Audio Transcription Tool",
    layout="wide",
    initial_sidebar_state="collapsed"
)

st.title("🎙️ Audio Transcription & Quality Check")
st.markdown("Local audio transcription using Whisper + OpenAI compatibility")

# Sidebar configuration
with st.sidebar:
    st.subheader("📦 Model Cache Status")
    status = get_models_status()
    downloaded = [m for m, info in status.items() if info["downloaded"]]
    missing = [m for m, info in status.items() if not info["downloaded"]]

    st.caption(f"Cache: `{get_model_cache_path()}`")
    if missing:
        st.warning(
            "Models not cached: " + ", ".join(missing) +
            " — run `python download_models.py " + " ".join(missing) + "` to pre-download."
        )
    else:
        st.success("All listed Whisper models are cached locally. No downloads will occur.")

    with st.container():
        st.write("**Downloaded Models:**")
        for model in downloaded:
            st.write(f"✓ {model}")
        if not downloaded:
            st.write("(none)")

    with st.container():
        st.write("**Not Downloaded:**")
        for model in missing:
            st.write(f"✗ {model}")
        if not missing:
            st.write("(none)")

    st.divider()
    st.header("⚙️ Settings")
    
    model_size = st.selectbox(
        "Whisper Model Size",
        ["tiny", "base", "small", "medium", "large", "large-v3"],
        index=0,  # default to "tiny"
        help="Larger models are more accurate but slower"
    )
    
    # Check if selected model is downloaded
    if not is_model_downloaded(model_size):
        st.warning(f"⚠️ Model '{model_size}' not cached. Run `python download_models.py` first to pre-download models, or it will download on first transcription.")
    
    device = st.selectbox(
        "Processing Device",
        ["auto", "cpu", "cuda"],
        help="auto: uses GPU if available, otherwise CPU"
    )
    
    compute_type = st.selectbox(
        "Compute Type",
        ["float16", "int8", "int8_float16", "float32"],
        index=0,
        help="float16: faster and uses less memory (requires GPU), float32: slower but always works"
    )
    
    vad_filter = st.checkbox(
        "Use VAD Filter",
        value=True,
        help="Skip silent segments in audio"
    )
    
    st.divider()
    st.info(f"✅ FFmpeg: {find_ffmpeg()}")

# Main tabs
tab1, tab2, tab3 = st.tabs(["📥 Input", "📝 Transcription", "✔️ Quality Check"])

# ====================
# TAB 1: INPUT
# ====================
with tab1:
    st.header("Input Audio Source")
    
    input_method = st.radio(
        "Choose input method:",
        ["YouTube URL", "Upload Audio File", "Upload Video File"],
        horizontal=True
    )
    
    audio_path = None
    
    if input_method == "YouTube URL":
        st.subheader("YouTube Video")
        url = st.text_input("Enter YouTube URL:")
        
        if url and st.button("📥 Download Video", key="download_btn"):
            try:
                with st.spinner("Downloading video..."):
                    downloads_dir = Path("downloads")
                    video_path = download_youtube_video(url, downloads_dir)
                    st.success(f"✅ Downloaded: {video_path.name}")
                    st.session_state.video_path = str(video_path)
            except Exception as e:
                st.error(f"❌ Error downloading: {e}")
    
    elif input_method == "Upload Audio File":
        st.subheader("Upload Audio File")
        uploaded_audio = st.file_uploader(
            "Upload audio file (mp3, wav, m4a, etc.)",
            type=["mp3", "wav", "m4a", "flac", "ogg"]
        )
        if uploaded_audio:
            # Save uploaded file
            temp_dir = Path(tempfile.gettempdir()) / "audio_transcription"
            temp_dir.mkdir(exist_ok=True)
            audio_path = temp_dir / uploaded_audio.name
            audio_path.write_bytes(uploaded_audio.getbuffer())
            st.success(f"✅ Uploaded: {uploaded_audio.name}")
            st.session_state.audio_path = str(audio_path)
    
    elif input_method == "Upload Video File":
        st.subheader("Upload Video File")
        uploaded_video = st.file_uploader(
            "Upload video file (mp4, avi, mov, etc.)",
            type=["mp4", "avi", "mov", "mkv", "webm"]
        )
        if uploaded_video:
            # Save uploaded file
            temp_dir = Path(tempfile.gettempdir()) / "audio_transcription"
            temp_dir.mkdir(exist_ok=True)
            video_path = temp_dir / uploaded_video.name
            video_path.write_bytes(uploaded_video.getbuffer())
            st.success(f"✅ Uploaded: {uploaded_video.name}")
            st.session_state.video_path = str(video_path)

# ====================
# TAB 2: TRANSCRIPTION
# ====================
with tab2:
    st.header("Audio Transcription")
    
    # Determine audio source
    audio_file = None
    
    if "audio_path" in st.session_state:
        audio_file = st.session_state.audio_path
        st.info(f"📁 Using uploaded audio: {Path(audio_file).name}")
    elif "video_path" in st.session_state:
        video_file = st.session_state.video_path
        st.info(f"🎬 Need to extract audio from: {Path(video_file).name}")
    
    col1, col2 = st.columns([3, 1])
    
    with col1:
        if st.button("▶️ Start Transcription", key="transcribe_btn", use_container_width=True):
            # Get or extract audio
            if "audio_path" in st.session_state:
                audio_file = st.session_state.audio_path
            elif "video_path" in st.session_state:
                with st.spinner("🎬 Extracting audio from video..."):
                    try:
                        video_path = st.session_state.video_path
                        temp_dir = Path(tempfile.gettempdir()) / "audio_transcription"
                        audio_file = extract_audio(video_path, temp_dir)
                        st.session_state.audio_path = str(audio_file)
                        st.success(f"✅ Audio extracted: {audio_file.name}")
                    except Exception as e:
                        st.error(f"❌ Error extracting audio: {e}")
                        audio_file = None
            else:
                st.warning("⚠️ Please provide audio or video input first")
                audio_file = None
            
            # Run transcription
            if audio_file:
                try:
                    with st.spinner(f"🔄 Transcribing with {model_size} model..."):
                        result = transcribe_audio_local(
                            audio_file,
                            model_size=model_size,
                            device=device,
                            compute_type=compute_type,
                            vad_filter=vad_filter
                        )
                        st.session_state.transcription_result = result
                        st.success("✅ Transcription complete!")
                except Exception as e:
                    st.error(f"❌ Transcription error: {e}")
    
    # Display results
    if "transcription_result" in st.session_state:
        result = st.session_state.transcription_result
        
        st.divider()
        st.subheader("Transcription Result")
        
        # Display metadata
        meta_col1, meta_col2 = st.columns(2)
        with meta_col1:
            st.metric("Language Detected", result.get("language", "unknown"))
        with meta_col2:
            st.metric("Segments", len(result["chunks"]))
        
        # Full text
        st.subheader("Full Transcription")
        st.text_area(
            "Transcribed Text:",
            value=result["text"],
            height=150,
            disabled=True,
            key="transcript_text"
        )
        
        # Display segments
        st.subheader("Transcription Segments")
        if result["chunks"]:
            segments_df = pd.DataFrame(result["chunks"])
            segments_df["start"] = segments_df["start"].apply(lambda x: f"{x:.2f}s")
            segments_df["end"] = segments_df["end"].apply(lambda x: f"{x:.2f}s")
            st.dataframe(segments_df, use_container_width=True)
        
        # SRT format
        with st.expander("📄 SRT Format"):
            st.text_area(
                "SRT Subtitle File:",
                value=result["srt"],
                height=200,
                disabled=True
            )
        
        # Download options
        st.divider()
        st.subheader("💾 Download Results")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.download_button(
                label="📝 Download Text (.txt)",
                data=result["text"],
                file_name=f"transcription_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                mime="text/plain"
            )
        
        with col2:
            st.download_button(
                label="📄 Download SRT",
                data=result["srt"],
                file_name=f"transcription_{datetime.now().strftime('%Y%m%d_%H%M%S')}.srt",
                mime="text/plain"
            )
        
        with col3:
            json_data = {
                "text": result["text"],
                "language": result.get("language", "unknown"),
                "segment_count": len(result["chunks"]),
                "segments": result["srt_segments"],
                "timestamp": datetime.now().isoformat()
            }
            import json
            st.download_button(
                label="📊 Download JSON",
                data=json.dumps(json_data, indent=2),
                file_name=f"transcription_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                mime="application/json"
            )

# ====================
# TAB 3: QUALITY CHECK
# ====================
with tab3:
    st.header("📊 Quality Check")
    
    if "transcription_result" in st.session_state:
        result = st.session_state.transcription_result
        transcript = result["text"]
        
        # Run quality check
        quality = check_transcription_quality(transcript)
        
        # ===== TEXT STATISTICS =====
        st.subheader("📏 Text Statistics")
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Word Count", quality["word_count"])
        with col2:
            st.metric("Character Count", quality["char_count"])
        with col3:
            st.metric("Sentence Count", quality["sentence_count"])
        with col4:
            st.metric("Avg Word Length", f"{quality['avg_word_length']:.2f}")
        
        # ===== QUALITY ASSESSMENT =====
        st.divider()
        st.subheader("✅ Quality Assessment")
        
        # Only fail if profanity found OR filler ratio > 10%
        has_critical_issues = quality["profanity_found"] or quality["filler_ratio_percent"] > 10
        
        if has_critical_issues:
            st.error("❌ Quality Check Failed:")
            if quality["profanity_found"]:
                st.write(f"• Profanity detected: {', '.join(quality['profanity_found'])}")
            if quality["filler_ratio_percent"] > 10:
                st.write(f"• Excessive filler words: {quality['filler_word_count']} instances ({quality['filler_ratio_percent']}% of words)")
        else:
            st.success("✅ Quality Check Passed!")
            # Show filler words as info if present but not excessive
            if quality["filler_word_count"] > 0:
                st.info(f"ℹ️ Filler words found: {quality['filler_word_count']} instances ({quality['filler_ratio_percent']}% of words) - within acceptable range")
        
        # ===== ADVANCED ANALYSIS =====
        st.divider()
        st.subheader("🔬 Advanced Analysis")
        
        if st.button("▶️ Run Analysis", type="primary"):
            with st.spinner("Analyzing transcription..."):
                nlp_results = run_full_nlp_analysis(transcript, use_transformers=False)
                st.session_state.nlp_results = nlp_results
        
        if "nlp_results" in st.session_state:
            nlp = st.session_state.nlp_results
            
            # ----- SENTIMENT ANALYSIS -----
            with st.expander("😊 Sentiment Analysis", expanded=True):
                if "sentiment" in nlp and "error" not in nlp["sentiment"]:
                    sent = nlp["sentiment"]
                    
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Sentiment", f"{sent['emoji']} {sent['label']}")
                    with col2:
                        st.metric("Polarity", f"{sent['polarity']:.2f}", 
                                  help="-1 (negative) to +1 (positive)")
                    with col3:
                        st.metric("Subjectivity", f"{sent['subjectivity']:.2f}",
                                  help="0 (objective) to 1 (subjective)")
                    
                    st.progress((sent['polarity'] + 1) / 2, text=f"Polarity Scale: {sent['polarity']:.2f}")
                else:
                    st.warning("Sentiment analysis not available.")
            
            # ----- TOXICITY DETECTION -----
            with st.expander("⚠️ Toxicity Detection", expanded=True):
                if "toxicity" in nlp:
                    tox = nlp["toxicity"]
                    
                    if tox.get("is_toxic", False):
                        st.error(tox.get("status", "🔴 Toxic Content Detected"))
                        
                        if "toxic_words_found" in tox:
                            for category, words in tox["toxic_words_found"].items():
                                st.write(f"**{category.title()}:** {', '.join(words)}")
                        
                        col1, col2 = st.columns(2)
                        with col1:
                            st.metric("Total Toxic Words", tox.get("total_toxic_count", 0))
                        with col2:
                            st.metric("Toxicity Score", f"{tox.get('toxicity_score', 0):.1%}")
                    else:
                        st.success(tox.get("status", "🟢 Clean Content"))
            
            # ----- NAMED ENTITIES -----
            with st.expander("🏷️ Named Entities"):
                if "entities" in nlp and "error" not in nlp["entities"]:
                    ent = nlp["entities"]
                    
                    st.metric("Total Entities Found", ent["entity_count"])
                    
                    if ent["entities"]:
                        for entity_type, entities in ent["entities"].items():
                            desc = ent["descriptions"].get(entity_type, entity_type)
                            st.write(f"**{desc}:** {', '.join(entities)}")
                    else:
                        st.info("No named entities found in the text.")
                else:
                    st.warning("Entity recognition not available.")
            
            # ----- KEYWORDS -----
            with st.expander("🔑 Keywords (TF-IDF)"):
                if "keywords" in nlp and "error" not in nlp["keywords"]:
                    kw = nlp["keywords"]
                    
                    if kw["keywords"]:
                        kw_df = pd.DataFrame(kw["keywords"], columns=["Keyword", "Score"])
                        
                        col1, col2 = st.columns([2, 1])
                        with col1:
                            st.bar_chart(kw_df.set_index("Keyword"))
                        with col2:
                            st.dataframe(kw_df, use_container_width=True)
                else:
                    st.warning("Keyword extraction not available.")
            
            # ----- SUMMARIZATION -----
            with st.expander("📝 Summary"):
                if "summary" in nlp:
                    summ = nlp["summary"]
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("Original Sentences", summ.get("original_sentences", "N/A"))
                    with col2:
                        st.metric("Compression Ratio", f"{summ.get('compression_ratio', 0):.0%}")
                    
                    st.info(summ["summary"])
            
            # ----- READABILITY -----
            with st.expander("📊 Readability Scores"):
                if "readability" in nlp:
                    read = nlp["readability"]
                    
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("Flesch Reading Ease", read["flesch_reading_ease"],
                                  help="0-100, higher = easier")
                    with col2:
                        st.metric("Flesch-Kincaid Grade", read["flesch_kincaid_grade"],
                                  help="US grade level")
                    with col3:
                        st.metric("Gunning Fog Index", read["gunning_fog_index"],
                                  help="Years of education needed")
                    with col4:
                        st.metric("SMOG Index", read["smog_index"],
                                  help="Years of education needed")
                    
                    st.info(f"**Reading Level:** {read['reading_level']}")
        
        # ===== TOP WORDS =====
        st.divider()
        st.subheader("🔤 Top Words")
        from collections import Counter
        import re
        
        words = re.findall(r'\b\w+\b', transcript.lower())
        stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'is', 'was', 'are', 'be', 'been', 'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should', 'may', 'might', 'must', 'can', 'this', 'that', 'these', 'those', 'i', 'you', 'he', 'she', 'it', 'we', 'they'}
        words = [w for w in words if w not in stop_words and len(w) > 2]
        
        word_freq = Counter(words).most_common(15)
        if word_freq:
            freq_df = pd.DataFrame(word_freq, columns=["Word", "Frequency"])
            col1, col2 = st.columns([2, 1])
            with col1:
                st.bar_chart(freq_df.set_index("Word"))
            with col2:
                st.dataframe(freq_df, use_container_width=True)
        
        # ===== AI MONETIZATION ADVICE =====
        st.divider()
        st.subheader("💰 AI Monetization Advice")
        
        if GEMINI_AVAILABLE:
            if st.button("🤖 Get AI Advice", type="secondary"):
                with st.spinner("Getting AI advice..."):
                    try:
                        genai.configure(api_key="YOUR_API_KEY_HERE")
                        model = genai.GenerativeModel("models/gemini-2.5-flash")
                        
                        prompt = f"""Analyze this video transcription and provide:

1. **2-LINE SUMMARY**: A brief 2-line summary of the content.

2. **WORDS TO AVOID**: List any profanity, inappropriate language, or demonetization-risk words found. For each word, suggest a safe alternative.

3. **MONETIZATION ADVICE**: Give 3-5 specific tips to make this video more YouTube-monetization friendly.

Transcription:
{transcript[:3000]}

Format your response clearly with headers."""
                        
                        response = model.generate_content(prompt)
                        st.session_state.ai_advice = response.text
                    except Exception as e:
                        st.error(f"AI Error: {str(e)}")
            
            if "ai_advice" in st.session_state:
                st.markdown(st.session_state.ai_advice)
        else:
            st.warning("Install google-generativeai: `pip install google-generativeai`")
        
    else:
        st.info("👈 Please transcribe audio first to check quality")

# Footer
st.divider()
st.markdown(
    """
    ---
    **Audio Transcription Tool** | Local Processing | Advanced Text Analysis
    
    Built with Streamlit, Faster-Whisper, spaCy, TextBlob & scikit-learn
    """
)
