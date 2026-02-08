import os
# Fix OpenMP conflict error
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

from pathlib import Path
from typing import Dict, Any, List
import logging
import math
from model_manager import MODEL_CACHE_DIR

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def _format_ts(t: float) -> str:
    """Format timestamp for SRT format"""
    h = int(t // 3600)
    m = int((t % 3600) // 60)
    s = int(t % 60)
    ms = int((t - math.floor(t)) * 1000)
    return f"{h:02}:{m:02}:{s:02},{ms:03}"

def transcribe_audio_local(
    audio_path: Path | str,
    model_size: str = "large-v3",
    device: str = "auto",
    compute_type: str = "float16",
    vad_filter: bool = True,
) -> Dict[str, Any]:
    """
    Transcribe audio locally using faster-whisper
    
    Args:
        audio_path: Path to audio file
        model_size: Whisper model size (tiny, base, small, medium, large, large-v3)
        device: 'auto', 'cuda', or 'cpu'
        compute_type: 'float16', 'int8', 'int8_float16', or 'float32'
        vad_filter: Use VAD filter to skip silent segments
    
    Returns:
        Dict with keys: text, chunks (with start/end/text), srt, srt_segments
    """
    try:
        from faster_whisper import WhisperModel
    except ImportError:
        raise RuntimeError("faster-whisper not installed. Install with: pip install faster-whisper")
    
    audio_path = Path(audio_path)
    if not audio_path.exists():
        raise FileNotFoundError(f"Audio file not found: {audio_path}")
    
    # Device selection
    dev = device
    if device == "auto":
        try:
            import torch
            if torch.cuda.is_available():
                dev = "cuda"
                logger.info("Using CUDA GPU for transcription")
            else:
                dev = "cpu"
                logger.info("Using CPU for transcription")
        except Exception:
            dev = "cpu"
            logger.info("Using CPU for transcription (torch not available)")
    
    logger.info(f"Loading Whisper model '{model_size}' on {dev} ({compute_type})")
    
    try:
        model = WhisperModel(
            model_size, 
            device=dev, 
            compute_type=compute_type,
            download_root=str(MODEL_CACHE_DIR)
        )
    except Exception as e:
        logger.warning(f"Error loading with {compute_type}, falling back to float32: {e}")
        model = WhisperModel(
            model_size, 
            device=dev, 
            compute_type="float32",
            download_root=str(MODEL_CACHE_DIR)
        )
    
    logger.info(f"Transcribing: {audio_path.name}")
    
    # Run transcription
    segments, info = model.transcribe(
        str(audio_path),
        vad_filter=vad_filter,
        word_timestamps=True,
        language=None,
    )
    
    # Process segments
    seg_list: List[dict] = []
    srt_lines: List[str] = []
    srt_segments: List[dict] = []
    idx = 1
    full_text_parts: List[str] = []
    
    for seg in segments:
        start = float(seg.start or 0.0)
        end = float(seg.end or 0.0)
        text = seg.text.strip() if seg.text else ""
        
        if text:  # Only add non-empty segments
            seg_list.append({"start": start, "end": end, "text": text})
            full_text_parts.append(text)
            srt_segments.append({"index": idx, "start": start, "end": end, "text": text})
            
            srt_lines.append(str(idx))
            srt_lines.append(f"{_format_ts(start)} --> {_format_ts(end)}")
            srt_lines.append(text)
            srt_lines.append("")
            idx += 1
    
    text_full = " ".join(full_text_parts).strip()
    srt_text = "\n".join(srt_lines)
    
    logger.info(f"Transcription complete: {len(text_full)} chars, {len(seg_list)} segments")
    
    return {
        "text": text_full,
        "chunks": seg_list,
        "srt": srt_text,
        "srt_segments": srt_segments,
        "language": info.language if hasattr(info, 'language') else "unknown"
    }

def check_transcription_quality(transcription: str) -> Dict[str, Any]:
    """
    Check quality metrics of transcription including profanity and filler words
    
    Returns:
        Dict with quality metrics
    """
    import re
    
    word_count = len(transcription.split())
    char_count = len(transcription)
    sentence_count = len(re.split(r'[.!?]+', transcription)) - 1
    
    # Check for common issues (critical quality issues only)
    issues = []
    
    # Check if mostly uppercase (might indicate issues)
    uppercase_ratio = sum(1 for c in transcription if c.isupper()) / max(char_count, 1)
    if uppercase_ratio > 0.8:
        issues.append("High uppercase ratio - may indicate transcription issues")
    
    # Check if has minimal content
    if word_count < 10:
        issues.append("Very short transcription - may indicate audio quality issues")
    
    # Check for profanity and inappropriate words (informational only, not quality issues)
    profanity_list = ["shit", "fuck", "fucking", "damn", "hell", "ass", "bitch", "bastard"]
    filler_words = ["like", "just", "really", "actually", "basically", "literally", "you know", "i mean"]
    
    # Convert to lowercase for checking
    text_lower = transcription.lower()
    words_lower = text_lower.split()
    
    # Find profanity (informational)
    found_profanity = []
    for word in profanity_list:
        if re.search(r'\b' + re.escape(word) + r'\b', text_lower):
            count = len(re.findall(r'\b' + re.escape(word) + r'\b', text_lower))
            found_profanity.append(f"{word} ({count}x)")
    
    if found_profanity:
        issues.append(f"Profanity detected: {', '.join(found_profanity)}")
    
    # Count excessive filler words (informational)
    filler_count = 0
    found_fillers = []
    for filler in filler_words:
        if ' ' in filler:  # Multi-word fillers
            count = text_lower.count(filler)
        else:
            count = len(re.findall(r'\b' + re.escape(filler) + r'\b', text_lower))
        
        if count > 0:
            filler_count += count
            if count >= 5:  # Only report if used 5+ times
                found_fillers.append(f"{filler} ({count}x)")
    
    if found_fillers:
        issues.append(f"Excessive filler words: {', '.join(found_fillers)}")
    
    # Calculate filler ratio
    filler_ratio = (filler_count / max(word_count, 1)) * 100
    
    return {
        "word_count": word_count,
        "char_count": char_count,
        "sentence_count": max(0, sentence_count),
        "avg_word_length": char_count / max(word_count, 1),
        "profanity_found": found_profanity,
        "filler_word_count": filler_count,
        "filler_ratio_percent": round(filler_ratio, 1),
        "quality_issues": issues,
        "estimated_quality": "good" if not issues else "warning"
    }
