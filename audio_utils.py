import subprocess
from pathlib import Path
import shutil
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def find_ffmpeg():
    """Find ffmpeg in common Windows locations or system PATH"""
    paths = [
        'C:\\ffmpeg\\ffmpeg-8.0.1-essentials_build\\bin\\ffmpeg.exe',
        'C:\\ffmpeg\\bin\\ffmpeg.exe',
    ]
    for p in paths:
        if Path(p).exists():
            logger.info(f"Found ffmpeg at: {p}")
            return str(p)
    
    ffmpeg = shutil.which('ffmpeg')
    if ffmpeg:
        logger.info(f"Found ffmpeg in PATH: {ffmpeg}")
        return ffmpeg
    
    logger.warning("ffmpeg not found - trying default 'ffmpeg'")
    return 'ffmpeg'

FFMPEG = find_ffmpeg()

def extract_audio(video_path: str | Path, output_dir: str | Path = None, bitrate_k: str = '160') -> Path:
    """Extract audio from video file"""
    video_path = Path(video_path)
    if output_dir is None:
        output_dir = video_path.parent / "audio_temp"
    else:
        output_dir = Path(output_dir)
    
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{video_path.stem}.mp3"
    
    logger.info(f"Extracting audio from: {video_path}")
    cmd = [FFMPEG, '-y', '-i', str(video_path), '-vn', '-acodec', 'libmp3lame', 
           '-ab', f'{bitrate_k}k', '-ac', '2', str(output_path)]
    
    try:
        subprocess.run(cmd, check=True, capture_output=True)
        logger.info(f"Audio extracted: {output_path}")
        return output_path
    except subprocess.CalledProcessError as e:
        logger.error(f"Error extracting audio: {e}")
        raise

def download_youtube_video(url: str, output_dir: str | Path = None) -> Path:
    """Download audio (MP3) from YouTube URL"""
    if output_dir is None:
        output_dir = Path("downloads")
    else:
        output_dir = Path(output_dir)
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Downloading audio from YouTube: {url}")
    
    try:
        import yt_dlp
        ydl_opts = {
            'format': 'bestaudio/best',
            'outtmpl': str(output_dir / '%(title)s.%(ext)s'),
            'postprocessors': [{
                'key': 'FFmpegExtractAudio',
                'preferredcodec': 'mp3',
                'preferredquality': '192',
            }],
            'quiet': False,
            'no_warnings': False,
        }
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=True)
            # After postprocessing, the file will have .mp3 extension
            audio_path = output_dir / f"{info['title']}.mp3"
            logger.info(f"Downloaded audio: {audio_path}")
            return audio_path
    except Exception as e:
        logger.error(f"Error downloading audio: {e}")
        raise
