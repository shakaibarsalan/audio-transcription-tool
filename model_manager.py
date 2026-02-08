import os
import logging
from pathlib import Path
from typing import List, Optional

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Set model cache directory
MODEL_CACHE_DIR = Path.home() / ".cache" / "whisper-models"
os.environ["CT2_HOME"] = str(MODEL_CACHE_DIR)
REQUIRED_FILES = ["config.json", "model.bin", "tokenizer.json"]
VOCAB_CANDIDATES = ["vocabulary.json", "vocabulary.txt"]


def get_available_models() -> List[str]:
    """Get list of Whisper models available"""
    return ["tiny", "base", "small", "medium", "large", "large-v3"]


def get_model_cache_path() -> Path:
    """Get the directory where models are cached"""
    return MODEL_CACHE_DIR


def is_model_downloaded(model_size: str) -> bool:
    """Fast check: verify required files exist in cache directory without loading the model."""
    # faster-whisper stores models under models--Systran--faster-whisper-<size>/snapshots/<hash>
    import glob
    
    pattern = str(MODEL_CACHE_DIR / f"models--Systran--faster-whisper-{model_size}" / "snapshots" / "*" / "config.json")
    config_files = glob.glob(pattern)
    
    if not config_files:
        return False
    
    # If config.json exists and has content, model is cached
    for config_file in config_files:
        if Path(config_file).stat().st_size > 0:
            return True
    
    return False


def download_model(model_size: str) -> bool:
    """
    Download a Whisper model for offline use
    
    Args:
        model_size: Model size to download (tiny, base, small, medium, large, large-v3)
    
    Returns:
        True if successful, False otherwise
    """
    if model_size not in get_available_models():
        logger.error(f"Invalid model size: {model_size}. Available: {get_available_models()}")
        return False
    
    # Create cache directory if it doesn't exist
    MODEL_CACHE_DIR.mkdir(parents=True, exist_ok=True)

    if is_model_downloaded(model_size):
        logger.info(f"Model '{model_size}' already cached")
        return True
    
    try:
        from faster_whisper import WhisperModel
        
        logger.info(f"Downloading model '{model_size}' to {MODEL_CACHE_DIR}...")
        logger.info("This may take a few minutes on first run...")
        
        WhisperModel(
            model_size,
            device="cpu",  # Use CPU for downloading
            download_root=str(MODEL_CACHE_DIR)
        )
        
        logger.info(f"✓ Model '{model_size}' downloaded successfully")
        return True
        
    except Exception as e:
        logger.error(f"Failed to download model '{model_size}': {e}")
        return False


def download_all_models(models: Optional[List[str]] = None) -> dict:
    """
    Download multiple models
    
    Args:
        models: List of model sizes to download. If None, downloads default models
    
    Returns:
        Dict with download results
    """
    if models is None:
        models = ["base", "small"]  # Default models
    
    results = {}
    for model in models:
        results[model] = download_model(model)
    
    return results


def get_models_status() -> dict:
    """Get status of all available models"""
    status = {}
    for model in get_available_models():
        status[model] = {
            "downloaded": is_model_downloaded(model),
            "cache_path": str(MODEL_CACHE_DIR / model)
        }
    return status


if __name__ == "__main__":
    # Script to download models
    import sys
    
    print("=" * 60)
    print("Whisper Model Manager")
    print("=" * 60)
    print(f"\nCache directory: {MODEL_CACHE_DIR}")
    print("\nAvailable models:", ", ".join(get_available_models()))
    
    print("\n" + "=" * 60)
    print("Current Status:")
    print("=" * 60)
    
    status = get_models_status()
    for model, info in status.items():
        status_str = "✓ Downloaded" if info["downloaded"] else "✗ Not downloaded"
        print(f"{model:15} - {status_str}")
    
    # If arguments provided, download those models
    if len(sys.argv) > 1:
        models_to_download = sys.argv[1:]
        print("\n" + "=" * 60)
        print("Downloading models...")
        print("=" * 60)
        
        results = download_all_models(models_to_download)
        
        print("\n" + "=" * 60)
        print("Download Summary:")
        print("=" * 60)
        for model, success in results.items():
            status_str = "✓ Success" if success else "✗ Failed"
            print(f"{model:15} - {status_str}")
    else:
        print("\n" + "=" * 60)
        print("Usage: python model_manager.py [model1] [model2] ...")
        print("=" * 60)
        print("Example: python model_manager.py base small")
        print("         python model_manager.py large")
