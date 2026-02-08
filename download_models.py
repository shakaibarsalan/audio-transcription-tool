#!/usr/bin/env python3
"""
Setup script to pre-download Whisper models locally
Run this once to download models, then they'll be cached for offline use
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from model_manager import download_all_models, get_models_status, get_model_cache_path


def main():
    print("\n" + "=" * 70)
    print(" " * 15 + "Whisper Model Pre-Download Setup")
    print("=" * 70)
    
    print(f"\nModels will be cached at: {get_model_cache_path()}")
    
    # Check current status
    print("\n" + "-" * 70)
    print("Current Model Status:")
    print("-" * 70)
    
    status = get_models_status()
    for model, info in status.items():
        status_str = "✓ Downloaded" if info["downloaded"] else "✗ Not downloaded"
        print(f"  {model:12} - {status_str}")
    
    # Ask which models to download
    print("\n" + "-" * 70)
    print("Recommended to download:")
    print("-" * 70)
    print("  base  (140 MB)  - Fast, reasonable quality")
    print("  small (240 MB)  - Better quality, slower")
    print("  large (2.9 GB)  - Best quality, much slower")
    
    user_input = input("\nEnter models to download (comma-separated, e.g. 'base,small'): ").strip()
    
    if not user_input:
        print("No models selected. Exiting.")
        return
    
    models = [m.strip() for m in user_input.split(",")]
    
    print("\n" + "=" * 70)
    print("Starting download... This may take several minutes.")
    print("=" * 70 + "\n")
    
    results = download_all_models(models)
    
    print("\n" + "=" * 70)
    print("Download Summary:")
    print("=" * 70)
    
    for model, success in results.items():
        status_str = "✓ Success" if success else "✗ Failed"
        print(f"  {model:12} - {status_str}")
    
    # Final status
    print("\n" + "-" * 70)
    print("Final Model Status:")
    print("-" * 70)
    
    status = get_models_status()
    for model, info in status.items():
        status_str = "✓ Downloaded" if info["downloaded"] else "✗ Not downloaded"
        print(f"  {model:12} - {status_str}")
    
    print("\n" + "=" * 70)
    print("✓ Setup complete! Models are now cached for offline use.")
    print("  You can now run the transcription app without re-downloading.")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
