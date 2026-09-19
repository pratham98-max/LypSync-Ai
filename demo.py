"""
Demo script showing the pipeline with sample data
Creates a test video and demonstrates the full workflow
"""

import os
import sys
import numpy as np
import cv2
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding='utf-8')
        sys.stderr.reconfigure(encoding='utf-8')
    except Exception:
        pass

os.environ.setdefault("COQUI_TOS_AGREED", "1")
if os.path.exists("D:\\ai_cache"):
    os.environ.setdefault("TTS_HOME", r"D:\ai_cache\tts")
    os.environ.setdefault("HF_HOME", r"D:\ai_cache\huggingface")
    os.environ.setdefault("TORCH_HOME", r"D:\ai_cache\torch")


def create_sample_video(output_path: str, duration: int = 5, fps: int = 30):
    """Create a simple test video with a face placeholder."""
    
    width, height = 640, 480
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    total_frames = duration * fps
    
    for i in range(total_frames):
        # Create a frame with gradient background
        frame = np.zeros((height, width, 3), dtype=np.uint8)
        
        # Add gradient
        for y in range(height):
            frame[y, :] = [int(50 + y * 0.3), int(50 + y * 0.1), int(100 + y * 0.2)]
        
        # Draw a face placeholder (circle)
        center = (width // 2, height // 2)
        cv2.circle(frame, center, 100, (200, 180, 160), -1)  # Face
        cv2.circle(frame, (center[0] - 30, center[1] - 20), 15, (255, 255, 255), -1)  # Left eye
        cv2.circle(frame, (center[0] + 30, center[1] - 20), 15, (255, 255, 255), -1)  # Right eye
        cv2.circle(frame, (center[0] - 30, center[1] - 20), 7, (50, 50, 50), -1)  # Left pupil
        cv2.circle(frame, (center[0] + 30, center[1] - 20), 7, (50, 50, 50), -1)  # Right pupil
        
        # Animated mouth
        mouth_open = int(10 + 15 * np.sin(i * 0.3))
        cv2.ellipse(frame, (center[0], center[1] + 40), (30, mouth_open), 0, 0, 180, (100, 50, 50), -1)
        
        # Add text
        cv2.putText(frame, f"Sample Video - Frame {i+1}/{total_frames}", 
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        out.write(frame)
    
    out.release()
    print(f"Created sample video: {output_path}")
    return output_path


def create_sample_audio(output_path: str, duration: float = 5.0, sample_rate: int = 22050):
    """Create a simple test audio file (sine wave)."""
    import scipy.io.wavfile as wav
    
    t = np.linspace(0, duration, int(sample_rate * duration))
    # Create a simple tone
    frequency = 440  # A4 note
    audio = (np.sin(2 * np.pi * frequency * t) * 0.3 * 32767).astype(np.int16)
    
    wav.write(output_path, sample_rate, audio)
    print(f"Created sample audio: {output_path}")
    return output_path


def test_modules():
    """Test individual modules."""
    print("=" * 60)
    print("TESTING INDIVIDUAL MODULES")
    print("=" * 60)
    
    # Test transcriber
    print("\n[1] Testing Transcriber...")
    try:
        from modules.transcriber import Transcriber
        transcriber = Transcriber(model_name="base")  # Use base for faster testing
        print("✅ Transcriber initialized successfully")
    except Exception as e:
        print(f"❌ Transcriber failed: {e}")
    
    # Test translator
    print("\n[2] Testing Translator...")
    try:
        from modules.translator import IndicTranslator
        print("   Loading IndicTrans2 model (this may take a while)...")
        translator = IndicTranslator()
        
        # Test translation
        test_text = "Hello, how are you?"
        result = translator.translate(test_text, "hi")
        print(f"   English: {test_text}")
        print(f"   Hindi: {result}")
        print("✅ Translator working correctly")
    except Exception as e:
        print(f"❌ Translator failed: {e}")
    
    # Test voice cloner
    print("\n[3] Testing Voice Cloner...")
    try:
        from modules.voice_cloner import VoiceCloner
        print("   Loading XTTS-v2 model (this may take a while)...")
        cloner = VoiceCloner()
        print(f"   Supported languages: {cloner.get_supported_languages()}")
        print("✅ Voice Cloner initialized successfully")
    except Exception as e:
        print(f"❌ Voice Cloner failed: {e}")
    
    # Test lip syncer
    print("\n[4] Testing Lip Syncer...")
    try:
        from modules.lip_sync import LipSyncer
        syncer = LipSyncer(model_type="wav2lip")
        print("✅ Lip Syncer initialized successfully")
    except Exception as e:
        print(f"❌ Lip Syncer failed: {e}")
    
    print("\n" + "=" * 60)
    print("MODULE TESTING COMPLETE")
    print("=" * 60)


def demo_pipeline():
    """Run a demonstration of the full pipeline."""
    print("=" * 60)
    print("DEMO: Full Pipeline Test")
    print("=" * 60)
    
    # Create sample files
    samples_dir = "samples"
    os.makedirs(samples_dir, exist_ok=True)
    
    video_path = os.path.join(samples_dir, "demo_video.mp4")
    audio_path = os.path.join(samples_dir, "demo_voice.wav")
    
    # Create samples
    create_sample_video(video_path)
    create_sample_audio(audio_path)
    
    print("\nSample files created. To run the full pipeline:")
    print(f"1. Replace {video_path} with a real video containing a speaking person")
    print(f"2. Replace {audio_path} with a voice sample (3-6 seconds)")
    print("3. Run: python run.py --config config.yaml")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Demo and testing utilities")
    parser.add_argument("--test-modules", action="store_true", help="Test individual modules")
    parser.add_argument("--create-samples", action="store_true", help="Create sample video/audio")
    parser.add_argument("--demo", action="store_true", help="Run full demo")
    
    args = parser.parse_args()
    
    if args.test_modules:
        test_modules()
    elif args.create_samples or args.demo:
        demo_pipeline()
    else:
        print("Usage:")
        print("  python demo.py --test-modules    # Test individual modules")
        print("  python demo.py --create-samples  # Create sample files")
        print("  python demo.py --demo            # Run full demo")
