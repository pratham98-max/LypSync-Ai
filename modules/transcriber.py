"""
Speech-to-Text Transcription Module using OpenAI Whisper
Extracts English transcript with timestamps from input audio/video
"""

import os
import torch
import whisper
from typing import List, Dict, Any, Optional
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class Transcriber:
    """
    Whisper-based speech-to-text transcription with timestamp support.
    """
    
    SUPPORTED_MODELS = ["tiny", "base", "small", "medium", "large", "large-v2", "large-v3"]
    
    def __init__(self, model_name: str = "large-v3", device: Optional[str] = None):
        """
        Initialize the Whisper transcriber.
        
        Args:
            model_name: Whisper model size (tiny, base, small, medium, large, large-v3)
            device: Device to use ('cuda' or 'cpu'). Auto-detected if None.
        """
        if model_name not in self.SUPPORTED_MODELS:
            raise ValueError(f"Model must be one of: {self.SUPPORTED_MODELS}")
        
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Loading Whisper model '{model_name}' on {self.device}...")
        
        self.model = whisper.load_model(model_name, device=self.device)
        logger.info("Whisper model loaded successfully.")
    
    def transcribe(
        self, 
        audio_path: str, 
        language: str = "en",
        word_timestamps: bool = True
    ) -> Dict[str, Any]:
        """
        Transcribe audio/video file to text with timestamps.
        
        Args:
            audio_path: Path to audio or video file
            language: Source language code (default: 'en' for English)
            word_timestamps: Whether to include word-level timestamps
            
        Returns:
            Dictionary containing:
                - text: Full transcription text
                - segments: List of segments with start, end, text
                - language: Detected/specified language
        """
        if not os.path.exists(audio_path):
            raise FileNotFoundError(f"Audio file not found: {audio_path}")
        
        logger.info(f"Transcribing: {audio_path}")
        
        result = self.model.transcribe(
            audio_path,
            language=language,
            word_timestamps=word_timestamps,
            verbose=False
        )
        
        # Process segments for easier downstream use
        processed_segments = []
        for segment in result["segments"]:
            processed_segment = {
                "id": segment["id"],
                "start": segment["start"],
                "end": segment["end"],
                "text": segment["text"].strip(),
            }
            
            # Include word-level timestamps if available
            if word_timestamps and "words" in segment:
                processed_segment["words"] = [
                    {
                        "word": w["word"].strip(),
                        "start": w["start"],
                        "end": w["end"]
                    }
                    for w in segment["words"]
                ]
            
            processed_segments.append(processed_segment)
        
        transcription = {
            "text": result["text"].strip(),
            "segments": processed_segments,
            "language": result.get("language", language),
            "duration": processed_segments[-1]["end"] if processed_segments else 0
        }
        
        logger.info(f"Transcription complete. Duration: {transcription['duration']:.2f}s, "
                   f"Segments: {len(processed_segments)}")
        
        return transcription
    
    def get_text_with_timestamps(self, transcription: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Get a simplified list of text chunks with timestamps for translation.
        
        Args:
            transcription: Output from transcribe()
            
        Returns:
            List of dicts with 'text', 'start', 'end' for each segment
        """
        return [
            {
                "text": seg["text"],
                "start": seg["start"],
                "end": seg["end"]
            }
            for seg in transcription["segments"]
        ]


def extract_audio_from_video(video_path: str, output_path: str) -> str:
    """
    Extract audio track from video file.
    
    Args:
        video_path: Path to input video
        output_path: Path for output audio file
        
    Returns:
        Path to extracted audio file
    """
    import subprocess
    
    # Check if video exists
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video file not found: {video_path}")
    
    # Try to use imageio-ffmpeg bundled binary first
    try:
        import imageio_ffmpeg
        ffmpeg_path = imageio_ffmpeg.get_ffmpeg_exe()
    except ImportError:
        ffmpeg_path = "ffmpeg"  # Fall back to system ffmpeg
    
    cmd = [
        ffmpeg_path, "-y",
        "-i", video_path,
        "-vn",  # No video
        "-acodec", "pcm_s16le",
        "-ar", "16000",  # 16kHz for Whisper
        "-ac", "1",  # Mono
        output_path
    ]
    
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        logger.info(f"Extracted audio to: {output_path}")
    except subprocess.CalledProcessError as e:
        logger.error(f"FFmpeg error: {e.stderr}")
        raise RuntimeError(f"Failed to extract audio: {e.stderr}")
    except FileNotFoundError:
        raise FileNotFoundError(
            "FFmpeg not found. Install with: pip install imageio-ffmpeg OR "
            "download from https://ffmpeg.org/download.html"
        )
    
    return output_path


if __name__ == "__main__":
    # Test transcriber
    transcriber = Transcriber(model_name="base")
    print("Transcriber initialized successfully!")
