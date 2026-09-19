"""
Voice Cloning Module using Coqui TTS (XTTS-v2)
Zero-shot voice cloning with cross-language support
"""

import os

os.environ.setdefault("COQUI_TOS_AGREED", "1")
if os.path.exists("D:\\ai_cache"):
    os.environ.setdefault("TTS_HOME", r"D:\ai_cache\tts")
    os.environ.setdefault("HF_HOME", r"D:\ai_cache\huggingface")
    os.environ.setdefault("TORCH_HOME", r"D:\ai_cache\torch")

import torch
import torchaudio
from TTS.api import TTS
from typing import Optional, List, Dict
import logging
import numpy as np

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class VoiceCloner:
    """
    XTTS-v2 based voice cloning with multilingual support.
    Requires only 3-6 seconds of reference audio for zero-shot cloning.
    """
    
    # XTTS-v2 supported languages
    SUPPORTED_LANGUAGES = {
        "hi": "hi",   # Hindi (native support)
        "ta": "ta",   # Tamil
        "te": "te",   # Telugu  
        "bn": "bn",   # Bengali
        "mr": "mr",   # Marathi
        "gu": "gu",   # Gujarati
        "kn": "kn",   # Kannada
        "ml": "ml",   # Malayalam
        "pa": "pa",   # Punjabi
        "en": "en",   # English
    }
    
    def __init__(self, device: Optional[str] = None, model_name: str = "tts_models/multilingual/multi-dataset/xtts_v2"):
        """
        Initialize the XTTS-v2 voice cloner.
        
        Args:
            device: Device to use ('cuda' or 'cpu'). Auto-detected if None.
            model_name: TTS model identifier
        """
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model_name = model_name
        
        logger.info(f"Loading XTTS-v2 model on {self.device}...")
        
        self.tts = TTS(model_name=model_name).to(self.device)
        
        logger.info("XTTS-v2 model loaded successfully.")
        
        self.sample_rate = 22050  # XTTS default sample rate
    
    def clone_voice(
        self,
        text: str,
        speaker_wav: str,
        language: str,
        output_path: Optional[str] = None
    ) -> np.ndarray:
        """
        Generate speech in cloned voice for given text.
        
        Args:
            text: Text to synthesize
            speaker_wav: Path to reference audio for voice cloning (3-6+ seconds)
            language: Target language code
            output_path: Optional path to save the generated audio
            
        Returns:
            Generated audio as numpy array
        """
        if not os.path.exists(speaker_wav):
            raise FileNotFoundError(f"Speaker reference file not found: {speaker_wav}")
        
        if language not in self.SUPPORTED_LANGUAGES:
            logger.warning(f"Language '{language}' may not be fully supported. Attempting anyway...")
            lang_code = language
        else:
            lang_code = self.SUPPORTED_LANGUAGES[language]
        
        logger.info(f"Generating speech in {language} with cloned voice...")
        
        # Generate speech
        if output_path:
            self.tts.tts_to_file(
                text=text,
                speaker_wav=speaker_wav,
                language=lang_code,
                file_path=output_path
            )
            
            # Load and return the audio
            waveform, sr = torchaudio.load(output_path)
            audio = waveform.numpy().squeeze()
        else:
            audio = self.tts.tts(
                text=text,
                speaker_wav=speaker_wav,
                language=lang_code
            )
            audio = np.array(audio)
        
        logger.info(f"Generated {len(audio) / self.sample_rate:.2f}s of audio")
        
        return audio
    
    def synthesize_segments(
        self,
        segments: List[Dict],
        speaker_wav: str,
        language: str,
        output_dir: str
    ) -> List[Dict]:
        """
        Synthesize multiple text segments with timestamp alignment.
        
        Args:
            segments: List of dicts with 'text', 'start', 'end' keys
            speaker_wav: Path to reference audio for voice cloning
            language: Target language code
            output_dir: Directory to save individual audio segments
            
        Returns:
            List of segments with 'audio_path' added
        """
        os.makedirs(output_dir, exist_ok=True)
        
        synthesized_segments = []
        
        for i, seg in enumerate(segments):
            output_path = seg.get("audio_path") or os.path.join(output_dir, f"segment_{i:04d}.wav")
            
            # Skip if already exists
            if os.path.exists(output_path):
                logger.info(f"Segment {i} already exists, skipping synthesis.")
                try:
                    waveform, sr = torchaudio.load(output_path)
                    synthesized_segments.append({
                        **seg,
                        "audio_path": output_path,
                        "audio_duration": waveform.shape[1] / sr
                    })
                    continue
                except Exception as e:
                    logger.warning(f"Existing segment {i} is invalid, re-synthesizing: {e}")
            
            try:
                audio = self.clone_voice(
                    text=seg["text"],
                    speaker_wav=speaker_wav,
                    language=language,
                    output_path=output_path
                )
                
                synthesized_segments.append({
                    **seg,
                    "audio_path": output_path,
                    "audio_duration": len(audio) / self.sample_rate
                })
                
            except Exception as e:
                logger.error(f"Failed to synthesize segment {i}: {e}")
                synthesized_segments.append({
                    **seg,
                    "audio_path": None,
                    "error": str(e)
                })
        
        logger.info(f"Synthesized {len([s for s in synthesized_segments if s.get('audio_path')])} / {len(segments)} segments")
        
        return synthesized_segments
    
    def adjust_audio_speed(
        self,
        audio_path: str,
        target_duration: float,
        output_path: str,
        max_speed: float = 1.2
    ) -> float:
        """
        Adjust audio speed to match target duration, with a safety cap.
        
        Returns:
            Actual duration of the resulting audio file.
        """
        waveform, sr = torchaudio.load(audio_path)
        current_duration = waveform.shape[1] / sr
        
        if target_duration < 0.01:
            torchaudio.save(output_path, waveform, sr)
            return current_duration
            
        # Calculate speed factor
        speed_factor = current_duration / target_duration
        
        # CAP the speed factor to keep it natural
        if speed_factor > max_speed:
            logger.info(f"Speed factor {speed_factor:.2f}x exceeds limit. Capping at {max_speed}x.")
            speed_factor = max_speed
        
        if abs(speed_factor - 1.0) < 0.01:
            torchaudio.save(output_path, waveform, sr)
            return current_duration

        import subprocess
        cmd = [
            "ffmpeg", "-y",
            "-i", audio_path,
            "-filter:a", f"atempo={speed_factor}",
            output_path
        ]
        subprocess.run(cmd, check=True, capture_output=True)
        
        actual_duration = current_duration / speed_factor
        logger.info(f"Adjusted audio speed to {speed_factor:.2f}x. Final duration: {actual_duration:.2f}s")
        
        return actual_duration

    @classmethod
    def get_supported_languages(cls) -> List[str]:
        """Get list of supported language codes."""
        return list(cls.SUPPORTED_LANGUAGES.keys())


if __name__ == "__main__":
    # Test voice cloner initialization
    cloner = VoiceCloner()
    print("VoiceCloner initialized successfully!")
    print(f"Supported languages: {cloner.get_supported_languages()}")
