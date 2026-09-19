"""
Lip Sync Module using Wav2Lip
Generates lip-synced video from input video and audio
"""

import os
import sys
import subprocess
import torch
import cv2
import numpy as np
from typing import Optional, Tuple
import logging
import tempfile
import shutil

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class LipSyncer:
    """
    Wav2Lip-based lip synchronization.
    Takes video and audio to produce lip-synced output.
    """
    
    # Wav2Lip model URLs
    WAV2LIP_MODELS = {
        "wav2lip": "https://github.com/justinjohn0306/Wav2Lip/releases/download/models/wav2lip.pth",
        "wav2lip_gan": "https://github.com/justinjohn0306/Wav2Lip/releases/download/models/wav2lip_gan.pth"
    }
    
    def __init__(
        self, 
        model_type: str = "wav2lip",
        device: Optional[str] = None,
        models_dir: str = "models"
    ):
        """
        Initialize the Wav2Lip lip syncer.
        
        Args:
            model_type: Model variant ('wav2lip' or 'wav2lip_gan')
            device: Device to use ('cuda' or 'cpu'). Auto-detected if None.
            models_dir: Directory to store downloaded models
        """
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model_type = model_type
        self.models_dir = models_dir
        
        os.makedirs(models_dir, exist_ok=True)
        
        logger.info(f"LipSyncer initialized with {model_type} on {self.device}")
        
        # Store paths for Wav2Lip components
        self.wav2lip_dir = os.path.abspath(os.path.join(models_dir, "Wav2Lip"))
        self.model_path = None
        
        self._setup_wav2lip()
    
    def _download_file(self, url: str, dest_path: str, desc: str):
        """Download a file with retries and timeout."""
        import requests
        import time
        
        max_retries = 3
        timeout = 30
        
        for attempt in range(max_retries):
            try:
                logger.info(f"Downloading {desc} (Attempt {attempt + 1}/{max_retries})...")
                response = requests.get(url, stream=True, timeout=timeout)
                response.raise_for_status()
                
                total_size = int(response.headers.get('content-length', 0))
                block_size = 1024 * 64 # 64KB
                
                with open(dest_path, 'wb') as f:
                    downloaded = 0
                    for data in response.iter_content(block_size):
                        downloaded += len(data)
                        f.write(data)
                        if total_size > 0 and downloaded % (block_size * 20) == 0:
                            done = int(50 * downloaded / total_size)
                            sys.stdout.write(f"\r[{'=' * done}{' ' * (50-done)}] {downloaded/1024/1024:.1f}/{total_size/1024/1024:.1f} MB")
                            sys.stdout.flush()
                
                print("") # New line after progress bar
                logger.info(f"Download complete: {dest_path}")
                return
            except Exception as e:
                logger.warning(f"Download failed: {e}")
                if attempt < max_retries - 1:
                    time.sleep(5)
                else:
                    raise RuntimeError(f"Failed to download {desc} after {max_retries} attempts: {e}")

    def _setup_wav2lip(self):
        """Download and setup Wav2Lip repository and models."""
        
        # Check if Wav2Lip is already cloned
        if not os.path.exists(self.wav2lip_dir):
            logger.info("Cloning Wav2Lip repository...")
            subprocess.run([
                "git", "clone", 
                "https://github.com/Rudrabha/Wav2Lip.git",
                self.wav2lip_dir
            ], check=True)
        
        # Download model if not present
        model_filename = f"{self.model_type}.pth"
        self.model_path = os.path.join(self.wav2lip_dir, "checkpoints", model_filename)
        
        if not os.path.exists(self.model_path) or os.path.getsize(self.model_path) < 1000000:
            os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
            model_url = self.WAV2LIP_MODELS[self.model_type]
            self._download_file(model_url, self.model_path, self.model_type)
        
        # Download face detection model
        face_det_path = os.path.join(self.wav2lip_dir, "face_detection", "detection", "sfd", "s3fd.pth")
        if not os.path.exists(face_det_path) or os.path.getsize(face_det_path) < 1000000:
            os.makedirs(os.path.dirname(face_det_path), exist_ok=True)
            # Using mirror for face detection as well
            face_det_url = "https://github.com/justinjohn0306/Wav2Lip/releases/download/models/s3fd.pth"
            self._download_file(face_det_url, face_det_path, "face detection model")
    
    def sync_lips(
        self,
        video_path: str,
        audio_path: str,
        output_path: str,
        face_padding: Tuple[int, int, int, int] = (0, 10, 0, 0),
        resize_factor: int = 1
    ) -> str:
        """
        Generate lip-synced video.
        
        Args:
            video_path: Path to input video
            audio_path: Path to audio file for lip sync
            output_path: Path for output video
            face_padding: Padding around detected face (top, bottom, left, right)
            resize_factor: Resize factor for video (1 = original size)
            
        Returns:
            Path to the output video
        """
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video file not found: {video_path}")
        if not os.path.exists(audio_path):
            raise FileNotFoundError(f"Audio file not found: {audio_path}")
        
        logger.info(f"Starting lip sync: {video_path} + {audio_path}")
        
        # Build Wav2Lip inference command
        # Use relative path for script since we set cwd below
        inference_script = "inference.py"
        
        # Create output directory if needed
        os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)
        
        # Delete existing output to prevent stale results
        if os.path.exists(output_path):
            try:
                os.remove(output_path)
            except OSError:
                pass
        
        # Convert all paths to absolute for the subprocess
        abs_model_path = os.path.abspath(self.model_path)
        abs_video_path = os.path.abspath(video_path)
        abs_audio_path = os.path.abspath(audio_path)
        abs_output_path = os.path.abspath(output_path)
        
        cmd = [
            sys.executable, inference_script,
            "--checkpoint_path", abs_model_path,
            "--face", abs_video_path,
            "--audio", abs_audio_path,
            "--outfile", abs_output_path,
            "--pads", str(face_padding[0]), str(face_padding[1]), 
                      str(face_padding[2]), str(face_padding[3]),
            "--resize_factor", str(resize_factor),
        ]
        
        if self.device == "cpu":
            cmd.append("--nosmooth")
        
        logger.info(f"Running Wav2Lip inference...")
        
        # Run inference
        result = subprocess.run(
            cmd,
            cwd=self.wav2lip_dir
        )
        
        if result.returncode != 0:
            logger.error(f"Wav2Lip inference failed with return code {result.returncode}")
            raise RuntimeError(f"Lip sync failed. Check terminal output for details.")
        
        if not os.path.exists(output_path):
            raise RuntimeError("Output video was not created")
        
        logger.info(f"Lip-synced video saved to: {output_path}")
        
        return output_path
    
    def sync_lips_simple(
        self,
        video_path: str,
        audio_path: str,
        output_path: str
    ) -> str:
        """
        Simple lip sync using ffmpeg for audio replacement (fallback method).
        This doesn't actually sync lips but replaces audio.
        
        Args:
            video_path: Path to input video
            audio_path: Path to audio file
            output_path: Path for output video
            
        Returns:
            Path to the output video
        """
        logger.info("Using simple audio replacement (fallback method)")
        
        cmd = [
            "ffmpeg", "-y",
            "-i", video_path,
            "-i", audio_path,
            "-c:v", "copy",
            "-map", "0:v:0",
            "-map", "1:a:0",
            "-shortest",
            output_path
        ]
        
        subprocess.run(cmd, check=True, capture_output=True)
        
        logger.info(f"Video with replaced audio saved to: {output_path}")
        
        return output_path


class MuseTalkLipSyncer:
    """
    Alternative lip syncer using MuseTalk for higher quality.
    (Placeholder for future implementation)
    """
    
    def __init__(self, device: Optional[str] = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        logger.warning("MuseTalk integration is not yet implemented. Using Wav2Lip instead.")


if __name__ == "__main__":
    # Test lip syncer initialization
    syncer = LipSyncer(model_type="wav2lip")
    print("LipSyncer initialized successfully!")
