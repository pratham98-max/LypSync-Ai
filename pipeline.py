"""
Main Pipeline Orchestrator for Lip Sync Voice Cloning
Coordinates all stages: Transcription → Translation → Voice Cloning → Lip Sync
"""

import os
import yaml
import json
import argparse
import logging
import tempfile
import shutil
import sys
import time
from typing import Dict, Any, Optional
from pathlib import Path
import torch
import torchaudio
import soundfile as sf

os.environ.setdefault("COQUI_TOS_AGREED", "1")
if os.path.exists("D:\\ai_cache"):
    os.environ.setdefault("TTS_HOME", r"D:\ai_cache\tts")
    os.environ.setdefault("HF_HOME", r"D:\ai_cache\huggingface")
    os.environ.setdefault("TORCH_HOME", r"D:\ai_cache\torch")

# ============================================================
# WINDOWS DLL FIX: Add FFmpeg to DLL search path
# Required for torchcodec and XTTS-v2 on Windows
# ============================================================
def setup_dll_paths():
    """Add FFmpeg bin directory to DLL search path for Windows."""
    if os.name != 'nt':
        return
    
    # Try to find ffmpeg in PATH
    ffmpeg_exe = shutil.which("ffmpeg")
    if ffmpeg_exe:
        ffmpeg_bin = os.path.dirname(ffmpeg_exe)
        try:
            os.add_dll_directory(ffmpeg_bin)
            print(f"  [DEBUG] Added DLL directory: {ffmpeg_bin}")
        except Exception as e:
            print(f"  [DEBUG] Failed to add DLL directory: {e}")
    else:
        # Fallback: check winget default path
        winget_path = Path(os.environ.get("LOCALAPPDATA", "")) / "Microsoft" / "WinGet" / "Packages"
        if winget_path.exists():
            for p in winget_path.glob("**/ffmpeg*/bin"):
                try:
                    os.add_dll_directory(str(p))
                    print(f"  [DEBUG] Added Winget DLL directory: {p}")
                    break
                except Exception:
                    continue

setup_dll_paths()

# ============================================================
# MONKEY PATCH: Fix torchaudio.load for Windows
# torchaudio 2.11+ can have issues with missing torchcodec DLLs.
# We patch it to use soundfile instead.
# ============================================================
original_torchaudio_load = torchaudio.load

def patched_torchaudio_load(filepath, **kwargs):
    """Load audio using soundfile as a fallback for torchaudio."""
    try:
        # Try original first
        return original_torchaudio_load(filepath, **kwargs)
    except Exception as e:
        # Fallback to soundfile
        try:
            data, samplerate = sf.read(filepath)
            if len(data.shape) == 1:
                tensor = torch.FloatTensor(data).unsqueeze(0)
            else:
                tensor = torch.FloatTensor(data.T)
            return tensor, samplerate
        except Exception as inner_e:
            raise e # Raise original error if fallback fails

torchaudio.load = patched_torchaudio_load
# ============================================================

# ============================================================
# MONKEY PATCH: Fix num2words for Hindi (hi)
# XTTS-v2 tokenizer calls num2words with lang='hi' which is not supported.
# We patch it to return the number as a string.
# ============================================================
try:
    import num2words
    original_num2words = num2words.num2words
    
    def patched_num2words(number, lang=None, **kwargs):
        try:
            return original_num2words(number, lang=lang, **kwargs)
        except (NotImplementedError, KeyError):
            # Fallback for unsupported languages like Hindi
            return str(number)
            
    num2words.num2words = patched_num2words
except ImportError:
    pass
# ============================================================

# Rich console for beautiful output
try:
    from rich.console import Console
    from rich.panel import Panel
    from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn
    from rich.table import Table
    from rich import print as rprint
    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False

# Setup console
console = Console() if RICH_AVAILABLE else None

def print_header(text: str):
    """Print a styled header."""
    if RICH_AVAILABLE:
        console.print(Panel(text, style="bold cyan", expand=False))
    else:
        print("\n" + "=" * 60)
        print(f"  {text}")
        print("=" * 60)

def print_step(step_num: int, total: int, title: str):
    """Print a step indicator."""
    if RICH_AVAILABLE:
        console.print(f"\n[bold green]>> STEP {step_num}/{total}:[/bold green] [bold white]{title}[/bold white]")
    else:
        print(f"\n>> STEP {step_num}/{total}: {title}")

def print_info(message: str):
    """Print an info message."""
    if RICH_AVAILABLE:
        console.print(f"  [cyan][i][/cyan] {message}")
    else:
        print(f"  [i] {message}")

def print_success(message: str):
    """Print a success message."""
    if RICH_AVAILABLE:
        console.print(f"  [green][OK][/green] {message}")
    else:
        print(f"  [OK] {message}")

def print_warning(message: str):
    """Print a warning message."""
    if RICH_AVAILABLE:
        console.print(f"  [yellow][WARN][/yellow] {message}")
    else:
        print(f"  [WARN] {message}")

def print_error(message: str):
    """Print an error message."""
    if RICH_AVAILABLE:
        console.print(f"  [red][ERR][/red] {message}")
    else:
        print(f"  [ERR] {message}")

def print_progress(message: str):
    """Print a progress message."""
    if RICH_AVAILABLE:
        console.print(f"  [dim]->[/dim] {message}")
    else:
        print(f"  -> {message}")

# Configure logging to be less verbose
logging.basicConfig(
    level=logging.WARNING,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class LipSyncVoiceClonePipeline:
    """
    End-to-end pipeline for:
    1. Extracting speech from video
    2. Translating to target Indic language
    3. Generating cloned voice in target language
    4. Creating lip-synced video output
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the pipeline with configuration.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.temp_dir = config.get("output", {}).get("temp_dir", "temp")
        os.makedirs(self.temp_dir, exist_ok=True)
        
        # Get device settings
        gpu_device = config.get("advanced", {}).get("gpu_device", 0)
        self.device = f"cuda:{gpu_device}" if gpu_device >= 0 and torch.cuda.is_available() else "cpu"
        
        gpu_info = ""
        if self.device.startswith("cuda"):
            gpu_name = torch.cuda.get_device_name(gpu_device)
            gpu_mem = round(torch.cuda.get_device_properties(gpu_device).total_memory / 1024**3, 1)
            gpu_info = f" [bold green]GPU ACTIVE:[/bold green] {gpu_name} ({gpu_mem}GB)"
        
        print_info(f"Pipeline initialized on device: {self.device}{gpu_info}")
        
        # Initialize components (lazy loading)
        self._transcriber = None
        self._translator = None
        self._voice_cloner = None
        self._lip_syncer = None
    
    def _save_checkpoint(self, data: Any, filename: str):
        """Save a checkpoint to the temp directory."""
        path = os.path.join(self.temp_dir, filename)
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        print_info(f"Checkpoint saved: {path}")

    def _load_checkpoint(self, filename: str) -> Optional[Any]:
        """Load a checkpoint from the temp directory if it exists."""
        path = os.path.join(self.temp_dir, filename)
        if os.path.exists(path):
            try:
                with open(path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                print_success(f"Resuming from checkpoint: {path}")
                return data
            except Exception as e:
                print_warning(f"Failed to load checkpoint {path}: {e}")
        return None
    
    def _generate_silence(self, duration: float, output_path: str):
        """Generate a silent wav file of given duration."""
        import subprocess
        sample_rate = self.config.get("advanced", {}).get("sample_rate", 22050)
        cmd = [
            "ffmpeg", "-y",
            "-f", "lavfi", "-i", f"anullsrc=r={sample_rate}:cl=mono",
            "-t", f"{duration:.3f}",
            output_path
        ]
        subprocess.run(cmd, check=True, capture_output=True)
    
    @property
    def transcriber(self):
        """Lazy load transcriber."""
        if self._transcriber is None:
            print_progress("Loading Whisper model for speech recognition...")
            from modules.transcriber import Transcriber
            whisper_model = self.config.get("settings", {}).get("whisper_model", "large-v3")
            self._transcriber = Transcriber(model_name=whisper_model, device=self.device)
            print_success(f"Whisper model ({whisper_model}) loaded!")
        return self._transcriber
    
    @property
    def translator(self):
        """Lazy load translator."""
        if self._translator is None:
            print_progress("Loading M2M100 model for translation...")
            from modules.translator import IndicTranslator
            self._translator = IndicTranslator(device=self.device)
            print_success("M2M100 translation model loaded!")
        return self._translator
    
    @property
    def voice_cloner(self):
        """Lazy load voice cloner."""
        if self._voice_cloner is None:
            print_progress("Loading XTTS-v2 model for voice cloning...")
            from modules.voice_cloner import VoiceCloner
            self._voice_cloner = VoiceCloner(device=self.device)
            print_success("XTTS-v2 voice cloning model loaded!")
        return self._voice_cloner
    
    @property
    def lip_syncer(self):
        """Lazy load lip syncer."""
        if self._lip_syncer is None:
            print_progress("Setting up Wav2Lip for lip synchronization...")
            from modules.lip_sync import LipSyncer
            lip_model = self.config.get("settings", {}).get("lip_sync_model", "wav2lip")
            self._lip_syncer = LipSyncer(model_type=lip_model, device=self.device)
            print_success(f"Wav2Lip ({lip_model}) initialized!")
        return self._lip_syncer
    
    def run(
        self,
        video_path: Optional[str] = None,
        voice_sample_path: Optional[str] = None,
        target_language: Optional[str] = None,
        output_path: Optional[str] = None
    ) -> str:
        """
        Run the full pipeline.
        """
        # Get paths from config or overrides
        video_path = video_path or self.config["input"]["video_path"]
        voice_sample_path = voice_sample_path or self.config["input"]["voice_sample_path"]
        target_language = target_language or self.config["settings"]["target_language"]
        output_path = output_path or self.config["output"]["output_path"]
        
        # Print pipeline header
        print_header("LIP SYNC VOICE CLONE PIPELINE")
        
        # Print configuration
        if RICH_AVAILABLE:
            table = Table(show_header=False, box=None)
            table.add_column("Key", style="cyan")
            table.add_column("Value", style="white")
            table.add_row("Input Video", video_path)
            table.add_row("Voice Sample", voice_sample_path)
            table.add_row("Target Language", target_language)
            table.add_row("Output", output_path)
            console.print(table)
        else:
            print(f"  Input Video: {video_path}")
            print(f"  Voice Sample: {voice_sample_path}")
            print(f"  Target Language: {target_language}")
            print(f"  Output: {output_path}")
        
        # Create output directory
        os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)
        
        start_time = time.time()
        
        # ============================================================
        # STEP 1: Extract audio and transcribe
        # ============================================================
        print_step(1, 4, "Speech Recognition (Whisper)")
        
        segments = self._load_checkpoint("transcription.json")
        
        if segments is None:
            print_progress("Extracting audio from video...")
            from modules.transcriber import extract_audio_from_video
            audio_path = os.path.join(self.temp_dir, "extracted_audio.wav")
            extract_audio_from_video(video_path, audio_path)
            print_success(f"Audio extracted: {audio_path}")
            
            print_progress("Transcribing speech to text...")
            transcription = self.transcriber.transcribe(audio_path)
            segments = self.transcriber.get_text_with_timestamps(transcription)
            
            # Save checkpoint
            self._save_checkpoint(segments, "transcription.json")
            
            print_success(f"Transcription complete!")
            print_info(f"  Found {len(segments)} speech segments")
        else:
            print_success("Transcription loaded from cache.")
            print_info(f"  Found {len(segments)} speech segments")
        
        # Show sample of transcription
        if segments:
            sample_text = segments[0]['text'][:150] + "..." if len(segments[0]['text']) > 150 else segments[0]['text']
            if RICH_AVAILABLE:
                console.print(f"  [dim]Preview: \"{sample_text}\"[/dim]")
            else:
                print(f"  Preview: \"{sample_text}\"")
        
        # ============================================================
        # STEP 2: Translate to target language
        # ============================================================
        print_step(2, 4, f"Translation (English → {target_language.upper()})")
        
        translated_segments = self._load_checkpoint("translated_segments.json")
        
        if translated_segments is None:
            print_progress(f"Translating {len(segments)} segments to {target_language}...")
            translated_segments = self.translator.translate_segments(segments, target_language)
            
            # Save checkpoint
            self._save_checkpoint(translated_segments, "translated_segments.json")
            print_success("Translation complete!")
        else:
            print_success("Translation loaded from cache.")
        
        # Show sample translation
        if segments and translated_segments:
            original = segments[0]['text'][:60] + "..." if len(segments[0]['text']) > 60 else segments[0]['text']
            translated = translated_segments[0]['text'][:60] + "..." if len(translated_segments[0]['text']) > 60 else translated_segments[0]['text']
            print_info(f"  Original: \"{original}\"")
            print_info(f"  Translated: \"{translated}\"")
        
        # ============================================================
        # STEP 3: Generate cloned voice audio
        # ============================================================
        generated_audio_path = os.path.join(self.temp_dir, "generated_audio.wav")
        
        if os.path.exists(generated_audio_path):
            print_success(f"Generated audio found in cache: {generated_audio_path}")
        else:
            audio_segments_dir = os.path.join(self.temp_dir, "audio_segments")
            os.makedirs(audio_segments_dir, exist_ok=True)
            
            # Try to load existing processed segments
            processed_segments = self._load_checkpoint("processed_segments.json")
            
            if processed_segments:
                print_info(f"  Found {len(processed_segments)} segments in checkpoint.")
                # Verify audio files exist for these segments
                for i, seg in enumerate(processed_segments):
                    if seg.get("audio_path") and not os.path.exists(seg["audio_path"]):
                        print_warning(f"  Missing audio for segment {i}, will re-synthesize.")
                        processed_segments = None
                        break
            
            if processed_segments is None:
                print_progress("Synthesizing cloned voice for individual segments...")
                print_info(f"  Voice sample: {voice_sample_path}")
                
                processed_segments = self.voice_cloner.synthesize_segments(
                    segments=translated_segments,
                    speaker_wav=voice_sample_path,
                    language=target_language,
                    output_dir=audio_segments_dir
                )
                
                # Save checkpoint
                self._save_checkpoint(processed_segments, "processed_segments.json")
            else:
                print_success("Resuming voice cloning from segment checkpoint.")
            
            # Concatenate and match durations
            print_progress("Matching audio durations and concatenating...")
            import subprocess
            
            # Create a file list for ffmpeg concatenation
            concat_list_path = os.path.join(self.temp_dir, "concat_list.txt")
            current_time = 0.0
            with open(concat_list_path, "w", encoding='utf-8') as f:
                for i, seg in enumerate(processed_segments):
                    # Insert silence if there's a gap before this segment
                    gap_dur = seg["start"] - current_time
                    if gap_dur > 0.01:
                        silence_path = os.path.join(audio_segments_dir, f"gap_{i:04d}.wav")
                        self._generate_silence(gap_dur, silence_path)
                        abs_silence_path = os.path.abspath(silence_path).replace("\\", "/")
                        f.write(f"file '{abs_silence_path}'\n")
                    
                    if seg.get("audio_path"):
                        target_dur = seg["end"] - seg["start"]
                        if target_dur <= 0:
                            print_warning(f"  Skipping segment {i} due to zero duration.")
                            # Still update current_time to avoid accumulating gap
                            current_time = max(current_time, seg["end"])
                            continue
                            
                        adjusted_path = os.path.join(audio_segments_dir, f"adj_{i:04d}.wav")
                        
                        # Adjust speed to match segment duration, but with a natural cap
                        actual_seg_dur = self.voice_cloner.adjust_audio_speed(
                            seg["audio_path"], target_dur, adjusted_path, max_speed=1.15
                        )
                        
                        abs_path = os.path.abspath(adjusted_path).replace("\\", "/")
                        f.write(f"file '{abs_path}'\n")
                        current_time += actual_seg_dur
            
            # Concatenate with ffmpeg
            try:
                cmd = [
                    "ffmpeg", "-y",
                    "-f", "concat",
                    "-safe", "0",
                    "-i", concat_list_path,
                    "-c:a", "pcm_s16le",
                    generated_audio_path
                ]
                subprocess.run(cmd, check=True, capture_output=True)
                print_success(f"Voice cloning and concatenation complete!")
            except Exception as e:
                print_error(f"Failed to concatenate audio: {e}")
                # Fallback to simple concatenation if speed adjustment failed
                raise e
        
        print_info(f"  Audio path: {generated_audio_path}")
        
        # ============================================================
        # STEP 4: Generate lip-synced video
        # ============================================================
        print_step(4, 4, "Lip Synchronization (Wav2Lip)")
        
        print_progress("Detecting faces in video...")
        print_progress("Generating lip movements to match audio...")
        
        resize_factor = self.config.get("advanced", {}).get("resize_factor", 1)
        try:
            self.lip_syncer.sync_lips(
                video_path=video_path,
                audio_path=generated_audio_path,
                output_path=output_path,
                resize_factor=resize_factor
            )
            print_success("Lip sync complete!")
        except Exception as e:
            print_warning(f"Wav2Lip failed: {e}")
            print_progress("Using audio replacement fallback...")
            self.lip_syncer.sync_lips_simple(
                video_path=video_path,
                audio_path=generated_audio_path,
                output_path=output_path
            )
            print_success("Audio replacement complete!")
        
        # ============================================================
        # DONE!
        # ============================================================
        elapsed_time = time.time() - start_time
        
        print("")
        if RICH_AVAILABLE:
            console.print(Panel(
                f"[bold green]PIPELINE COMPLETED SUCCESSFULLY![/bold green]\n\n"
                f"[white]Output saved to:[/white] [cyan]{output_path}[/cyan]\n"
                f"[white]Total time:[/white] [yellow]{elapsed_time:.1f} seconds[/yellow]",
                style="green",
                expand=False
            ))
        else:
            print("\n" + "=" * 60)
            print("  PIPELINE COMPLETED SUCCESSFULLY!")
            print(f"  Output saved to: {output_path}")
            print(f"  Total time: {elapsed_time:.1f} seconds")
            print("=" * 60)
        
        return output_path
    
    def cleanup(self):
        """Clean up temporary files."""
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
            print_info(f"Cleaned up temporary directory: {self.temp_dir}")


def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from YAML or JSON file."""
    print_progress(f"Loading configuration from {config_path}...")
    with open(config_path, 'r', encoding='utf-8') as f:
        if config_path.endswith('.json'):
            config = json.load(f)
        else:
            config = yaml.safe_load(f)
    print_success("Configuration loaded!")
    return config


def list_languages():
    """List all supported languages."""
    print_header("SUPPORTED INDIC LANGUAGES")
    
    # Import here to avoid loading models just to list languages
    from modules.translator import IndicTranslator
    
    if RICH_AVAILABLE:
        table = Table(show_header=True, header_style="bold magenta")
        table.add_column("Code", style="cyan", width=6)
        table.add_column("Language", style="white")
        
        for code, name in IndicTranslator.LANGUAGE_NAMES.items():
            table.add_row(code, name)
        
        console.print(table)
    else:
        print("\n  Code  |  Language")
        print("  " + "-" * 25)
        for code, name in IndicTranslator.LANGUAGE_NAMES.items():
            print(f"  {code:5} |  {name}")
    
    print("")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="🎬 Lip Sync Voice Clone Pipeline for Indic Languages"
    )
    parser.add_argument(
        "--config", "-c",
        type=str,
        default="config.yaml",
        help="Path to configuration file (YAML or JSON)"
    )
    parser.add_argument(
        "--video", "-v",
        type=str,
        help="Override input video path"
    )
    parser.add_argument(
        "--voice", "-s",
        type=str,
        help="Override voice sample path"
    )
    parser.add_argument(
        "--language", "-l",
        type=str,
        help="Override target language"
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        help="Override output path"
    )
    parser.add_argument(
        "--list-languages",
        action="store_true",
        help="List supported languages and exit"
    )
    
    args = parser.parse_args()
    
    # ASCII art banner (Windows-compatible)
    print("""
+===============================================================+
|     ** LIP SYNC VOICE CLONE - INDIC LANGUAGES EDITION **     |
|                                                               |
|  Whisper (STT) -> IndicTrans2 -> XTTS-v2 -> Wav2Lip          |
+===============================================================+
    """)
    
    if args.list_languages:
        list_languages()
        return
    
    try:
        # Load configuration
        config = load_config(args.config)
        
        # Create and run pipeline
        pipeline = LipSyncVoiceClonePipeline(config)
        
        output_path = pipeline.run(
            video_path=args.video,
            voice_sample_path=args.voice,
            target_language=args.language,
            output_path=args.output
        )
        
    except FileNotFoundError as e:
        print_error(f"File not found: {e}")
        sys.exit(1)
    except Exception as e:
        print_error(f"Pipeline failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
