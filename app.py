"""
LipSync Web UI: AI Voice Cloning & Multilingual Lip Synchronization
Powered by Whisper, M2M100, XTTS-v2, and Wav2Lip.
"""

import os
import sys
import tempfile
import time
import subprocess
from pathlib import Path
import gradio as gr
import torch

# Ensure UTF-8 console output and environment paths
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

# Add project root to sys.path
BASE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE_DIR))

# Supported Languages mapping
LANGUAGE_OPTIONS = {
    "Hindi (हिंदी)": "hi",
    "Tamil (தமிழ்)": "ta",
    "Telugu (తెలుగు)": "te",
    "Bengali (বাংলা)": "bn",
    "Marathi (मराठी)": "mr",
    "Gujarati (ગુજરાતી)": "gu",
    "Kannada (ಕನ್ನಡ)": "kn",
    "Malayalam (മലയാളം)": "ml",
    "Punjabi (ਪੰਜਾਬੀ)": "pa",
    "Urdu (اردو)": "ur",
}

# Global pipeline instance (lazy loaded)
_PIPELINE = None

def get_pipeline(whisper_model="base", resize_factor=2):
    global _PIPELINE
    from pipeline import LipSyncVoiceClonePipeline
    config = {
        "input": {
            "video_path": "samples/sample1.mp4",
            "voice_sample_path": "samples/voice_sample.wav"
        },
        "output": {
            "output_path": "output/result.mp4",
            "temp_dir": "temp"
        },
        "settings": {
            "target_language": "hi",
            "whisper_model": whisper_model,
            "voice_model": "xtts-v2",
            "lip_sync_model": "wav2lip"
        },
        "advanced": {
            "gpu_device": -1,
            "resize_factor": resize_factor,
            "sample_rate": 22050,
            "preserve_quality": True,
            "chunk_duration": 30
        }
    }
    if _PIPELINE is None:
        _PIPELINE = LipSyncVoiceClonePipeline(config)
    else:
        _PIPELINE.config = config
    return _PIPELINE


def process_lipsync(
    video_file,
    voice_file,
    auto_extract_voice,
    target_language_label,
    whisper_model,
    resize_factor,
    progress=gr.Progress(track_tqdm=True)
):
    """Run the complete 4-step pipeline from Gradio UI."""
    if video_file is None:
        return None, None, "❌ Please upload or select a source video."

    target_lang = LANGUAGE_OPTIONS.get(target_language_label, "hi")
    timestamp = int(time.time())
    output_dir = os.path.join(BASE_DIR, "output")
    os.makedirs(output_dir, exist_ok=True)
    final_output_path = os.path.join(output_dir, f"result_{target_lang}_{timestamp}.mp4")
    temp_dir = os.path.join(BASE_DIR, "temp", f"session_{timestamp}")
    os.makedirs(temp_dir, exist_ok=True)

    logs = []
    def log(msg):
        logs.append(f"[{time.strftime('%H:%M:%S')}] {msg}")
        return "\n".join(logs)

    try:
        # Determine voice sample path
        voice_path = None
        if voice_file is not None:
            voice_path = voice_file
            log("Using uploaded speaker voice reference.")
        elif auto_extract_voice:
            log("Auto-extracting 6-second speaker audio from source video...")
            voice_path = os.path.join(temp_dir, "auto_voice.wav")
            extract_cmd = [
                "ffmpeg", "-y", "-i", video_file,
                "-vn", "-ac", "1", "-ar", "22050",
                "-ss", "0", "-t", "6", voice_path
            ]
            subprocess.run(extract_cmd, check=True, capture_output=True)
            log("Speaker voice reference extracted.")
        else:
            default_sample = os.path.join(BASE_DIR, "samples", "voice_sample.wav")
            if os.path.exists(default_sample):
                voice_path = default_sample
                log("Using default sample reference voice.")
            else:
                return None, None, "❌ No voice sample provided and auto-extract was unchecked."

        # Initialize pipeline
        progress(0.05, desc="Initializing ML models...")
        pipeline = get_pipeline(whisper_model=whisper_model, resize_factor=int(resize_factor))
        pipeline.temp_dir = temp_dir

        # ============================================================
        # STEP 1: Transcription
        # ============================================================
        progress(0.15, desc="Step 1/4: Transcribing speech with Whisper...")
        log("▶ STEP 1/4: Speech Recognition (Whisper)")
        from modules.transcriber import extract_audio_from_video
        extracted_audio = os.path.join(temp_dir, "extracted_audio.wav")
        extract_audio_from_video(video_file, extracted_audio)

        transcription = pipeline.transcriber.transcribe(extracted_audio)
        segments = pipeline.transcriber.get_text_with_timestamps(transcription)
        log(f"Transcription complete: Found {len(segments)} segments.")
        if segments:
            log(f"Sample transcript: \"{segments[0]['text'][:120]}\"")

        # ============================================================
        # STEP 2: Translation
        # ============================================================
        progress(0.35, desc=f"Step 2/4: Translating to {target_language_label}...")
        log(f"▶ STEP 2/4: Neural Translation (English → {target_lang.upper()})")
        translated_segments = pipeline.translator.translate_segments(segments, target_lang)
        log("Translation complete.")
        if translated_segments:
            log(f"Sample translation: \"{translated_segments[0]['text'][:120]}\"")

        # ============================================================
        # STEP 3: Voice Cloning
        # ============================================================
        progress(0.55, desc="Step 3/4: Synthesizing cloned speech with XTTS-v2...")
        log(f"▶ STEP 3/4: Voice Cloning (XTTS-v2) in {target_lang}")
        audio_segments_dir = os.path.join(temp_dir, "audio_segments")
        os.makedirs(audio_segments_dir, exist_ok=True)

        processed_segments = pipeline.voice_cloner.synthesize_segments(
            segments=translated_segments,
            speaker_wav=voice_path,
            language=target_lang,
            output_dir=audio_segments_dir
        )

        # Match audio durations and concatenate
        generated_audio_path = os.path.join(temp_dir, "generated_audio.wav")
        concat_list_path = os.path.join(temp_dir, "concat_list.txt")
        current_time = 0.0

        with open(concat_list_path, "w", encoding='utf-8') as f:
            for i, seg in enumerate(processed_segments):
                gap_dur = seg["start"] - current_time
                if gap_dur > 0.01:
                    silence_path = os.path.join(audio_segments_dir, f"gap_{i:04d}.wav")
                    pipeline._generate_silence(gap_dur, silence_path)
                    abs_silence_path = os.path.abspath(silence_path).replace("\\", "/")
                    f.write(f"file '{abs_silence_path}'\n")

                if seg.get("audio_path"):
                    target_dur = seg["end"] - seg["start"]
                    if target_dur <= 0:
                        current_time = max(current_time, seg["end"])
                        continue
                    adjusted_path = os.path.join(audio_segments_dir, f"adj_{i:04d}.wav")
                    actual_seg_dur = pipeline.voice_cloner.adjust_audio_speed(
                        seg["audio_path"], target_dur, adjusted_path, max_speed=1.15
                    )
                    abs_path = os.path.abspath(adjusted_path).replace("\\", "/")
                    f.write(f"file '{abs_path}'\n")
                    current_time += actual_seg_dur

        concat_cmd = [
            "ffmpeg", "-y", "-f", "concat", "-safe", "0",
            "-i", concat_list_path, "-c:a", "pcm_s16le",
            generated_audio_path
        ]
        subprocess.run(concat_cmd, check=True, capture_output=True)
        log("Voice synthesis & audio concatenation complete.")

        # ============================================================
        # STEP 4: Lip Synchronization
        # ============================================================
        progress(0.75, desc="Step 4/4: Generating lip movements with Wav2Lip...")
        log("▶ STEP 4/4: Lip Synchronization (Wav2Lip)")
        log("Detecting faces and synchronizing mouth movements to speech...")

        pipeline.lip_syncer.sync_lips(
            video_path=video_file,
            audio_path=generated_audio_path,
            output_path=final_output_path,
            resize_factor=int(resize_factor)
        )

        progress(1.0, desc="Completed!")
        log("🎉 Pipeline completed successfully!")
        log(f"Exported to: {final_output_path}")

        return final_output_path, generated_audio_path, "\n".join(logs)

    except Exception as e:
        import traceback
        err_trace = traceback.format_exc()
        all_logs = "\n".join(logs)
        return None, None, f"{all_logs}\n\n{err_trace}"


def clone_voice_only(text, voice_file, language_label, progress=gr.Progress()):
    """Quick audio-only voice cloning playground."""
    if not text or not text.strip():
        return None, "Please enter some text to synthesize."
    if voice_file is None:
        sample_path = os.path.join(BASE_DIR, "samples", "voice_sample.wav")
        if os.path.exists(sample_path):
            voice_file = sample_path
        else:
            return None, "Please provide a speaker voice sample."

    lang_code = LANGUAGE_OPTIONS.get(language_label, "hi")
    output_dir = os.path.join(BASE_DIR, "output")
    os.makedirs(output_dir, exist_ok=True)
    out_audio = os.path.join(output_dir, f"tts_{lang_code}_{int(time.time())}.wav")

    progress(0.3, desc="Loading XTTS-v2...")
    pipeline = get_pipeline()
    progress(0.6, desc="Synthesizing audio...")
    pipeline.voice_cloner.clone_voice(
        text=text.strip(),
        speaker_wav=voice_file,
        language=lang_code,
        output_path=out_audio
    )
    progress(1.0, desc="Done!")
    return out_audio, f"Successfully synthesized in {language_label}."


def create_ui():
    custom_css = """
    .gradio-container { max-width: 1200px !important; margin: 0 auto !important; }
    .header-box { text-align: center; padding: 20px 0 10px 0; }
    .header-box h1 { font-size: 2.2rem; font-weight: 800; margin-bottom: 8px; }
    .header-box p { color: #6b7280; font-size: 1rem; }
    .status-box { font-family: monospace; font-size: 0.85rem; }
    """

    with gr.Blocks(title="LipSync: Multilingual Voice Clone & Lip Sync", css=custom_css, theme=gr.themes.Soft()) as demo:
        with gr.Column(elem_classes=["header-box"]):
            gr.Markdown(
                """
                # 🎙️ LipSync: AI Voice Cloning & Lip Synchronization
                ### Convert English speech videos into Indic languages with cloned speaker voice and realistic lip sync
                """
            )

        with gr.Tabs():
            with gr.TabItem("🎬 Video LipSync Pipeline"):
                with gr.Row():
                    with gr.Column(scale=1):
                        gr.Markdown("### 1. Source Video & Voice")
                        input_video = gr.Video(
                            label="Source Video (English)",
                            sources=["upload"],
                            value=os.path.join(BASE_DIR, "samples", "sample1.mp4") if os.path.exists(os.path.join(BASE_DIR, "samples", "sample1.mp4")) else None
                        )
                        auto_voice_chk = gr.Checkbox(
                            label="Auto-extract voice reference from video (Recommended)",
                            value=True,
                            info="Extracts first 6 seconds of speech as speaker reference"
                        )
                        input_voice = gr.Audio(
                            label="Custom Speaker Reference (Optional)",
                            type="filepath",
                            sources=["upload", "microphone"],
                            visible=True
                        )

                        gr.Markdown("### 2. Language & Pipeline Settings")
                        lang_dropdown = gr.Dropdown(
                            label="Target Indic Language",
                            choices=list(LANGUAGE_OPTIONS.keys()),
                            value="Hindi (हिंदी)"
                        )

                        with gr.Accordion("⚙️ Advanced Model Settings", open=False):
                            whisper_opt = gr.Dropdown(
                                label="Whisper Model Size",
                                choices=["tiny", "base", "small", "medium", "large-v3"],
                                value="base",
                                info="Base is fast on CPU. Larger models offer higher precision."
                            )
                            resize_opt = gr.Slider(
                                label="Wav2Lip Resize Factor",
                                minimum=1, maximum=4, step=1, value=2,
                                info="2 is recommended for 720p+ videos on CPU."
                            )

                        run_btn = gr.Button("🚀 Start LipSync & Voice Cloning", variant="primary", size="lg")

                    with gr.Column(scale=1):
                        gr.Markdown("### 3. Output & Real-time Progress")
                        output_video = gr.Video(label="Generated Lip-Synced Video", interactive=False)
                        output_audio = gr.Audio(label="Synthesized Cloned Audio", interactive=False)
                        status_text = gr.Textbox(
                            label="Processing Logs",
                            lines=12,
                            elem_classes=["status-box"],
                            interactive=False,
                            placeholder="Status logs and intermediate transcripts will appear here..."
                        )

                run_btn.click(
                    fn=process_lipsync,
                    inputs=[input_video, input_voice, auto_voice_chk, lang_dropdown, whisper_opt, resize_opt],
                    outputs=[output_video, output_audio, status_text]
                )

            with gr.TabItem("🗣️ Voice Clone Playground (TTS Only)"):
                gr.Markdown("### Quick Voice Cloning Audio Playground\nSynthesize speech in any Indic language using the speaker's cloned voice.")
                with gr.Row():
                    with gr.Column():
                        tts_text = gr.Textbox(
                            label="Text to Speak",
                            placeholder="Enter text here (e.g. नमस्ते, आप कैसे हैं? / Hello, welcome to this video)",
                            lines=4,
                            value="नमस्ते, आप कैसे हैं? हम एआई वॉयस क्लोनिंग का परीक्षण कर रहे हैं।"
                        )
                        tts_lang = gr.Dropdown(
                            label="Language",
                            choices=list(LANGUAGE_OPTIONS.keys()),
                            value="Hindi (हिंदी)"
                        )
                        tts_ref_audio = gr.Audio(
                            label="Speaker Reference Audio",
                            type="filepath",
                            sources=["upload", "microphone"],
                            value=os.path.join(BASE_DIR, "samples", "voice_sample.wav") if os.path.exists(os.path.join(BASE_DIR, "samples", "voice_sample.wav")) else None
                        )
                        tts_btn = gr.Button("🔊 Clone Voice & Generate Speech", variant="primary")
                    with gr.Column():
                        tts_out_audio = gr.Audio(label="Generated Cloned Audio")
                        tts_status = gr.Textbox(label="Status", lines=2)

                tts_btn.click(
                    fn=clone_voice_only,
                    inputs=[tts_text, tts_ref_audio, tts_lang],
                    outputs=[tts_out_audio, tts_status]
                )

            with gr.TabItem("ℹ️ About & Hardware"):
                gr.Markdown(
                    """
                    ### 🏗️ Architecture
                    1. **Speech Recognition**: OpenAI Whisper (`base`)
                    2. **Neural Machine Translation**: Facebook M2M100 (`418M`)
                    3. **Zero-Shot Voice Cloning**: Coqui XTTS-v2 (`CPML`)
                    4. **Neural Lip Synchronization**: Wav2Lip + S3FD Face Detector

                    ### 💻 System Environment
                    - **Execution Mode**: CPU Optimized (AMD Graphics detected)
                    - **Model Weights Cache**: `D:\\ai_cache`
                    - **Supported Indic Languages**: Hindi, Tamil, Telugu, Bengali, Marathi, Gujarati, Kannada, Malayalam, Punjabi, Urdu
                    """
                )

    return demo


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 7860))
    app = create_ui()
    print(f"\n🚀 Launching LipSync Web UI on http://127.0.0.1:{port}\n")
    app.launch(server_name="127.0.0.1", server_port=port, share=False)
