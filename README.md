# 🎙️ LipSync: AI Voice Cloning & Multilingual Lip Synchronization

An end-to-end AI-powered video translation pipeline that converts English-speaking videos into multiple Indic languages while preserving the speaker's voice and synchronizing realistic lip movements. The system integrates speech recognition, neural machine translation, zero-shot voice cloning, and AI-based lip synchronization, supporting both high-quality batch processing and low-latency real-time streaming.

---

## 🚀 Features

- 🎙️ Automatic speech transcription using **OpenAI Whisper**
- 🌏 Multilingual translation with **IndicTrans2** (22+ Indic languages)
- 🗣️ Zero-shot voice cloning using **XTTS-v2**
- 🎭 AI-powered lip synchronization using **Wav2Lip**
- ⚡ Real-time streaming pipeline using asynchronous processing
- 🔄 Automatic fallback to **gTTS** for unsupported languages
- 🎥 End-to-end automated video translation pipeline
- 📦 Modular architecture for easy extension and deployment

---

# 🏗️ System Architecture

## Standard Batch Pipeline

Optimized for maximum output quality by processing the complete video before synthesis.

```text
                     Input Video
                          │
                          ▼
              FFmpeg Audio Extraction
                          │
                          ▼
            Whisper Speech Recognition
                          │
                          ▼
             IndicTrans2 Translation
                          │
                          ▼
              XTTS-v2 Voice Cloning
                          │
                          ▼
             Wav2Lip Synchronization
                          │
                          ▼
              Final Translated Video
```

### Processing Workflow

1. Extract audio from the source video using **FFmpeg**.
2. Transcribe speech into timestamped text using **OpenAI Whisper**.
3. Translate transcripts into the selected Indic language using **IndicTrans2**.
4. Generate cloned speech using **XTTS-v2**.
5. Synchronize generated speech with facial movements using **Wav2Lip**.
6. Export the translated lip-synced video.

---

## ⚡ Real-Time Streaming Pipeline

Designed for low-latency applications using asynchronous processing and streaming.

### Optimizations

- Parallel sentence-level execution using **asyncio**
- Streaming audio generation
- Lightweight inference modules
- Reduced end-to-end latency
- Incremental video synthesis

| Batch Pipeline | Streaming Pipeline |
|----------------|--------------------|
| Sequential execution | Async parallel execution |
| File-based workflow | Streaming generators |
| Maximum quality | Minimum latency |
| Offline processing | Interactive inference |

---

# 🛠️ Technology Stack

| Component | Technology |
|-----------|------------|
| Programming Language | Python |
| Speech Recognition | OpenAI Whisper |
| Translation | IndicTrans2 |
| Voice Cloning | XTTS-v2 |
| Lip Synchronization | Wav2Lip |
| Video Processing | FFmpeg |
| Fallback TTS | gTTS |
| Parallel Processing | asyncio |
| Deep Learning | PyTorch |

---

# 🌍 Supported Languages

| Language | Code |
|----------|------|
| Hindi | hi |
| Tamil | ta |
| Telugu | te |
| Malayalam | ml |
| Kannada | kn |
| Bengali | bn |
| Marathi | mr |
| Gujarati | gu |
| Punjabi | pa |

> **Note:** Additional Indic languages are supported through IndicTrans2.

---

# 🔄 Fault-Tolerant Voice Generation

XTTS-v2 does not natively support every Indic language. To prevent failures, the system automatically switches to **gTTS** whenever voice cloning is unavailable.

## Fallback Workflow

```text
              Target Language
                     │
                     ▼
          Supported by XTTS-v2?
               │           │
             Yes           No
              │             │
              ▼             ▼
         XTTS-v2         gTTS Fallback
              │             │
              └──────┬──────┘
                     ▼
          Continue Video Generation
```

This ensures uninterrupted execution even for unsupported languages while maintaining the complete translation workflow.

---

# 📂 Project Structure

```text
LipSync/
│
├── config.yaml
├── pipeline.py
├── pipeline_realtime.py
├── requirements.txt
│
├── models/
│
├── modules/
│   ├── transcriber.py
│   ├── translator.py
│   ├── voice_cloner.py
│   ├── voice_cloner_stream.py
│   └── lip_sync.py
│
├── samples/
├── output/
└── REALTIME_IMPLEMENTATION.md
```

---

# ⚙️ Installation

## Prerequisites

- Python 3.10+
- NVIDIA GPU (Recommended)
- FFmpeg installed
- CUDA-enabled PyTorch (Recommended)

## Clone Repository

```bash
git clone https://github.com/<your-username>/LipSync.git

cd LipSync
```

## Create Virtual Environment

### Windows

```bash
python -m venv venv

venv\Scripts\activate
```

### Linux / macOS

```bash
python3 -m venv venv

source venv/bin/activate
```

## Install Dependencies

```bash
pip install -r requirements.txt
```

---

# 🚀 Running the Project

## Configure

Update the target language in `config.yaml`.

```yaml
input:
  video_path: samples/input_video.mp4
  voice_sample_path: samples/voice.wav

settings:
  target_language: hi

output:
  output_path: output/result.mp4
```

## Web UI (Recommended)

Launch the interactive Gradio Web Studio to upload videos, select Indic languages, and preview results directly in your browser:

```bash
python app.py
```
Open your browser at: `http://127.0.0.1:7860`

## CLI Batch Pipeline

```bash
python run.py --config config.yaml
```

## Real-Time Pipeline

```bash
python pipeline_realtime.py
```

---

# 💻 Hardware Requirements

| Component | Recommendation |
|-----------|----------------|
| GPU | NVIDIA GPU (8GB+ VRAM) |
| RAM | 16GB+ |
| Python | 3.10+ |
| Storage | 10GB+ |
| FFmpeg | Installed |

---

# 📌 Tech Stack

```text
Python • PyTorch • OpenAI Whisper • IndicTrans2 • XTTS-v2
Wav2Lip • FFmpeg • asyncio • CUDA
```

---

# 🔮 Future Improvements

- Support for additional multilingual TTS models
- Live webcam translation
- Speaker diarization for multi-speaker videos
- Real-time web application deployment
- Voice emotion preservation
- Distributed GPU inference
- ONNX/TensorRT optimization for faster inference

---

## 📜 License

This project is intended for research and educational purposes.
