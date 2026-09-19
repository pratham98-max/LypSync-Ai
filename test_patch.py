import torchaudio
import soundfile as sf
import torch
import os

def patched_load(filepath, **kwargs):
    print(f"Loading via patch: {filepath}")
    data, samplerate = sf.read(filepath)
    if len(data.shape) == 1:
        tensor = torch.FloatTensor(data).unsqueeze(0)
    else:
        tensor = torch.FloatTensor(data.T)
    return tensor, samplerate

# Apply patch
torchaudio.load = patched_load

# Test
try:
    audio, sr = torchaudio.load('samples/voice_sample.wav')
    print(f"Success! Shape: {audio.shape}, SR: {sr}")
except Exception as e:
    print(f"Error: {e}")
