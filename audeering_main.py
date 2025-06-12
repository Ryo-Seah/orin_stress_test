import os
import numpy as np
import sounddevice as sd
import audeer
import audonnx

# Constants
SAMPLING_RATE = 16000
DURATION = 3  # seconds

# Ensure model is downloaded and extracted
model_url = "https://zenodo.org/record/6221127/files/w2v2-L-robust-12.6bc4a7fd-1.1.0.zip"
cache_dir = "cache/"
model_dir = "model/"
zip_filename = "w2v2-L-robust-12.6bc4a7fd-1.1.0.zip"

# Ensure model is downloaded and extracted
archive_path = audeer.download_url(model_url, cache_dir, filename=zip_filename)
extracted_path = audeer.extract_archive(archive_path, model_dir)
onnx_model = audonnx.load(extracted_path)

# Function to record audio
def record_audio(duration, rate):
    print("🎙️  Listening...")
    devices = sd.query_devices()
    for idx, device in enumerate(devices):
        if device['max_input_channels'] > 0:
            print(f"[{idx}] {device['name']}")
    audio = sd.rec(int(duration * rate), samplerate=rate, channels=1, dtype='float32')
    sd.wait()
    audio = np.squeeze(audio)
    print("Amplitude:", np.abs(audio).mean())
    return audio

# Predict stress from audio using arousal
def predict_stress(audio):
    output = onnx_model(audio, sampling_rate=SAMPLING_RATE)
    arousal, dominance, valence = output["logits"]
    return float(arousal), float(dominance), float(valence)

# Main loop
if __name__ == "__main__":
    while True:
        audio = record_audio(DURATION, SAMPLING_RATE)
        arousal, dominance, valence = predict_stress(audio)
        print(f"🧠 Arousal (Stress): {arousal:.2f} | Dominance: {dominance:.2f} | Valence: {valence:.2f}")
