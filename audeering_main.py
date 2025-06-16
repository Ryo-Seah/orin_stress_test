import os
import numpy as np
import sounddevice as sd
import audeer
import audonnx

# Constants
SAMPLING_RATE = 16000
DURATION = 3  # seconds

# Paths
archive_path = "cache/w2v2-L-robust-12.6bc4a7fd-1.1.0.zip"
model_dir = "model/"

# Create model dir if it doesn't exist
os.makedirs(model_dir, exist_ok=True)

# # Extract from local ZIP
# extracted_path = audeer.extract_archive(archive_path, model_dir)

# Load ONNX model from extracted path
onnx_model = audonnx.load(model_dir)

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
