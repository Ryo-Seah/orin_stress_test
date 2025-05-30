import sounddevice as sd
import numpy as np
import torch
from transformers import Wav2Vec2ForSequenceClassification, Wav2Vec2Processor
import librosa

# Constants
SAMPLING_RATE = 16000
DURATION = 3  # seconds
LABELS = ['angry', 'calm', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']

# Load model & processor
model_name = "Dpngtm/wav2vec2-emotion-recognition"
model = Wav2Vec2ForSequenceClassification.from_pretrained(model_name)
processor = Wav2Vec2Processor.from_pretrained(model_name)
model.eval()

STRESS_WEIGHTS = {
    "angry": 1.0,
    "fear": 1.0,
    "disgust": 0.8,
    "sad": 0.6,
    "neutral": 0.2,
    "surprise": 0.5,
    "happy": 0.0,
    "calm": 0.0
}

def record_audio(duration, rate):
    print("🎙️  Listening...")
    # print(sd.query_devices())
    devices = sd.query_devices()
    for idx, device in enumerate(devices):
        if device['max_input_channels'] > 0:
            print(f"[{idx}] {device['name']}")
            
    audio = sd.rec(int(duration * rate), samplerate=rate, channels=1, dtype='float32')
    print("Amplitude:", np.abs(audio).mean())
    sd.wait()
    return np.squeeze(audio)

def predict_emotion(audio):
    inputs = processor(audio, sampling_rate=SAMPLING_RATE, return_tensors="pt", padding=True)
    with torch.no_grad():
        logits = model(**inputs).logits
    probs = torch.softmax(logits, dim=-1).squeeze()
    stress_score = sum(STRESS_WEIGHTS[LABELS[i]] * probs[i].item() for i in range(len(LABELS)))
    
    pred_id = int(torch.argmax(logits, dim=-1))
    return LABELS[pred_id], probs[pred_id].item(), stress_score

# Main loop
if __name__ == "__main__":
    while True:
        audio = record_audio(DURATION, SAMPLING_RATE)
        label, confidence, stress = predict_emotion(audio)
        print(f"🧠 Emotion: {label} ({confidence*100:.1f}%) | 🔻 Stress Score: {stress:.2f}")

