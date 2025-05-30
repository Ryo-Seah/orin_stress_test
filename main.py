import sounddevice as sd
import numpy as np
import torch
from transformers import Wav2Vec2ForSequenceClassification, Wav2Vec2Processor
import librosa

# Constants
SAMPLING_RATE = 16000
DURATION = 2  # seconds
LABELS = ['angry', 'calm', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']

# Load model & processor
model = Wav2Vec2ForSequenceClassification.from_pretrained("ehcalabres/wav2vec2-lg-xlsr-en-speech-emotion-recognition")
processor = Wav2Vec2Processor.from_pretrained("ehcalabres/wav2vec2-lg-xlsr-en-speech-emotion-recognition")
model.eval()

def record_audio(duration, rate):
    print("🎙️  Listening...")
    audio = sd.rec(int(duration * rate), samplerate=rate, channels=1, dtype='float32')
    sd.wait()
    return np.squeeze(audio)

def predict_emotion(audio):
    inputs = processor(audio, sampling_rate=SAMPLING_RATE, return_tensors="pt", padding=True)
    with torch.no_grad():
        logits = model(**inputs).logits
    pred_id = int(torch.argmax(logits, dim=-1))
    score = torch.softmax(logits, dim=-1).squeeze()
    return LABELS[pred_id], score[pred_id].item()

# Main loop
if __name__ == "__main__":
    while True:
        audio = record_audio(DURATION, SAMPLING_RATE)
        label, confidence = predict_emotion(audio)
        print(f"🧠 Emotion: {label} ({confidence*100:.1f}%)")
