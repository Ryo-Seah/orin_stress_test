"""
Audio VAD (Valence, Arousal, Dominance) Detection Module
Handles audio recording and emotion recognition for stress detection system.
"""

import numpy as np
import sounddevice as sd
import audonnx
from collections import deque
from typing import Optional, Dict
import threading
import queue
import time


class AudioVADDetector:
    """Audio-based VAD detection using emotion recognition."""
    
    def __init__(self, model_path: str, sampling_rate: int = 16000, duration: float = 3.0, device_id: int = 1):
        """
        Initialize audio VAD detector.
        
        Args:
            model_path: Path to audio emotion model
            sampling_rate: Audio sampling rate
            duration: Audio recording duration in seconds
            device_id: Audio input device ID (default: 1)
        """
        self.sampling_rate = sampling_rate
        self.duration = duration
        self.device_id = device_id
        self.vad_buffer = deque(maxlen=30)  # Keep last 30 VAD scores
        
        # Load audio model
        try:
            self.audio_model = audonnx.load(model_path)
            self.audio_available = True
            print("✅ Audio emotion model loaded successfully")
        except Exception as e:
            print(f"❌ Failed to load audio model: {e}")
            self.audio_model = None
            self.audio_available = False
    
    def record_audio_chunk(self) -> Optional[np.ndarray]:
        """Record a short audio chunk."""
        if not self.audio_available:
            return None
            
        try:
            # Check available audio devices
            devices = sd.query_devices()
            default_input = sd.default.device[0] if hasattr(sd.default, 'device') else None
            
            print("🎤 Available Audio Devices:")
            for i, device in enumerate(devices):
                device_type = []
                if device['max_input_channels'] > 0:
                    device_type.append("INPUT")
                if device['max_output_channels'] > 0:
                    device_type.append("OUTPUT")
                
                marker = "👉" if i == default_input else "  "
                print(f"{marker} {i}: {device['name']} ({', '.join(device_type)})")
                print(f"     Channels: In={device['max_input_channels']}, Out={device['max_output_channels']}")
                print(f"     Sample Rate: {device['default_samplerate']}")
            
            print(f"\n🎤 Default input device: {default_input}")
            print(f"🎤 Using device {self.device_id} instead of default device {default_input}")
            print(f"🎤 Recording {self.duration}s audio at {self.sampling_rate}Hz...")
            audio = sd.rec(
                int(self.duration * self.sampling_rate), 
                samplerate=self.sampling_rate, 
                channels=1, 
                dtype='float32',
                device=self.device_id
            )
            sd.wait()
            audio_squeezed = np.squeeze(audio)
            
            # Enhanced debugging
            print(f"🎤 Recorded audio shape: {audio_squeezed.shape}")
            print(f"🎤 Audio range: [{audio_squeezed.min():.6f}, {audio_squeezed.max():.6f}]")
            print(f"🎤 Audio std: {audio_squeezed.std():.6f}")
            print(f"🎤 Non-zero values: {np.count_nonzero(audio_squeezed)}/{len(audio_squeezed)}")
            
            # Check if audio is all zeros
            if np.all(audio_squeezed == 0):
                print("⚠️  WARNING: Audio is all zeros! Check microphone permissions and hardware.")
            
            return audio_squeezed
        except Exception as e:
            print(f"Audio recording error: {e}")
            return None
    
    def predict_vad(self, audio: np.ndarray) -> Optional[Dict[str, float]]:
        """
        Predict VAD (Valence, Arousal, Dominance) from audio.
        
        Args:
            audio: Audio signal
            
        Returns:
            Dictionary with VAD scores and amplitude or None if prediction fails
        """
        if not self.audio_available or audio is None:
            return None
            
        try:
            # Calculate audio amplitude (RMS)
            amplitude = np.sqrt(np.mean(audio ** 2))
            
            # Additional amplitude metrics for debugging
            peak_amplitude = np.max(np.abs(audio))
            mean_amplitude = np.mean(np.abs(audio))
            
            print(f"📊 Audio Analysis:")
            print(f"   RMS Amplitude: {amplitude:.6f}")
            print(f"   Peak Amplitude: {peak_amplitude:.6f}")
            print(f"   Mean Absolute: {mean_amplitude:.6f}")
            
            # If amplitude is very low, suggest checking microphone
            if amplitude < 1e-6:
                print("⚠️  Very low amplitude detected. Check:")
                print("   - Microphone permissions in System Preferences")
                print("   - Microphone hardware connection")
                print("   - Audio input levels")
            
            output = self.audio_model(audio, sampling_rate=self.sampling_rate)
            
            # Debug: Print raw output
            print(f"🔍 Audio model output keys: {list(output.keys())}")
            if "logits" in output:
                print(f"🔍 Audio model logits shape: {output['logits'].shape}")
                logits = np.squeeze(output["logits"])
                print(f"🔍 Squeezed logits: {logits}")
                
                if len(logits) == 3:
                    arousal, dominance, valence = logits
                    
                    # Store VAD scores with amplitude
                    vad_dict = {
                        'valence': float(valence),
                        'arousal': float(arousal), 
                        'dominance': float(dominance),
                        'amplitude': float(amplitude)
                    }
                    
                    print(f"🎭 VAD Scores - V:{valence:.3f} A:{arousal:.3f} D:{dominance:.3f} | Amplitude: {amplitude:.4f}")
                    
                    # Store in buffer
                    self.vad_buffer.append(vad_dict)
                    
                    return vad_dict
                else:
                    print(f"❌ Unexpected logits length: {len(logits)}, expected 3")
            else:
                print(f"❌ No 'logits' key in output: {list(output.keys())}")
            
        except Exception as e:
            print(f"Audio prediction error: {e}")
            import traceback
            traceback.print_exc()
            
        return None
    
    def get_latest_vad(self) -> Optional[Dict[str, float]]:
        """Get the latest VAD scores."""
        if len(self.vad_buffer) > 0:
            return self.vad_buffer[-1]
        return None
    
    def get_smoothed_vad(self) -> Optional[Dict[str, float]]:
        """Get smoothed VAD scores using exponential weighted average."""
        if len(self.vad_buffer) == 0:
            return None
        
        # Use exponential weighted average
        weights = np.exp(np.linspace(-1, 0, len(self.vad_buffer)))
        
        valence_avg = np.average([vad['valence'] for vad in self.vad_buffer], weights=weights)
        arousal_avg = np.average([vad['arousal'] for vad in self.vad_buffer], weights=weights)
        dominance_avg = np.average([vad['dominance'] for vad in self.vad_buffer], weights=weights)
        amplitude_avg = np.average([vad['amplitude'] for vad in self.vad_buffer], weights=weights)
        
        return {
            'valence': float(valence_avg),
            'arousal': float(arousal_avg),
            'dominance': float(dominance_avg),
            'amplitude': float(amplitude_avg)
        }


class AudioVADProcessor:
    """Audio VAD processing with threading support."""
    
    def __init__(self, detector: AudioVADDetector):
        """
        Initialize audio VAD processor.
        
        Args:
            detector: AudioVADDetector instance
        """
        self.detector = detector
        self.vad_queue = queue.Queue(maxsize=5)
        self.audio_thread = None
        self.running = False
    
    def start_processing(self):
        """Start audio processing thread."""
        if not self.detector.audio_available:
            print("⚠️ Audio not available, skipping audio processing")
            return
            
        self.running = True
        self.audio_thread = threading.Thread(target=self._audio_processing_loop)
        self.audio_thread.daemon = True
        self.audio_thread.start()
        print("✅ Audio VAD processing thread started")
    
    def stop_processing(self):
        """Stop audio processing thread."""
        print("🔇 Stopping audio processing...")
        self.running = False
        if self.audio_thread and self.audio_thread.is_alive():
            print("⏳ Waiting for audio thread to finish...")
            self.audio_thread.join(timeout=3)
            if self.audio_thread.is_alive():
                print("⚠️  Audio thread did not stop gracefully")
            else:
                print("✅ Audio thread stopped successfully")
        
        # Try to stop any ongoing audio recording
        try:
            sd.stop()
            print("🎤 Audio recording stopped")
        except Exception as e:
            print(f"⚠️  Error stopping audio: {e}")
    
    def _audio_processing_loop(self):
        """Background thread for audio processing."""
        while self.running:
            try:
                # Record audio
                audio = self.detector.record_audio_chunk()
                if audio is not None:
                    # Predict VAD from audio
                    vad_scores = self.detector.predict_vad(audio)
                    if vad_scores is not None:
                        # Put result in queue for main thread
                        if not self.vad_queue.full():
                            self.vad_queue.put(vad_scores)
                
            except Exception as e:
                print(f"Audio thread error: {e}")
                time.sleep(1)  # Wait before retrying
    
    def get_vad_from_queue(self) -> Optional[Dict[str, float]]:
        """Get VAD scores from queue (non-blocking)."""
        try:
            return self.vad_queue.get_nowait()
        except queue.Empty:
            return None
    
    def process_queue(self) -> list:
        """Process all VAD scores in queue and return them."""
        vad_results = []
        try:
            while not self.vad_queue.empty():
                vad_scores = self.vad_queue.get_nowait()
                vad_results.append(vad_scores)
        except queue.Empty:
            pass
        return vad_results
