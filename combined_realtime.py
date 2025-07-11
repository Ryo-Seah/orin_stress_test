"""
Multi-modal Real-time Stress Detection System
Combines video pose analysis and audio emotion recognition for comprehensive stress assessment.
"""

import cv2
import numpy as np
import tensorflow as tf
import time
import os
import sys
from collections import deque
from typing import Optional, Tuple, Dict
import threading
import queue
import sounddevice as sd
import audonnx

# Add the stress training directory to path
stress_training_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'stress_training')
sys.path.append(stress_training_dir)

# Import with detailed error reporting
print(f"📁 Script location: {os.path.dirname(os.path.abspath(__file__))}")
print(f"📁 Stress training dir: {stress_training_dir}")
print(f"📁 Stress training exists: {os.path.exists(stress_training_dir)}")

try:
    print("🔄 Attempting to import pose analysis modules...")
    from stress_training.data_processing import PoseFeatureExtractor
    from stress_training.data_processing import StressScoreCalculator
    POSE_IMPORTS_AVAILABLE = True
    print("✅ Pose analysis modules imported successfully")
    
except ImportError as e:
    print(f"❌ ImportError: {e}")
    print(f"❌ Error type: {type(e).__name__}")

except Exception as e:
    print(f"❌ Unexpected error during import: {e}")
    import traceback
    traceback.print_exc()
    PoseFeatureExtractor = None
    StressScoreCalculator = None
    POSE_IMPORTS_AVAILABLE = False


class AudioStressDetector:
    """Audio-based stress detection using emotion recognition."""
    
    def __init__(self, model_path: str, sampling_rate: int = 16000, duration: float = 3.0):
        """
        Initialize audio stress detector.
        
        Args:
            model_path: Path to audio emotion model
            sampling_rate: Audio sampling rate
            duration: Audio recording duration in seconds
        """
        self.sampling_rate = sampling_rate
        self.duration = duration
        self.audio_buffer = deque(maxlen=10)  # Keep last 10 audio predictions
        self.vad_scores = deque(maxlen=10)  # Keep last 10 VAD scores
        
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
            print(f"🎤 Recording {self.duration}s audio at {self.sampling_rate}Hz...")
            audio = sd.rec(
                int(self.duration * self.sampling_rate), 
                samplerate=self.sampling_rate, 
                channels=1, 
                dtype='float32'
            )
            sd.wait()
            audio_squeezed = np.squeeze(audio)
            print(f"🎤 Recorded audio shape: {audio_squeezed.shape}, range: [{audio_squeezed.min():.3f}, {audio_squeezed.max():.3f}]")
            return audio_squeezed
        except Exception as e:
            print(f"Audio recording error: {e}")
            return None
    
    def predict_audio_vad(self, audio: np.ndarray) -> Optional[Dict[str, float]]:
        """
        Predict VAD (Valence, Arousal, Dominance) from audio.
        
        Args:
            audio: Audio signal
            
        Returns:
            Dictionary with VAD scores or None if prediction fails
        """
        if not self.audio_available or audio is None:
            return None
            
        try:
            output = self.audio_model(audio, sampling_rate=self.sampling_rate)
            
            # Debug: Print raw output
            print(f"🔍 Audio model output keys: {list(output.keys())}")
            if "logits" in output:
                print(f"🔍 Audio model logits shape: {output['logits'].shape}")
                logits = np.squeeze(output["logits"])
                print(f"🔍 Squeezed logits: {logits}")
                
                if len(logits) == 3:
                    arousal, dominance, valence = logits
                    
                    # Store VAD scores
                    vad_dict = {
                        'valence': float(valence),
                        'arousal': float(arousal), 
                        'dominance': float(dominance)
                    }
                    
                    print(f"🎭 VAD Scores - Valence: {valence:.3f}, Arousal: {arousal:.3f}, Dominance: {dominance:.3f}")
                    
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
    
    def get_smoothed_audio_stress(self) -> Optional[float]:
        """Get smoothed audio stress score."""
        if len(self.audio_buffer) == 0:
            return None
        
        # Use exponential weighted average
        weights = np.exp(np.linspace(-1, 0, len(self.audio_buffer)))
        weighted_avg = np.average(list(self.audio_buffer), weights=weights)
        return float(weighted_avg)


class MultiModalStressDetector:
    """Multi-modal stress detection combining video pose and audio analysis."""
    
    def __init__(self, 
                 stress_model_path: str,
                 movenet_model_path: str,
                 audio_model_path: str,
                 sequence_length: int = 30,
                 confidence_threshold: float = 0.3):
        """
        Initialize multi-modal stress detector.
        
        Args:
            stress_model_path: Path to pose-based stress model
            movenet_model_path: Path to MoveNet model
            audio_model_path: Path to audio emotion model
            sequence_length: Video sequence length
            confidence_threshold: Pose confidence threshold
        """
        self.sequence_length = sequence_length
        self.confidence_threshold = confidence_threshold
        
        # Initialize buffers
        self.pose_buffer = deque(maxlen=sequence_length)
        self.video_stress_scores = deque(maxlen=50)
        self.audio_stress_scores = deque(maxlen=50)
        
        # Load models
        self.load_movenet_model(movenet_model_path)
        self.load_pose_stress_model(stress_model_path)
        self.setup_feature_extractor()
        
        # Initialize audio detector
        self.audio_detector = AudioStressDetector(audio_model_path)
        
        # Audio processing thread
        self.audio_queue = queue.Queue(maxsize=5)
        self.audio_thread = None
        self.running = False
    
    def load_movenet_model(self, model_path: str):
        """Load MoveNet TFLite model."""
        print(f"Loading MoveNet model from: {model_path}")
        
        self.movenet_interpreter = tf.lite.Interpreter(model_path=model_path)
        self.movenet_interpreter.allocate_tensors()
        
        self.movenet_input_details = self.movenet_interpreter.get_input_details()
        self.movenet_output_details = self.movenet_interpreter.get_output_details()
        self.input_size = self.movenet_input_details[0]['shape'][2]
        
        print(f"MoveNet loaded successfully. Input size: {self.input_size}")
    
    def load_pose_stress_model(self, model_path: str):
        """Load pose-based stress detection model."""
        if not os.path.exists(model_path):
            print(f"Pose stress model not found at: {model_path}")
            self.stress_interpreter = None
            return
            
        try:
            self.stress_interpreter = tf.lite.Interpreter(model_path=model_path)
            self.stress_interpreter.allocate_tensors()
            
            self.stress_input_details = self.stress_interpreter.get_input_details()
            self.stress_output_details = self.stress_interpreter.get_output_details()
            
            print("✅ Pose stress detection model loaded successfully")
            print(f"Input shape: {self.stress_input_details[0]['shape']}")
            print(f"Output shape: {self.stress_output_details[0]['shape']}")
            
        except Exception as e:
            print(f"Error loading pose stress model: {e}")
            self.stress_interpreter = None
    
    def setup_feature_extractor(self):
        """Setup pose feature extraction."""
        if not POSE_IMPORTS_AVAILABLE:
            print("Warning: Pose feature extraction modules not available")
            self.feature_extractor = None
            return
            
        try:
            self.feature_extractor = PoseFeatureExtractor(self.confidence_threshold)
            print("✅ Pose feature extractor setup successfully")
        except Exception as e:
            print(f"Warning: Could not setup pose feature extractor: {e}")
            self.feature_extractor = None
    
    def detect_pose_movenet(self, frame: np.ndarray) -> np.ndarray:
        """Detect pose using MoveNet."""
        img = cv2.resize(frame, (self.input_size, self.input_size))
        input_data = np.expand_dims(img, axis=0).astype(np.uint8)
        
        self.movenet_interpreter.set_tensor(
            self.movenet_input_details[0]['index'], 
            input_data
        )
        self.movenet_interpreter.invoke()
        
        keypoints = self.movenet_interpreter.get_tensor(
            self.movenet_output_details[0]['index']
        )[0][0]  # Shape: (17, 3)
        
        return keypoints
    
    def convert_movenet_to_coco_format(self, movenet_keypoints: np.ndarray) -> np.ndarray:
        """Convert MoveNet keypoints to COCO format."""
        coco_keypoints = np.zeros((18, 3))
        
        # MoveNet to COCO mapping
        mapping = {
            0: 0,   1: 15,  2: 14,  3: 17,  4: 16,  5: 5,   6: 2,   7: 6,   8: 3,
            9: 7,   10: 4,  11: 11, 12: 8,  13: 12, 14: 9,  15: 13, 16: 10,
        }
        
        for movenet_idx, coco_idx in mapping.items():
            coco_keypoints[coco_idx] = movenet_keypoints[movenet_idx]
        
        # Estimate neck position
        left_shoulder = coco_keypoints[5]
        right_shoulder = coco_keypoints[2]
        
        if left_shoulder[2] > self.confidence_threshold and right_shoulder[2] > self.confidence_threshold:
            coco_keypoints[1] = [(left_shoulder[0] + right_shoulder[0]) / 2,
                                (left_shoulder[1] + right_shoulder[1]) / 2,
                                min(left_shoulder[2], right_shoulder[2])]
        
        return coco_keypoints
    
    def predict_video_stress(self) -> Optional[float]:
        """Predict stress from video pose sequence."""
        if (len(self.pose_buffer) < self.sequence_length or 
            self.stress_interpreter is None or 
            self.feature_extractor is None):
            return None
        
        try:
            pose_sequence = np.array(list(self.pose_buffer))
            features = self.feature_extractor.extract_pose_features(pose_sequence, input_format='coco17')
            features_input = features.reshape(1, *features.shape).astype(np.float32)
            
            self.stress_interpreter.set_tensor(
                self.stress_input_details[0]['index'], 
                features_input
            )
            self.stress_interpreter.invoke()
            
            stress_score = self.stress_interpreter.get_tensor(
                self.stress_output_details[0]['index']
            )[0][0]
            
            return float(stress_score)
            
        except Exception as e:
            print(f"Video stress prediction error: {e}")
            return None
    
    def audio_processing_thread(self):
        """Background thread for audio processing."""
        while self.running:
            try:
                # Record audio
                audio = self.audio_detector.record_audio_chunk()
                if audio is not None:
                    # Predict VAD from audio
                    vad_scores = self.audio_detector.predict_audio_vad(audio)
                    if vad_scores is not None:
                        
                        # Store VAD scores in a buffer for display
                        if not hasattr(self.audio_detector, 'vad_buffer'):
                            self.audio_detector.vad_buffer = deque(maxlen=30)  # Keep last 30 VAD scores
                        self.audio_detector.vad_buffer.append(vad_scores)
                        
                        # Put result in queue for main thread
                        if not self.audio_queue.full():
                            self.audio_queue.put(vad_scores)
                
            except Exception as e:
                print(f"Audio thread error: {e}")
                time.sleep(1)  # Wait before retrying
    
    def get_combined_stress_score(self) -> Dict[str, Optional[float]]:
        """
        Get stress scores from video modality only.
        
        Returns:
            Dictionary with video_stress only (audio stress removed)
        """
        # Get smoothed video scores
        video_stress = None
        if len(self.video_stress_scores) > 0:
            weights = np.exp(np.linspace(-1, 0, len(self.video_stress_scores)))
            video_stress = float(np.average(list(self.video_stress_scores), weights=weights))
        
        return {
            'video_stress': video_stress,
            'audio_stress': None,  # No longer computing audio stress
            'combined_stress': video_stress  # Only video stress now
        }
    
    def draw_multi_modal_info(self, frame: np.ndarray, keypoints: np.ndarray, 
                             stress_scores: Dict[str, Optional[float]]) -> np.ndarray:
        """Draw multi-modal stress information on frame."""
        annotated_frame = frame.copy()
        h, w = frame.shape[:2]
        
        # Draw pose keypoints
        for i, (x, y, c) in enumerate(keypoints):
            if c > self.confidence_threshold:
                cx, cy = int(x * w), int(y * h)
                cv2.circle(annotated_frame, (cx, cy), 4, (0, 255, 0), -1)
        
        # Draw stress information
        y_offset = 30
        
        # Video stress
        video_stress = stress_scores['video_stress']
        if video_stress is not None:
            cv2.putText(annotated_frame, f"Video Stress: {video_stress:.3f}", 
                       (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
            y_offset += 30
        
        # Display VAD scores if available (no audio stress conversion)
        vad_scores = self.get_latest_vad_scores()
        if vad_scores is not None:
            vad_text = f"VAD - V:{vad_scores['valence']:.2f} A:{vad_scores['arousal']:.2f} D:{vad_scores['dominance']:.2f}"
            cv2.putText(annotated_frame, vad_text, 
                       (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 200), 2)
            y_offset += 30
        
        # Video-only stress (since audio only provides VAD now)
        combined_stress = stress_scores['combined_stress']
        if combined_stress is not None:
            # Determine stress level and color
            if combined_stress < 0.3:
                level, color = "LOW", (0, 255, 0)
            elif combined_stress < 0.7:
                level, color = "MEDIUM", (0, 255, 255)
            else:
                level, color = "HIGH", (0, 0, 255)
            
            cv2.putText(annotated_frame, f"Overall Stress: {combined_stress:.3f} ({level})", 
                       (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
            
            # Stress bar
            bar_width, bar_height = 200, 20
            bar_x, bar_y = 10, y_offset + 20
            
            cv2.rectangle(annotated_frame, (bar_x, bar_y), 
                         (bar_x + bar_width, bar_y + bar_height), (100, 100, 100), -1)
            
            fill_width = int(combined_stress * bar_width)
            cv2.rectangle(annotated_frame, (bar_x, bar_y),
                         (bar_x + fill_width, bar_y + bar_height), color, -1)
        
        # Buffer status
        vad_buffer_size = len(self.audio_detector.vad_buffer) if hasattr(self.audio_detector, 'vad_buffer') else 0
        buffer_text = f"Video: {len(self.pose_buffer)}/{self.sequence_length} | VAD: {vad_buffer_size}"
        cv2.putText(annotated_frame, buffer_text, (10, h - 20),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        return annotated_frame
    
    def run_multimodal_detection(self, camera_id: int = 0):
        """Run multi-modal stress detection."""
        cap = cv2.VideoCapture(camera_id)
        if not cap.isOpened():
            print("❌ Failed to open camera.")
            return
        
        # Start audio processing thread
        self.running = True
        if self.audio_detector.audio_available:
            self.audio_thread = threading.Thread(target=self.audio_processing_thread)
            self.audio_thread.daemon = True
            self.audio_thread.start()
            print("✅ Audio processing thread started")
        
        print("✅ Multi-modal stress detection started. Press 'q' to quit.")
        
        frame_count = 0
        fps_start_time = time.time()
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                frame_count += 1
                
                # Process video
                movenet_keypoints = self.detect_pose_movenet(frame)
                coco_keypoints = self.convert_movenet_to_coco_format(movenet_keypoints)
                self.pose_buffer.append(coco_keypoints)
                
                # Predict video stress
                if len(self.pose_buffer) == self.sequence_length:
                    video_stress = self.predict_video_stress()
                    if video_stress is not None:
                        self.video_stress_scores.append(video_stress)
                
                # Get VAD scores from queue (non-blocking)
                try:
                    while not self.audio_queue.empty():
                        vad_scores = self.audio_queue.get_nowait()
                        # Display VAD scores
                        print(f" VAD - V:{vad_scores['valence']:.3f} A:{vad_scores['arousal']:.3f} D:{vad_scores['dominance']:.3f}")
                except queue.Empty:
                    pass
                
                # Get combined stress scores
                stress_scores = self.get_combined_stress_score()
                
                # Draw annotations
                annotated_frame = self.draw_multi_modal_info(frame, coco_keypoints, stress_scores)
                
                # Display FPS and stats
                if frame_count % 30 == 0:
                    fps = 30 / (time.time() - fps_start_time)
                    fps_start_time = time.time()
                    
                    # Get latest VAD scores for display
                    latest_vad = self.get_latest_vad_scores()
                    vad_str = f"V:{latest_vad['valence']:.2f} A:{latest_vad['arousal']:.2f} D:{latest_vad['dominance']:.2f}" if latest_vad else 'N/A'
                    
                    print("FPS: {:.1f} | Video Stress: {} | VAD: {}".format(
                        fps,
                        f"{stress_scores['video_stress']:.3f}" if stress_scores['video_stress'] is not None else 'N/A',
                        vad_str
                    ))
                
                cv2.imshow("Multi-Modal Stress Detection", annotated_frame)
                
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
        
        finally:
            # Cleanup
            self.running = False
            if self.audio_thread:
                self.audio_thread.join(timeout=2)
            cap.release()
            cv2.destroyAllWindows()


def main():
    """Main function to run multi-modal stress detection."""
    base_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Model paths
    stress_model_path = os.path.join(base_dir, "training_results", "models", "gru_stress_model.tflite")
    movenet_model_path = os.path.join(base_dir, "movenet_thunder.tflite")
    audio_model_path = os.path.join(base_dir, "model")  # Directory containing audio model
    
    # Check if models exist
    if not os.path.exists(movenet_model_path):
        print(f"❌ MoveNet model not found at: {movenet_model_path}")
        return
    
    if not os.path.exists(stress_model_path):
        print(f"⚠️ Pose stress model not found at: {stress_model_path}")
        print("Will use audio-only stress detection...")
    
    if not os.path.exists(audio_model_path):
        print(f"⚠️ Audio model not found at: {audio_model_path}")
        print("Will use video-only stress detection...")
    
    try:
        # Initialize multi-modal detector
        detector = MultiModalStressDetector(
            stress_model_path=stress_model_path,
            movenet_model_path=movenet_model_path,
            audio_model_path=audio_model_path,
            sequence_length=30,
            confidence_threshold=0.3
        )
        
        # Run detection
        detector.run_multimodal_detection(camera_id=0)
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()