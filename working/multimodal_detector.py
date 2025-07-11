"""
Multi-modal Real-time Stress Detection System
Combines video pose analysis and audio VAD recognition for comprehensive assessment.
"""

import cv2
import numpy as np
import time
import os
from typing import Optional, Dict

# Import our custom modules
from audio_vad_detector import AudioVADDetector, AudioVADProcessor
from video_stress_detector import VideoStressDetector


class MultiModalDetector:
    """Multi-modal stress detection combining video pose and audio VAD analysis."""
    
    def __init__(self, 
                 stress_model_path: str,
                 movenet_model_path: str,
                 audio_model_path: str,
                 sequence_length: int = 30,
                 confidence_threshold: float = 0.3):
        """
        Initialize multi-modal detector.
        
        Args:
            stress_model_path: Path to pose-based stress model
            movenet_model_path: Path to MoveNet model
            audio_model_path: Path to audio emotion model
            sequence_length: Video sequence length
            confidence_threshold: Pose confidence threshold
        """
        # Initialize video detector
        self.video_detector = VideoStressDetector(
            stress_model_path=stress_model_path,
            movenet_model_path=movenet_model_path,
            sequence_length=sequence_length,
            confidence_threshold=confidence_threshold
        )
        
        # Initialize audio detector
        self.audio_detector = AudioVADDetector(audio_model_path)
        self.audio_processor = AudioVADProcessor(self.audio_detector)
        
        self.confidence_threshold = confidence_threshold
    
    def draw_info(self, frame: np.ndarray, keypoints: np.ndarray, 
                  video_stress: Optional[float], vad_scores: Optional[Dict[str, float]]) -> np.ndarray:
        """Draw multi-modal information on frame."""
        annotated_frame = self.video_detector.draw_keypoints(frame, keypoints)
        h, w = frame.shape[:2]
        
        # Draw information
        y_offset = 30
        
        # Video stress
        if video_stress is not None:
            cv2.putText(annotated_frame, f"Video Stress: {video_stress:.3f}", 
                       (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
            y_offset += 30
        
        # VAD scores
        if vad_scores is not None:
            vad_text = f"VAD - V:{vad_scores['valence']:.2f} A:{vad_scores['arousal']:.2f} D:{vad_scores['dominance']:.2f}"
            cv2.putText(annotated_frame, vad_text, 
                       (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 200), 2)
            y_offset += 30
        
        # Overall stress (video-only since we're not converting audio to stress)
        if video_stress is not None:
            # Determine stress level and color
            if video_stress < 0.3:
                level, color = "LOW", (0, 255, 0)
            elif video_stress < 0.7:
                level, color = "MEDIUM", (0, 255, 255)
            else:
                level, color = "HIGH", (0, 0, 255)
            
            cv2.putText(annotated_frame, f"Overall Stress: {video_stress:.3f} ({level})", 
                       (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
            
            # Stress bar
            bar_width, bar_height = 200, 20
            bar_x, bar_y = 10, y_offset + 20
            
            cv2.rectangle(annotated_frame, (bar_x, bar_y), 
                         (bar_x + bar_width, bar_y + bar_height), (100, 100, 100), -1)
            
            fill_width = int(video_stress * bar_width)
            cv2.rectangle(annotated_frame, (bar_x, bar_y),
                         (bar_x + fill_width, bar_y + bar_height), color, -1)
        
        # Buffer status
        video_status = self.video_detector.get_buffer_status()
        vad_buffer_size = len(self.audio_detector.vad_buffer)
        buffer_text = f"{video_status} | VAD: {vad_buffer_size}"
        cv2.putText(annotated_frame, buffer_text, (10, h - 20),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        return annotated_frame
    
    def run_detection(self, camera_id: int = 0):
        """Run multi-modal detection."""
        cap = cv2.VideoCapture(camera_id)
        if not cap.isOpened():
            print("❌ Failed to open camera.")
            return
        
        # Start audio processing
        self.audio_processor.start_processing()
        
        print("✅ Multi-modal detection started. Press 'q' to quit.")
        
        frame_count = 0
        fps_start_time = time.time()
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                frame_count += 1
                
                # Process video frame
                keypoints = self.video_detector.process_frame(frame)
                
                # Predict video stress
                video_stress = None
                if len(self.video_detector.pose_buffer) == self.video_detector.sequence_length:
                    video_stress = self.video_detector.predict_stress()
                
                # Get latest VAD scores from audio
                vad_results = self.audio_processor.process_queue()
                for vad in vad_results:
                    print(f"🎭 VAD - V:{vad['valence']:.3f} A:{vad['arousal']:.3f} D:{vad['dominance']:.3f}")
                
                latest_vad = self.audio_detector.get_latest_vad()
                
                # Draw annotations
                annotated_frame = self.draw_info(frame, keypoints, video_stress, latest_vad)
                
                # Display FPS and stats
                if frame_count % 30 == 0:
                    fps = 30 / (time.time() - fps_start_time)
                    fps_start_time = time.time()
                    
                    # Get smoothed values
                    smoothed_stress = self.video_detector.get_smoothed_stress()
                    smoothed_vad = self.audio_detector.get_smoothed_vad()
                    
                    vad_str = f"V:{smoothed_vad['valence']:.2f} A:{smoothed_vad['arousal']:.2f} D:{smoothed_vad['dominance']:.2f}" if smoothed_vad else 'N/A'
                    
                    print("FPS: {:.1f} | Video Stress: {} | VAD: {}".format(
                        fps,
                        f"{smoothed_stress:.3f}" if smoothed_stress is not None else 'N/A',
                        vad_str
                    ))
                
                cv2.imshow("Multi-Modal Stress Detection", annotated_frame)
                
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
        
        finally:
            # Cleanup
            self.audio_processor.stop_processing()
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
        print("Will continue with video pose tracking only...")
    
    if not os.path.exists(audio_model_path):
        print(f"⚠️ Audio model not found at: {audio_model_path}")
        print("Will continue with video-only detection...")
    
    try:
        # Initialize multi-modal detector
        detector = MultiModalDetector(
            stress_model_path=stress_model_path,
            movenet_model_path=movenet_model_path,
            audio_model_path=audio_model_path,
            sequence_length=30,
            confidence_threshold=0.3
        )
        
        # Run detection
        detector.run_detection(camera_id=0)
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
