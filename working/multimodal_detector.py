"""
Multi-modal Real-time Stress Detection System
Combines video pose analysis and audio VAD recognition for comprehensive assessment.
"""

import cv2
import numpy as np
import time
import os
import signal
import sys
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
        
        # Initialize audio detector with device 1
        self.audio_detector = AudioVADDetector(audio_model_path, device_id=1)
        self.audio_processor = AudioVADProcessor(self.audio_detector)
        
        self.confidence_threshold = confidence_threshold
        self.running = False
        self.cap = None
        
        # Setup signal handlers for proper cleanup
        self.setup_signal_handlers()
    
    def setup_signal_handlers(self):
        """Setup signal handlers for graceful shutdown."""
        def signal_handler(signum, frame):
            print(f"\n🛑 Received signal {signum}. Shutting down gracefully...")
            self.cleanup()
            sys.exit(0)
        
        # Handle various termination signals
        signal.signal(signal.SIGINT, signal_handler)   # Ctrl+C
        signal.signal(signal.SIGTERM, signal_handler)  # Termination
        
        # Handle Ctrl+Z (SIGTSTP) - suspend signal
        def suspend_handler(signum, frame):
            print(f"\n⏸️  Received suspend signal {signum}. Cleaning up resources...")
            self.cleanup()
            # Re-raise the signal to actually suspend
            signal.signal(signal.SIGTSTP, signal.SIG_DFL)
            os.kill(os.getpid(), signal.SIGTSTP)
        
        signal.signal(signal.SIGTSTP, suspend_handler)
    
    def cleanup(self):
        """Clean up all resources."""
        print("🧹 Cleaning up resources...")
        
        # Stop audio processing
        if hasattr(self, 'audio_processor') and self.audio_processor:
            self.audio_processor.stop_processing()
        
        # Release camera
        if self.cap and self.cap.isOpened():
            self.cap.release()
            print("📷 Camera released")
        
        # Close all OpenCV windows
        cv2.destroyAllWindows()
        print("🖼️  OpenCV windows closed")
        
        self.running = False
        print("✅ Cleanup completed")
    
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
        
        # VAD scores with amplitude
        if vad_scores is not None:
            vad_text = f"VAD - V:{vad_scores['valence']:.2f} A:{vad_scores['arousal']:.2f} D:{vad_scores['dominance']:.2f}"
            if 'amplitude' in vad_scores:
                vad_text += f" | Amp:{vad_scores['amplitude']:.4f}"
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
        self.cap = cv2.VideoCapture(camera_id)
        if not self.cap.isOpened():
            print("❌ Failed to open camera.")
            return
        
        # Start audio processing
        self.audio_processor.start_processing()
        self.running = True
        
        print("✅ Multi-modal detection started.")
        print("   Press 'q' to quit")
        print("   Press Ctrl+C to interrupt")
        print("   Press Ctrl+Z to suspend")
        
        frame_count = 0
        fps_start_time = time.time()
        
        try:
            while self.running:
                ret, frame = self.cap.read()
                if not ret:
                    print("❌ Failed to read frame from camera")
                    break
                
                frame_count += 1
                
                # Process video frame
                keypoints = self.video_detector.process_frame(frame)
                
                # Predict video stress
                video_stress = None
                if len(self.video_detector.pose_buffer) == self.video_detector.sequence_length:
                    video_stress = self.video_detector.predict_stress()
                
                # Get latest VAD scores from audio and print them with amplitude
                vad_results = self.audio_processor.process_queue()
                for vad in vad_results:
                    amp_str = f" | Amplitude: {vad['amplitude']:.4f}" if 'amplitude' in vad else ""
                    print(f"� AUDIO - VAD: V:{vad['valence']:.3f} A:{vad['arousal']:.3f} D:{vad['dominance']:.3f}{amp_str}")
                
                # Print video stress separately and clearly
                if video_stress is not None:
                    print(f"📹 VIDEO - Stress Score: {video_stress:.3f}")
                
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
                    
                    # Print comprehensive stats
                    print(f"\n📊 STATS (FPS: {fps:.1f})")
                    print(f"   📹 Video Stress: {f'{smoothed_stress:.3f}' if smoothed_stress is not None else 'N/A'}")
                    
                    if smoothed_vad:
                        amp_str = f" | Amplitude: {smoothed_vad['amplitude']:.4f}" if 'amplitude' in smoothed_vad else ""
                        print(f"   🎧 Audio VAD: V:{smoothed_vad['valence']:.2f} A:{smoothed_vad['arousal']:.2f} D:{smoothed_vad['dominance']:.2f}{amp_str}")
                    else:
                        print("   🎧 Audio VAD: N/A")
                    print("")  # Empty line for readability
                
                cv2.imshow("Multi-Modal Stress Detection", annotated_frame)
                
                # Check for 'q' key to quit
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    print("👋 Quit requested by user")
                    break
        
        except KeyboardInterrupt:
            print("\n⌨️  Keyboard interrupt received")
        except Exception as e:
            print(f"❌ Unexpected error: {e}")
            import traceback
            traceback.print_exc()
        
        finally:
            # Always cleanup regardless of how we exit
            self.cleanup()


def main():
    """Main function to run multi-modal stress detection."""
    base_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(base_dir)  # Go up one level from working folder
    
    # Model paths (relative to parent directory)
    stress_model_path = os.path.join(parent_dir, "training_results", "models", "gru_stress_model.tflite")
    movenet_model_path = os.path.join(parent_dir, "movenet_thunder.tflite")
    audio_model_path = os.path.join(parent_dir, "model")  # Directory containing audio model
    
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
    
    detector = None
    try:
        # Initialize multi-modal detector
        print("🚀 Initializing multi-modal detector...")
        detector = MultiModalDetector(
            stress_model_path=stress_model_path,
            movenet_model_path=movenet_model_path,
            audio_model_path=audio_model_path,
            sequence_length=30,
            confidence_threshold=0.3
        )
        
        # Run detection
        detector.run_detection(camera_id=0)
        
    except KeyboardInterrupt:
        print("\n⌨️  Keyboard interrupt in main")
    except Exception as e:
        print(f"❌ Error in main: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # Ensure cleanup happens even if detector wasn't created
        if detector:
            detector.cleanup()
        print("👋 Application terminated")


if __name__ == "__main__":
    # Additional signal handling at module level
    def emergency_cleanup(signum, frame):
        print(f"\n🚨 Emergency cleanup triggered by signal {signum}")
        try:
            cv2.destroyAllWindows()
            # Stop any ongoing sounddevice operations
            try:
                import sounddevice as sd
                sd.stop()
            except:
                pass
        except:
            pass
        sys.exit(1)
    
    signal.signal(signal.SIGTERM, emergency_cleanup)
    signal.signal(signal.SIGINT, emergency_cleanup)
    
    main()
