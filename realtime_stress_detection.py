"""
Real-time stress detection using trained models and MoveNet pose estimation.
Integrates with the existing MoveNet setup for live camera inference.
"""

import cv2
import numpy as np
import tensorflow as tf
import time
import os
import sys
from collections import deque
from typing import Optional, Tuple

# Add the stress training directory to path
stress_training_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'stress_training')
sys.path.append(stress_training_dir)

try:
    from evaluate import StressDetectionEvaluator
    from data_loader import BOLDDataLoader
except ImportError:
    print("Warning: Could not import stress detection modules. Make sure training modules are available.")
    StressDetectionEvaluator = None
    BOLDDataLoader = None


class RealTimeStressDetector:
    """
    Real-time stress detection combining MoveNet pose estimation 
    with trained temporal stress detection models.
    """
    
    def __init__(self, 
                 stress_model_path: str,
                 movenet_model_path: str,
                 sequence_length: int = 30,
                 confidence_threshold: float = 0.3):
        """
        Initialize the real-time stress detector.
        
        Args:
            stress_model_path: Path to trained stress detection model
            movenet_model_path: Path to MoveNet TFLite model
            sequence_length: Number of frames to accumulate for prediction
            confidence_threshold: Minimum confidence for pose keypoints
        """
        self.sequence_length = sequence_length
        self.confidence_threshold = confidence_threshold
        
        # Initialize pose sequence buffer
        self.pose_buffer = deque(maxlen=sequence_length)
        self.stress_scores = deque(maxlen=100)  # Keep last 100 predictions for smoothing
        
        # Load MoveNet model
        self.load_movenet_model(movenet_model_path)
        
        # Load stress detection model
        self.load_stress_model(stress_model_path)
        
        # Initialize feature extractor
        self.setup_feature_extractor()
        
    def load_movenet_model(self, model_path: str):
        """Load MoveNet TFLite model."""
        print(f"Loading MoveNet model from: {model_path}")
        
        self.movenet_interpreter = tf.lite.Interpreter(model_path=model_path)
        self.movenet_interpreter.allocate_tensors()
        
        self.movenet_input_details = self.movenet_interpreter.get_input_details()
        self.movenet_output_details = self.movenet_interpreter.get_output_details()
        self.input_size = self.movenet_input_details[0]['shape'][2]
        
        print(f"MoveNet loaded successfully. Input size: {self.input_size}")
        
    def load_stress_model(self, model_path: str):
        """Load trained stress detection model."""
        if not os.path.exists(model_path):
            print(f"Stress model not found at: {model_path}")
            self.stress_evaluator = None
            return
            
        try:
            config_path = os.path.join(
                os.path.dirname(os.path.dirname(model_path)), 
                'logs', 
                'training_results.json'
            )
            
            self.stress_evaluator = StressDetectionEvaluator(model_path, config_path)
            print("Stress detection model loaded successfully")
            
        except Exception as e:
            print(f"Error loading stress model: {e}")
            self.stress_evaluator = None
    
    def setup_feature_extractor(self):
        """Setup feature extraction using BOLDDataLoader configuration."""
        try:
            # Create a data loader for feature extraction
            # Note: In production, these parameters should be loaded from saved config
            self.feature_extractor = BOLDDataLoader(
                bold_root='',  # Not needed for feature extraction only
                sequence_length=self.sequence_length,
                overlap_ratio=0.0,  # No overlap needed for real-time
                min_confidence=self.confidence_threshold
            )
        except:
            print("Warning: Could not setup feature extractor. Stress detection disabled.")
            self.feature_extractor = None
    
    def detect_pose_movenet(self, frame: np.ndarray) -> np.ndarray:
        """
        Detect pose using MoveNet.
        
        Args:
            frame: Input image frame
            
        Returns:
            Keypoints array of shape (17, 3) - (x, y, confidence)
        """
        # Resize frame for MoveNet
        img = cv2.resize(frame, (self.input_size, self.input_size))
        input_data = np.expand_dims(img, axis=0).astype(np.uint8)
        
        # Set input tensor
        self.movenet_interpreter.set_tensor(
            self.movenet_input_details[0]['index'], 
            input_data
        )
        
        # Run inference
        self.movenet_interpreter.invoke()
        
        # Get output
        keypoints = self.movenet_interpreter.get_tensor(
            self.movenet_output_details[0]['index']
        )[0][0]  # Shape: (17, 3)
        
        return keypoints
    
    def convert_movenet_to_coco_format(self, movenet_keypoints: np.ndarray) -> np.ndarray:
        """
        Convert MoveNet keypoints (17 points) to COCO format (18 points) used in training.
        
        MoveNet order: nose, left_eye, right_eye, left_ear, right_ear, left_shoulder,
                      right_shoulder, left_elbow, right_elbow, left_wrist, right_wrist,
                      left_hip, right_hip, left_knee, right_knee, left_ankle, right_ankle
        
        COCO order: nose, neck, right_shoulder, right_elbow, right_wrist, left_shoulder,
                   left_elbow, left_wrist, right_hip, right_knee, right_ankle, left_hip,
                   left_knee, left_ankle, right_eye, left_eye, right_ear, left_ear
        """
        # Create COCO keypoints array
        coco_keypoints = np.zeros((18, 3))
        
        # MoveNet to COCO mapping
        mapping = {
            0: 0,   # nose -> nose
            1: 15,  # left_eye -> left_eye  
            2: 14,  # right_eye -> right_eye
            3: 17,  # left_ear -> left_ear
            4: 16,  # right_ear -> right_ear
            5: 5,   # left_shoulder -> left_shoulder
            6: 2,   # right_shoulder -> right_shoulder
            7: 6,   # left_elbow -> left_elbow
            8: 3,   # right_elbow -> right_elbow
            9: 7,   # left_wrist -> left_wrist
            10: 4,  # right_wrist -> right_wrist
            11: 11, # left_hip -> left_hip
            12: 8,  # right_hip -> right_hip
            13: 12, # left_knee -> left_knee
            14: 9,  # right_knee -> right_knee
            15: 13, # left_ankle -> left_ankle
            16: 10, # right_ankle -> right_ankle
        }
        
        # Map keypoints
        for movenet_idx, coco_idx in mapping.items():
            coco_keypoints[coco_idx] = movenet_keypoints[movenet_idx]
        
        # Estimate neck position (not in MoveNet)
        # Neck ≈ midpoint between shoulders
        left_shoulder = coco_keypoints[5]
        right_shoulder = coco_keypoints[2]
        
        if left_shoulder[2] > self.confidence_threshold and right_shoulder[2] > self.confidence_threshold:
            coco_keypoints[1] = [(left_shoulder[0] + right_shoulder[0]) / 2,
                                (left_shoulder[1] + right_shoulder[1]) / 2,
                                min(left_shoulder[2], right_shoulder[2])]
        
        return coco_keypoints
    
    def predict_stress_score(self) -> Optional[float]:
        """
        Predict stress score from accumulated pose sequence.
        
        Returns:
            Stress score (0-1) or None if not enough data
        """
        if (len(self.pose_buffer) < self.sequence_length or 
            self.stress_evaluator is None or 
            self.feature_extractor is None):
            return None
        
        try:
            # Convert buffer to numpy array
            pose_sequence = np.array(list(self.pose_buffer))  # Shape: (seq_len, 18, 3)
            
            # Extract features using the same method as training
            features = self.feature_extractor.extract_pose_features(pose_sequence)
            velocities = self.feature_extractor.compute_velocity_features(features)
            
            # Pad velocities to match sequence length
            padded_velocities = np.vstack([np.zeros((1, velocities.shape[1])), velocities])
            combined_features = np.hstack([features, padded_velocities])
            
            # Reshape for model input
            features_input = combined_features.reshape(1, *combined_features.shape)
            
            # Make prediction
            stress_score = self.stress_evaluator.predict(features_input.astype(np.float32))
            
            return float(stress_score[0])
            
        except Exception as e:
            print(f"Error predicting stress: {e}")
            return None
    
    def get_smoothed_stress_score(self) -> Optional[float]:
        """Get smoothed stress score from recent predictions."""
        if len(self.stress_scores) == 0:
            return None
        
        # Use weighted average with recent scores having higher weight
        weights = np.exp(np.linspace(-1, 0, len(self.stress_scores)))
        weighted_avg = np.average(list(self.stress_scores), weights=weights)
        
        return float(weighted_avg)
    
    def draw_pose_and_stress(self, frame: np.ndarray, keypoints: np.ndarray, 
                           stress_score: Optional[float]) -> np.ndarray:
        """
        Draw pose keypoints and stress information on frame.
        
        Args:
            frame: Input frame
            keypoints: COCO format keypoints (18, 3)
            stress_score: Current stress score
            
        Returns:
            Annotated frame
        """
        annotated_frame = frame.copy()
        h, w = frame.shape[:2]
        
        # Draw keypoints
        for i, (x, y, c) in enumerate(keypoints):
            if c > self.confidence_threshold:
                cx, cy = int(x * w), int(y * h)
                cv2.circle(annotated_frame, (cx, cy), 4, (0, 255, 0), -1)
        
        # Draw stress information
        if stress_score is not None:
            # Stress score text
            stress_text = f"Stress Score: {stress_score:.3f}"
            cv2.putText(annotated_frame, stress_text, (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            # Stress level indicator
            if stress_score < 0.3:
                level = "LOW"
                color = (0, 255, 0)  # Green
            elif stress_score < 0.7:
                level = "MEDIUM"
                color = (0, 255, 255)  # Yellow
            else:
                level = "HIGH"
                color = (0, 0, 255)  # Red
            
            cv2.putText(annotated_frame, f"Level: {level}", (10, 70),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
            
            # Stress bar
            bar_width = 200
            bar_height = 20
            bar_x, bar_y = 10, 90
            
            # Background bar
            cv2.rectangle(annotated_frame, 
                         (bar_x, bar_y), 
                         (bar_x + bar_width, bar_y + bar_height),
                         (100, 100, 100), -1)
            
            # Stress level bar
            fill_width = int(stress_score * bar_width)
            cv2.rectangle(annotated_frame,
                         (bar_x, bar_y),
                         (bar_x + fill_width, bar_y + bar_height),
                         color, -1)
        
        # Buffer status
        buffer_text = f"Buffer: {len(self.pose_buffer)}/{self.sequence_length}"
        cv2.putText(annotated_frame, buffer_text, (10, frame.shape[0] - 20),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        return annotated_frame
    
    def run_realtime_detection(self, camera_id: int = 0):
        """
        Run real-time stress detection from camera feed.
        
        Args:
            camera_id: Camera device ID
        """
        cap = cv2.VideoCapture(camera_id)
        if not cap.isOpened():
            print("❌ Failed to open camera.")
            return
        
        print("✅ Real-time stress detection started. Press 'q' to quit.")
        print("📋 Accumulating pose data for stress prediction...")
        
        frame_count = 0
        fps_start_time = time.time()
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_count += 1
            
            # Detect pose with MoveNet
            movenet_keypoints = self.detect_pose_movenet(frame)
            
            # Convert to COCO format for stress model
            coco_keypoints = self.convert_movenet_to_coco_format(movenet_keypoints)
            
            # Add to pose buffer
            self.pose_buffer.append(coco_keypoints)
            
            # Predict stress score
            stress_score = None
            if len(self.pose_buffer) == self.sequence_length:
                stress_score = self.predict_stress_score()
                if stress_score is not None:
                    self.stress_scores.append(stress_score)
                    stress_score = self.get_smoothed_stress_score()
            
            # Draw annotations
            annotated_frame = self.draw_pose_and_stress(frame, coco_keypoints, stress_score)
            
            # Calculate and display FPS
            if frame_count % 30 == 0:
                fps = 30 / (time.time() - fps_start_time)
                fps_start_time = time.time()
                print(f"FPS: {fps:.1f}, Buffer: {len(self.pose_buffer)}, "
                      f"Stress: {stress_score:.3f}" if stress_score else "Stress: N/A")
            
            # Display frame
            cv2.imshow("Real-time Stress Detection", annotated_frame)
            
            # Check for quit
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        cap.release()
        cv2.destroyAllWindows()


def main():
    """Main function to run real-time stress detection."""
    # Paths - update these to match your setup
    stress_model_path = "/Users/RyoSeah/Desktop/Stress_Detection/training_results/models/best_gru_model.h5"
    movenet_model_path = "/Users/RyoSeah/Desktop/Stress_Detection/movenet_thunder.tflite"
    
    # Check if models exist
    if not os.path.exists(movenet_model_path):
        print(f"❌ MoveNet model not found at: {movenet_model_path}")
        print("Please download MoveNet model or update the path.")
        return
    
    if not os.path.exists(stress_model_path):
        print(f"❌ Stress model not found at: {stress_model_path}")
        print("Please train a stress detection model first using train.py")
        print("Using pose detection only...")
        
    try:
        # Initialize detector
        detector = RealTimeStressDetector(
            stress_model_path=stress_model_path,
            movenet_model_path=movenet_model_path,
            sequence_length=30,  # 1 second at 30fps
            confidence_threshold=0.3
        )
        
        # Run real-time detection
        detector.run_realtime_detection(camera_id=0)
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
