"""
Video Pose-based Stress Detection Module
Handles pose detection and stress analysis from video input.
"""

import cv2
import numpy as np
import tensorflow as tf
import os
import sys
from collections import deque
from typing import Optional


class VideoStressDetector:
    """Video-based stress detection using pose analysis."""
    
    def __init__(self, 
                 stress_model_path: str,
                 movenet_model_path: str,
                 sequence_length: int = 30,
                 confidence_threshold: float = 0.3):
        """
        Initialize video stress detector.
        
        Args:
            stress_model_path: Path to pose-based stress model
            movenet_model_path: Path to MoveNet model
            sequence_length: Video sequence length
            confidence_threshold: Pose confidence threshold
        """
        self.sequence_length = sequence_length
        self.confidence_threshold = confidence_threshold
        
        # Initialize buffers
        self.pose_buffer = deque(maxlen=sequence_length)
        self.video_stress_scores = deque(maxlen=50)
        
        # Load models
        self.load_movenet_model(movenet_model_path)
        self.load_pose_stress_model(stress_model_path)
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
        # Add the stress training directory to path (go up one level from working folder)
        current_dir = os.path.dirname(os.path.abspath(__file__))
        parent_dir = os.path.dirname(current_dir)
        stress_training_dir = os.path.join(parent_dir, 'stress_training')
        sys.path.append(stress_training_dir)
        
        try:
            from stress_training.data_processing import PoseFeatureExtractor
            self.feature_extractor = PoseFeatureExtractor(self.confidence_threshold)
            print("✅ Pose feature extractor setup successfully")
        except Exception as e:
            print(f"Warning: Could not setup pose feature extractor: {e}")
            self.feature_extractor = None
    
    def detect_pose(self, frame: np.ndarray) -> np.ndarray:
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
    
    def process_frame(self, frame: np.ndarray) -> np.ndarray:
        """Process a frame and return COCO format keypoints."""
        movenet_keypoints = self.detect_pose(frame)
        coco_keypoints = self.convert_movenet_to_coco_format(movenet_keypoints)
        self.pose_buffer.append(coco_keypoints)
        return coco_keypoints
    
    def predict_stress(self) -> Optional[float]:
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
            
            # Store stress score
            self.video_stress_scores.append(float(stress_score))
            
            return float(stress_score)
            
        except Exception as e:
            print(f"Video stress prediction error: {e}")
            return None
    
    def get_smoothed_stress(self) -> Optional[float]:
        """Get smoothed video stress score."""
        if len(self.video_stress_scores) == 0:
            return None
        
        # Use exponential weighted average
        weights = np.exp(np.linspace(-1, 0, len(self.video_stress_scores)))
        weighted_avg = np.average(list(self.video_stress_scores), weights=weights)
        return float(weighted_avg)
    
    def draw_keypoints(self, frame: np.ndarray, keypoints: np.ndarray) -> np.ndarray:
        """Draw pose keypoints on frame."""
        annotated_frame = frame.copy()
        h, w = frame.shape[:2]
        
        for i, (x, y, c) in enumerate(keypoints):
            if c > self.confidence_threshold:
                cx, cy = int(x * w), int(y * h)
                cv2.circle(annotated_frame, (cx, cy), 4, (0, 255, 0), -1)
        
        return annotated_frame
    
    def get_buffer_status(self) -> str:
        """Get buffer status string."""
        return f"Video: {len(self.pose_buffer)}/{self.sequence_length}"
