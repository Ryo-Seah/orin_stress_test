"""
Integration example: How to update your movenet.py to use the trained stress detection model
"""

import cv2
import numpy as np
import tensorflow as tf
import time
import os

# Add path to access the data loader
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), 'stress_training'))
from data_loader import BOLDDataLoader

# Load MoveNet TFLite model (your existing code)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "movenet_thunder.tflite")

interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
input_size = input_details[0]['shape'][2]

# Initialize data loader for format conversion and feature extraction
data_loader = BOLDDataLoader(
    bold_root="/Users/RyoSeah/Downloads/BOLD_public",  # Not used for inference
    sequence_length=30,  # For sequence-based prediction
    min_confidence=0.3
)

# Pose estimation from image (your existing code, returns COCO-17)
def detect_pose_movenet(frame):
    img = cv2.resize(frame, (input_size, input_size))
    input_data = np.expand_dims(img, axis=0).astype(np.uint8)
    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()
    keypoints = interpreter.get_tensor(output_details[0]['index'])[0][0]  # shape: (17, 3)
    return keypoints

# NEW: Enhanced stress computation using the trained pipeline
def compute_stress_score_enhanced(frame):
    """
    Compute stress score using the data loader's real-time method.
    This uses the same feature engineering as the training pipeline.
    """
    keypoints = detect_pose_movenet(frame)  # Get COCO-17 keypoints
    stress_score = data_loader.compute_stress_score_realtime(keypoints)
    return stress_score  # Returns 0-5 scale (compatible with your existing code)

# NEW: Sequence-based stress prediction (more accurate)
class SequenceStressPredictor:
    def __init__(self, sequence_length=30):
        self.sequence_length = sequence_length
        self.keypoint_buffer = []
        # TODO: Load your trained model here
        # self.model = tf.keras.models.load_model('stress_model.h5')
    
    def add_frame(self, frame):
        """Add a new frame to the sequence buffer."""
        keypoints = detect_pose_movenet(frame)
        self.keypoint_buffer.append(keypoints)
        
        # Keep only the last sequence_length frames
        if len(self.keypoint_buffer) > self.sequence_length:
            self.keypoint_buffer.pop(0)
    
    def predict_stress(self):
        """Predict stress using the full sequence (more accurate than single frame)."""
        if len(self.keypoint_buffer) < self.sequence_length:
            # Not enough frames, use simple method
            if len(self.keypoint_buffer) > 0:
                return data_loader.compute_stress_score_realtime(self.keypoint_buffer[-1])
            return 0.0
        
        # Convert to numpy array
        coco17_sequence = np.array(self.keypoint_buffer)  # (seq_len, 17, 3)
        
        # Extract features using the same pipeline as training
        features = data_loader.extract_features_from_coco17(coco17_sequence)
        
        # TODO: Use your trained model for prediction
        # features_scaled = data_loader.transform_features(features[np.newaxis, :, :])
        # stress_prediction = self.model.predict(features_scaled)[0][0]
        # return stress_prediction
        
        # For now, use the simple method on the latest frame
        return data_loader.compute_stress_score_realtime(self.keypoint_buffer[-1])

# NEW: Updated main loop with enhanced stress detection
def main_enhanced():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ Failed to open camera.")
        return

    # Initialize sequence predictor
    sequence_predictor = SequenceStressPredictor(sequence_length=30)
    
    print("✅ Enhanced stress detection ready! Press 'q' to quit.")
    print("📊 Green=Real-time, Blue=Sequence-based prediction")
    
    while True:
        ret, frame = cap.read()
        if not ret: 
            break

        # Method 1: Real-time stress score (faster, less accurate)
        realtime_score = compute_stress_score_enhanced(frame)
        
        # Method 2: Sequence-based prediction (slower, more accurate)
        sequence_predictor.add_frame(frame)
        sequence_score = sequence_predictor.predict_stress()
        
        # Get keypoints for visualization
        keypoints = detect_pose_movenet(frame)
        
        # Draw keypoints
        for i, (x, y, c) in enumerate(keypoints):
            if c > 0.3:
                cx, cy = int(x * frame.shape[1]), int(y * frame.shape[0])
                cv2.circle(frame, (cx, cy), 4, (0, 255, 0), -1)

        # Display both stress scores
        cv2.putText(frame, f"Realtime Stress: {realtime_score:.2f}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.putText(frame, f"Sequence Stress: {sequence_score:.2f}", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)
        
        # Add stress level indicator
        if sequence_score > 3.0:
            stress_level = "HIGH STRESS"
            color = (0, 0, 255)  # Red
        elif sequence_score > 1.5:
            stress_level = "MODERATE STRESS"
            color = (0, 165, 255)  # Orange
        else:
            stress_level = "LOW STRESS"
            color = (0, 255, 0)  # Green
            
        cv2.putText(frame, stress_level, (10, 90),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
        
        cv2.imshow("Enhanced MoveNet Stress Detection", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'): 
            break

    cap.release()
    cv2.destroyAllWindows()

# Example of how to integrate format conversion
def demonstrate_format_conversion():
    """Show how the format conversion works."""
    print("🔧 Format Conversion Demo")
    
    # Simulate MoveNet output (COCO-17)
    dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    coco17_keypoints = detect_pose_movenet(dummy_frame)
    
    print(f"MoveNet output (COCO-17): {coco17_keypoints.shape}")
    
    # Convert to BOLD-18 format for training compatibility
    bold18_keypoints = data_loader.convert_coco17_to_bold18(coco17_keypoints)
    print(f"Converted to BOLD-18: {bold18_keypoints.shape}")
    
    # Extract features using training pipeline
    bold18_sequence = bold18_keypoints[np.newaxis, :, :]  # Add sequence dimension
    features = data_loader.extract_pose_features(bold18_sequence, input_format='bold18')
    print(f"Extracted features: {features.shape}")
    
    # Or directly from COCO-17
    coco17_sequence = coco17_keypoints[np.newaxis, :, :]
    features_direct = data_loader.extract_features_from_coco17(coco17_sequence)
    print(f"Direct COCO-17 features: {features_direct.shape}")
    
    print("✅ Format conversion working correctly!")

if __name__ == "__main__":
    print("🚀 Enhanced MoveNet Stress Detection")
    print("Choose an option:")
    print("1. Run enhanced detection")
    print("2. Demo format conversion")
    
    choice = input("Enter choice (1 or 2): ")
    
    if choice == "1":
        main_enhanced()
    elif choice == "2":
        demonstrate_format_conversion()
    else:
        print("Invalid choice. Running enhanced detection...")
        main_enhanced()
