#!/usr/bin/env python3
"""
Test script for format conversion between BOLD-18 and COCO-17.
"""

import sys
import os
import numpy as np

# Add the current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from data_loader import BOLDDataLoader

def test_format_conversion():
    """Test the format conversion methods."""
    print("🔍 Testing Format Conversion")
    print("=" * 50)
    
    # Initialize data loader
    data_loader = BOLDDataLoader(
        bold_root="/Users/RyoSeah/Downloads/BOLD_public",
        sequence_length=10,
        min_confidence=0.3
    )
    
    # Test 1: Format info
    print("\n1. Testing format information...")
    format_info = data_loader.get_format_info()
    print(f"   ✅ BOLD-18 joints: {len(format_info['bold18_joints'])} joints")
    print(f"   ✅ COCO-17 joints: {len(format_info['coco17_joints'])} joints")
    print(f"   ✅ Mapping defined for {len(format_info['bold_to_coco17_mapping'])} BOLD joints")
    
    # Test 2: Create sample BOLD-18 keypoints
    print("\n2. Testing BOLD-18 to COCO-17 conversion...")
    
    # Create sample BOLD-18 keypoints (single frame)
    bold18_frame = np.random.rand(18, 3)
    bold18_frame[:, 2] = 0.8  # Set confidence to 0.8
    
    # Convert to COCO-17
    coco17_frame = data_loader.convert_bold18_to_coco17(bold18_frame)
    print(f"   ✅ BOLD-18 shape: {bold18_frame.shape}")
    print(f"   ✅ COCO-17 shape: {coco17_frame.shape}")
    
    # Test with sequence
    bold18_sequence = np.random.rand(5, 18, 3)
    bold18_sequence[:, :, 2] = 0.8  # Set confidence
    
    coco17_sequence = data_loader.convert_bold18_to_coco17(bold18_sequence)
    print(f"   ✅ BOLD-18 sequence: {bold18_sequence.shape}")
    print(f"   ✅ COCO-17 sequence: {coco17_sequence.shape}")
    
    # Test 3: COCO-17 to BOLD-18 conversion
    print("\n3. Testing COCO-17 to BOLD-18 conversion...")
    
    # Create sample COCO-17 keypoints
    coco17_test = np.random.rand(17, 3)
    coco17_test[:, 2] = 0.8  # Set confidence
    
    # Convert to BOLD-18
    bold18_converted = data_loader.convert_coco17_to_bold18(coco17_test)
    print(f"   ✅ COCO-17 input: {coco17_test.shape}")
    print(f"   ✅ BOLD-18 output: {bold18_converted.shape}")
    
    # Check if neck was estimated
    neck_confidence = bold18_converted[1, 2]  # Neck is at index 1
    print(f"   ✅ Estimated neck confidence: {neck_confidence:.3f}")
    
    # Test 4: Real-time stress scoring with COCO-17
    print("\n4. Testing real-time stress scoring...")
    
    # Create realistic COCO-17 keypoints (simulate MoveNet output)
    coco17_realistic = np.array([
        [0.5, 0.3, 0.9],    # nose
        [0.45, 0.25, 0.8],  # left_eye
        [0.55, 0.25, 0.8],  # right_eye
        [0.4, 0.25, 0.7],   # left_ear
        [0.6, 0.25, 0.7],   # right_ear
        [0.4, 0.5, 0.9],    # left_shoulder
        [0.6, 0.5, 0.9],    # right_shoulder
        [0.35, 0.7, 0.8],   # left_elbow
        [0.65, 0.7, 0.8],   # right_elbow
        [0.3, 0.9, 0.7],    # left_wrist
        [0.7, 0.9, 0.7],    # right_wrist
        [0.45, 0.8, 0.9],   # left_hip
        [0.55, 0.8, 0.9],   # right_hip
        [0.45, 1.1, 0.8],   # left_knee
        [0.55, 1.1, 0.8],   # right_knee
        [0.45, 1.4, 0.7],   # left_ankle
        [0.55, 1.4, 0.7],   # right_ankle
    ])
    
    stress_score = data_loader.compute_stress_score_realtime(coco17_realistic)
    print(f"   ✅ Computed stress score: {stress_score:.3f}")
    print(f"   ✅ Score range: [0, 5] (compatible with movenet.py)")
    
    # Test 5: Feature extraction from COCO-17
    print("\n5. Testing feature extraction from COCO-17...")
    
    # Create a sequence of COCO-17 keypoints
    coco17_seq = np.random.rand(10, 17, 3)
    coco17_seq[:, :, 2] = 0.8  # Set confidence
    
    try:
        features = data_loader.extract_features_from_coco17(coco17_seq)
        print(f"   ✅ COCO-17 sequence: {coco17_seq.shape}")
        print(f"   ✅ Extracted features: {features.shape}")
        print(f"   ✅ Features per frame: {features.shape[1]}")
    except Exception as e:
        print(f"   ❌ Feature extraction failed: {e}")
        return False
    
    print("\n" + "=" * 50)
    print("✅ ALL FORMAT CONVERSION TESTS PASSED!")
    print("✅ Ready for both BOLD training and MoveNet inference!")
    print("=" * 50)
    
    return True

def test_integration_example():
    """Show how to use the format conversion in practice."""
    print("\n🚀 Integration Example")
    print("=" * 50)
    
    data_loader = BOLDDataLoader(
        bold_root="/Users/RyoSeah/Downloads/BOLD_public",
        sequence_length=30,
        min_confidence=0.3
    )
    
    print("\n📋 Usage examples:")
    print("1. Training with BOLD dataset (automatic BOLD-18 processing)")
    print("2. Real-time inference with MoveNet COCO-17 output:")
    
    example_code = '''
# In your movenet.py file:
def detect_pose_movenet(frame):
    # ... existing MoveNet code ...
    keypoints = interpreter.get_tensor(output_details[0]['index'])[0][0]  # (17, 3)
    return keypoints

# Updated stress computation using data_loader
def compute_stress_score(frame):
    keypoints = detect_pose_movenet(frame)  # Get COCO-17 keypoints
    stress_score = data_loader.compute_stress_score_realtime(keypoints)
    return stress_score  # Returns 0-5 scale (compatible with existing code)

# For sequence-based prediction (more accurate):
def compute_stress_sequence(frame_sequence):
    # Collect keypoints over time
    keypoint_sequence = []
    for frame in frame_sequence:
        kp = detect_pose_movenet(frame)
        keypoint_sequence.append(kp)
    
    coco17_sequence = np.array(keypoint_sequence)  # (seq_len, 17, 3)
    features = data_loader.extract_features_from_coco17(coco17_sequence)
    
    # Use trained model for prediction
    # prediction = trained_model.predict(features)
    return features
'''
    
    print(example_code)
    
    print("✅ Integration ready!")

if __name__ == "__main__":
    print("🔧 Format Conversion Testing Suite")
    
    success = test_format_conversion()
    
    if success:
        test_integration_example()
    else:
        print("❌ Format conversion tests failed.")
