"""
Handles conversion between different pose keypoint formats (BOLD-18, COCO-17).
"""
import numpy as np
from typing import Dict, Optional

class PoseFormatConverter:
    """Converts between BOLD-18 and COCO-17 pose formats."""
    
    def __init__(self, min_confidence: float = 0.3):
        self.min_confidence = min_confidence
        
        # BOLD dataset uses 18 keypoints
        self.bold_joint_names = [
            "Nose", "Neck", "RShoulder", "RElbow", "RWrist", "LShoulder", 
            "LElbow", "LWrist", "RHip", "RKnee", "RAnkle", "LHip", 
            "LKnee", "LAnkle", "REye", "LEye", "REar", "LEar"
        ]
        
        # Standard COCO-17 format
        self.coco17_joint_names = [
            "nose", "left_eye", "right_eye", "left_ear", "right_ear",
            "left_shoulder", "right_shoulder", "left_elbow", "right_elbow", 
            "left_wrist", "right_wrist", "left_hip", "right_hip",
            "left_knee", "right_knee", "left_ankle", "right_ankle"
        ]
        
        # Mapping from BOLD-18 indices to COCO-17 indices
        self.bold_to_coco17_mapping = {
            0: 0,   # Nose -> nose
            1: None, # Neck -> None (not in COCO-17, will be estimated)
            2: 6,   # RShoulder -> right_shoulder
            3: 8,   # RElbow -> right_elbow
            4: 10,  # RWrist -> right_wrist
            5: 5,   # LShoulder -> left_shoulder
            6: 7,   # LElbow -> left_elbow
            7: 9,   # LWrist -> left_wrist
            8: 12,  # RHip -> right_hip
            9: 14,  # RKnee -> right_knee
            10: 16, # RAnkle -> right_ankle
            11: 11, # LHip -> left_hip
            12: 13, # LKnee -> left_knee
            13: 15, # LAnkle -> left_ankle
            14: 2,  # REye -> right_eye
            15: 1,  # LEye -> left_eye
            16: 4,  # REar -> right_ear
            17: 3,  # LEar -> left_ear
        }
        
        # Mapping from COCO-17 indices to BOLD-18 indices (inverse mapping)
        self.coco17_to_bold_mapping = {
            0: 0,   # nose -> Nose
            1: 15,  # left_eye -> LEye
            2: 14,  # right_eye -> REye
            3: 17,  # left_ear -> LEar
            4: 16,  # right_ear -> REar
            5: 5,   # left_shoulder -> LShoulder
            6: 2,   # right_shoulder -> RShoulder
            7: 6,   # left_elbow -> LElbow
            8: 3,   # right_elbow -> RElbow
            9: 7,   # left_wrist -> LWrist
            10: 4,  # right_wrist -> RWrist
            11: 11, # left_hip -> LHip
            12: 8,  # right_hip -> RHip
            13: 12, # left_knee -> LKnee
            14: 9,  # right_knee -> RKnee
            15: 13, # left_ankle -> LAnkle
            16: 10, # right_ankle -> RAnkle
        }
    
    def convert_bold18_to_coco17(self, bold_keypoints: np.ndarray) -> np.ndarray:
        """
        Convert BOLD 18-keypoint format to COCO-17 format.
        
        Args:
            bold_keypoints: Shape (seq_len, 18, 3) or (18, 3)
            
        Returns:
            coco17_keypoints: Shape (seq_len, 17, 3) or (17, 3)
        """
        # Handle both single frame and sequence inputs
        squeeze_output = False
        if bold_keypoints.ndim == 2:
            bold_keypoints = bold_keypoints[np.newaxis, :]
            squeeze_output = True
        
        seq_len = bold_keypoints.shape[0]
        coco17_keypoints = np.zeros((seq_len, 17, 3))
        
        # Map BOLD-18 keypoints to COCO-17 positions
        for bold_idx, coco17_idx in self.bold_to_coco17_mapping.items():
            if coco17_idx is not None:
                coco17_keypoints[:, coco17_idx] = bold_keypoints[:, bold_idx]
        
        if squeeze_output:
            return coco17_keypoints[0]
        return coco17_keypoints
    
    def convert_coco17_to_bold18(self, coco17_keypoints: np.ndarray) -> np.ndarray:
        """
        Convert COCO-17 format to BOLD 18-keypoint format with estimated neck.
        
        Args:
            coco17_keypoints: Shape (seq_len, 17, 3) or (17, 3)
            
        Returns:
            bold_keypoints: Shape (seq_len, 18, 3) or (18, 3)
        """
        # Handle both single frame and sequence inputs
        squeeze_output = False
        if coco17_keypoints.ndim == 2:
            coco17_keypoints = coco17_keypoints[np.newaxis, :]
            squeeze_output = True
        
        seq_len = coco17_keypoints.shape[0]
        bold_keypoints = np.zeros((seq_len, 18, 3))
        
        # Map COCO-17 keypoints to BOLD-18 positions
        for coco17_idx, bold_idx in self.coco17_to_bold_mapping.items():
            bold_keypoints[:, bold_idx] = coco17_keypoints[:, coco17_idx]
        
        # Estimate neck position (BOLD index 1) from shoulders
        for i in range(seq_len):
            left_shoulder = coco17_keypoints[i, 5]   # COCO-17 left_shoulder
            right_shoulder = coco17_keypoints[i, 6]  # COCO-17 right_shoulder
            
            # Only estimate neck if both shoulders are confident
            if left_shoulder[2] > self.min_confidence and right_shoulder[2] > self.min_confidence:
                neck_x = (left_shoulder[0] + right_shoulder[0]) / 2
                neck_y = (left_shoulder[1] + right_shoulder[1]) / 2
                neck_conf = min(left_shoulder[2], right_shoulder[2])
                bold_keypoints[i, 1] = [neck_x, neck_y, neck_conf]
        
        if squeeze_output:
            return bold_keypoints[0]
        return bold_keypoints
    
    def get_format_info(self) -> Dict:
        """Get information about the keypoint formats."""
        return {
            'bold18_joints': self.bold_joint_names,
            'coco17_joints': self.coco17_joint_names,
            'bold_to_coco17_mapping': self.bold_to_coco17_mapping,
            'coco17_to_bold_mapping': self.coco17_to_bold_mapping
        }