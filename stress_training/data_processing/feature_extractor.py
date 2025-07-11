"""
Handles pose feature extraction and engineering.
"""
import numpy as np
from typing import List, Tuple

class PoseFeatureExtractor:
    """Extracts engineered features from pose keypoints."""
    
    def __init__(self, min_confidence: float = 0.3):
        self.min_confidence = min_confidence
        
        # Important joint pairs for distance features
        self.joint_pairs = [
            (1, 2), (1, 5),    # Neck to shoulders
            (2, 3), (3, 4),    # Right arm
            (5, 6), (6, 7),    # Left arm
            (8, 9), (9, 10),   # Right leg
            (11, 12), (12, 13), # Left leg
            (2, 5),            # Shoulder width
            (8, 11),           # Hip width
            (1, 8), (1, 11),   # Torso connections
        ]
    
    def extract_pose_features(self, keypoints: np.ndarray, input_format: str = 'bold18') -> np.ndarray:
        """
        Extract engineered features from raw pose keypoints.
        
        Args:
            keypoints: Array of shape (sequence_length, num_joints, 3) - (x, y, confidence)
            input_format: Either 'bold18' or 'coco17' to specify input format
            
        Returns:
            Feature array of shape (sequence_length, num_features)
        """
        if input_format == 'coco17':
            # Convert to BOLD-18 format for consistent processing
            from .pose_converter import PoseFormatConverter
            converter = PoseFormatConverter(self.min_confidence)
            keypoints = converter.convert_coco17_to_bold18(keypoints)
        elif input_format != 'bold18':
            raise ValueError(f"Unsupported input format: {input_format}. Use 'bold18' or 'coco17'")
        
        seq_len, num_joints, _ = keypoints.shape
        if num_joints != 18:
            raise ValueError(f"Expected 18 joints after conversion, got {num_joints}")
        
        features_list = []
        
        for frame_idx in range(seq_len):
            frame_kp = keypoints[frame_idx]  # (18, 3)
            frame_features = []
            
            # 1. Normalized coordinates (relative to torso)
            normalized_coords = self._extract_normalized_coordinates(frame_kp)
            frame_features.extend(normalized_coords)
            
            # 2. Joint distances
            distances = self._extract_distance_features(frame_kp)
            frame_features.extend(distances)
            
            # 3. Joint angles
            angles = self._extract_angle_features(frame_kp)
            frame_features.extend(angles)
            
            # 4. Postural features
            postural = self._extract_postural_features(frame_kp)
            frame_features.extend(postural)
            
            features_list.append(frame_features)
        
        return np.array(features_list)
    
    def compute_velocity_features(self, features: np.ndarray) -> np.ndarray:
        """
        Compute velocity features from consecutive frames.
        
        Args:
            features: Feature array of shape (sequence_length, num_features)
            
        Returns:
            Velocity features of shape (sequence_length-1, num_features)
        """
        if len(features) < 2:
            return np.zeros((0, features.shape[1]))
        
        # Compute frame-to-frame differences
        velocities = np.diff(features, axis=0)
        return velocities
    
    def _extract_normalized_coordinates(self, frame_kp: np.ndarray) -> List[float]:
        """Extract normalized coordinates relative to torso."""
        normalized_coords = []
        
        # Use neck and hip center as torso reference
        if frame_kp[1, 2] > self.min_confidence and frame_kp[8, 2] > self.min_confidence:  # Neck and RHip
            torso_center = (frame_kp[1, :2] + frame_kp[8, :2]) / 2
            torso_length = np.linalg.norm(frame_kp[1, :2] - frame_kp[8, :2])
            
            for joint_idx in range(18):
                if frame_kp[joint_idx, 2] > self.min_confidence:
                    norm_coord = (frame_kp[joint_idx, :2] - torso_center) / (torso_length + 1e-6)
                    normalized_coords.extend(norm_coord)
                else:
                    normalized_coords.extend([0.0, 0.0])
        else:
            normalized_coords.extend([0.0] * 36)  # 18 joints * 2 coordinates
        
        return normalized_coords
    
    def _extract_distance_features(self, frame_kp: np.ndarray) -> List[float]:
        """Extract distance features between joint pairs."""
        distances = []
        
        for joint1_idx, joint2_idx in self.joint_pairs:
            if (frame_kp[joint1_idx, 2] > self.min_confidence and 
                frame_kp[joint2_idx, 2] > self.min_confidence):
                distance = np.linalg.norm(frame_kp[joint1_idx, :2] - frame_kp[joint2_idx, :2])
                distances.append(distance)
            else:
                distances.append(0.0)
        
        return distances
    
    def _extract_angle_features(self, frame_kp: np.ndarray) -> List[float]:
        """Extract joint angle features."""
        angles = []
        
        # Shoulder angle (left shoulder - neck - right shoulder)
        if all(frame_kp[i, 2] > self.min_confidence for i in [1, 2, 5]):  # Neck, RShoulder, LShoulder
            v1 = frame_kp[2, :2] - frame_kp[1, :2]  # Neck to RShoulder
            v2 = frame_kp[5, :2] - frame_kp[1, :2]  # Neck to LShoulder
            angle = np.arccos(np.clip(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-6), -1, 1))
            angles.append(angle)
        else:
            angles.append(0.0)
        
        # Hip angle
        if all(frame_kp[i, 2] > self.min_confidence for i in [1, 8, 11]):  # Neck, RHip, LHip
            v1 = frame_kp[8, :2] - frame_kp[1, :2]   # Neck to RHip
            v2 = frame_kp[11, :2] - frame_kp[1, :2]  # Neck to LHip
            angle = np.arccos(np.clip(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-6), -1, 1))
            angles.append(angle)
        else:
            angles.append(0.0)
        
        return angles
    
    def _extract_postural_features(self, frame_kp: np.ndarray) -> List[float]:
        """Extract postural features like head tilt and slouch."""
        postural = []
        
        # Head tilt and slouch metrics
        if all(frame_kp[i, 2] > self.min_confidence for i in [0, 2, 5, 8, 11]):
            nose, r_shoulder, l_shoulder, r_hip, l_hip = frame_kp[[0, 2, 5, 8, 11]]
            
            # Head-shoulder alignment
            head_center = nose[:2]
            shoulder_center = (r_shoulder[:2] + l_shoulder[:2]) / 2
            hip_center = (r_hip[:2] + l_hip[:2]) / 2
            
            # Head tilt (horizontal deviation from shoulder center)
            head_tilt = abs(head_center[0] - shoulder_center[0])
            
            # Forward head posture (vertical position relative to shoulders)
            forward_head = max(0, shoulder_center[1] - head_center[1])
            
            # Shoulder slouch (vertical drop from ideal)
            shoulder_slope = abs(r_shoulder[1] - l_shoulder[1])
            
            # Torso alignment
            torso_lean = abs(shoulder_center[0] - hip_center[0])
            
            postural.extend([head_tilt, forward_head, shoulder_slope, torso_lean])
        else:
            postural.extend([0.0, 0.0, 0.0, 0.0])
        
        return postural