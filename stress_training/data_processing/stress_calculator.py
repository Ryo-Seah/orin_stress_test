"""
Handles stress score computation from BOLD annotations and real-time pose analysis.
"""
import numpy as np
from typing import Dict

class StressScoreCalculator:
    """Computes stress scores from various input sources."""
    
    def __init__(self):
        # Stress-related emotion indices for BOLD dataset (0-indexed)
        self.stress_emotions = [
            3,   # Anxiety (direct stress indicator)
            6,   # Confusion (cognitive stress)
            10,  # Disgust (negative emotion, stress-related)
            11,  # Doubt (uncertainty, stress-related)
            12,  # Empathic_pain (emotional stress)
            15,  # Fear (primary stress emotion)
            16,  # Horror (extreme fear, high stress)
            22,  # Sadness (depression-related stress)
        ]
        
        # Weights for combining different stress components
        self.vad_weight = 0.7
        self.categorical_weight = 0.3
    
    def compute_stress_score(self, annotations: Dict) -> float:
        """
        Compute stress score from BOLD annotations.
        
        Args:
            annotations: Dictionary containing valence, arousal, dominance, and categorical emotions
            
        Returns:
            Stress score between 0 and 1
        """
        valence = float(annotations['valence'])
        arousal = float(annotations['arousal'])  
        dominance = float(annotations['dominance'])
        categorical = annotations['categorical']  # comma-separated string of 26 values
        
        # Compute VAD-based stress
        vad_stress = self._compute_vad_stress(valence, arousal, dominance)
        
        # Compute categorical emotion-based stress
        categorical_stress = self._compute_categorical_stress(categorical)
        
        # Combined stress score
        if categorical_stress == 0.0:
            stress_score = vad_stress
        else:
            stress_score = self.vad_weight * vad_stress + self.categorical_weight * categorical_stress
        
        return np.clip(stress_score, 0.0, 1.0)
    
    def compute_stress_score_realtime(self, coco17_keypoints: np.ndarray) -> float:
        """
        Compute real-time stress score from pose keypoints alone.
        
        Args:
            coco17_keypoints: COCO-17 format keypoints
            
        Returns:
            Estimated stress score between 0 and 1
        """
        # Import here to avoid circular dependency
        from .feature_extractor import PoseFeatureExtractor
        from .pose_converter import PoseFormatConverter
        
        # Convert to BOLD-18 and extract features
        converter = PoseFormatConverter()
        feature_extractor = PoseFeatureExtractor()
        
        bold_keypoints = converter.convert_coco17_to_bold18(coco17_keypoints)
        
        # Add sequence dimension if single frame
        if bold_keypoints.ndim == 2:
            bold_keypoints = bold_keypoints[np.newaxis, :]
        
        # Extract postural stress indicators
        stress_indicators = []
        
        for frame_kp in bold_keypoints:
            # Head-shoulder misalignment
            if all(frame_kp[i, 2] > 0.3 for i in [0, 2, 5]):
                nose, r_shoulder, l_shoulder = frame_kp[[0, 2, 5]]
                shoulder_center = (r_shoulder[:2] + l_shoulder[:2]) / 2
                head_misalignment = np.linalg.norm(nose[:2] - shoulder_center) / 100.0
                stress_indicators.append(min(head_misalignment, 1.0))
            
            # Shoulder tension (asymmetry)
            if all(frame_kp[i, 2] > 0.3 for i in [2, 5]):
                shoulder_asymmetry = abs(frame_kp[2, 1] - frame_kp[5, 1]) / 50.0
                stress_indicators.append(min(shoulder_asymmetry, 1.0))
            
            # Arm tension (distance from body)
            if all(frame_kp[i, 2] > 0.3 for i in [1, 4, 7]):
                neck, r_wrist, l_wrist = frame_kp[[1, 4, 7]]
                avg_arm_distance = (np.linalg.norm(r_wrist[:2] - neck[:2]) + 
                                  np.linalg.norm(l_wrist[:2] - neck[:2])) / 2
                arm_tension = min(avg_arm_distance / 150.0, 1.0)
                stress_indicators.append(arm_tension)
        
        # Return average stress indicators
        return np.mean(stress_indicators) if stress_indicators else 0.0
    
    def _compute_vad_stress(self, valence: float, arousal: float, dominance: float) -> float:
        """Compute stress from VAD (Valence-Arousal-Dominance) values."""
        # Normalize VAD values to 0-1 range
        valence_norm = valence / 10.0 if valence > 1.0 else valence
        arousal_norm = arousal / 10.0 if arousal > 1.0 else arousal  
        dominance_norm = dominance / 10.0 if dominance > 1.0 else dominance
        
        # High arousal + Low valence + Low dominance = Higher stress
        vad_stress = (arousal_norm + (1 - valence_norm) + (1 - dominance_norm)) / 3.0
        
        return vad_stress
    
    def _compute_categorical_stress(self, categorical: str) -> float:
        """Compute stress from categorical emotions."""
        try:
            categorical_values = [float(x) for x in categorical.split(',')]
            stress_emotion_sum = sum(categorical_values[i] for i in self.stress_emotions 
                                   if i < len(categorical_values))
            categorical_stress = min(stress_emotion_sum / len(self.stress_emotions), 1.0)
            return categorical_stress
        except:
            return 0.0  # Fallback if parsing fails