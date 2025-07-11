"""
Handles BOLD dataset loading and preprocessing for training.
"""
import numpy as np
import pandas as pd
import os
from typing import Tuple, Dict
from sklearn.preprocessing import StandardScaler
import tensorflow as tf

from .pose_converter import PoseFormatConverter
from .feature_extractor import PoseFeatureExtractor
from .stress_calculator import StressScoreCalculator

class BOLDDataset:
    """Main class for loading and preprocessing BOLD dataset for stress detection training."""
    
    def __init__(self, 
                 bold_root: str,
                 sequence_length: int = 30,
                 overlap_ratio: float = 0.5,
                 min_confidence: float = 0.3):
        """
        Initialize BOLD dataset loader.
        
        Args:
            bold_root: Path to BOLD dataset root directory
            sequence_length: Fixed length for all sequences
            overlap_ratio: Overlap ratio for sliding windows (not used in segment mode)
            min_confidence: Minimum confidence threshold for keypoints
        """
        self.bold_root = bold_root
        self.sequence_length = sequence_length
        self.overlap_ratio = overlap_ratio
        self.min_confidence = min_confidence
        
        # Initialize components
        self.pose_converter = PoseFormatConverter(min_confidence)
        self.feature_extractor = PoseFeatureExtractor(min_confidence)
        self.stress_calculator = StressScoreCalculator()
        self.scaler = StandardScaler()
        self._scaler_fitted = False
    
    def load_dataset(self, split: str) -> Tuple[np.ndarray, np.ndarray]:
        """
        Load dataset for segment-level stress prediction according to BOLD format.
        
        Args:
            split: Dataset split ('train', 'val', 'test')
            
        Returns:
            X: (num_segments, sequence_length, num_features)
            y: (num_segments,) - one stress score per segment
        """
        # Look for CSV files in annotations subdirectory
        annotations_file = os.path.join(self.bold_root, "annotations", f"{split}.csv")
        
        if not os.path.exists(annotations_file):
            print(f"❌ Annotations file {annotations_file} not found")
            return np.array([]), np.array([])
        
        # Read CSV without header since BOLD format doesn't have headers
        annotations_df = pd.read_csv(annotations_file, header=None)
        print(f"✅ Found {split} annotations with {len(annotations_df)} rows and {len(annotations_df.columns)} columns")
        
        # BOLD CSV format: video_file, person_id, start_frame, end_frame, categorical_emotions(26), valence, arousal, dominance, gender, age, ethnicity
        column_names = ['video_file', 'person_id', 'start_frame', 'end_frame', 'categorical', 'valence', 'arousal', 'dominance', 'gender', 'age', 'ethnicity']
        
        # Handle variable number of columns
        while len(column_names) < len(annotations_df.columns):
            column_names.append(f'extra_{len(column_names)}')
        
        annotations_df.columns = column_names[:len(annotations_df.columns)]
        
        print(f"🔍 Sample video file: {annotations_df['video_file'].iloc[0] if len(annotations_df) > 0 else 'None'}")
        
        X_segments = []
        y_segments = []
        processed_count = 0
        
        for idx, annotation in annotations_df.iterrows():
            try:
                # BOLD structure: video_file is like "003/video_name.mp4/segment_id.mp4"
                video_path = annotation['video_file']
                person_id = int(annotation['person_id'])
                start_frame = int(annotation['start_frame'])
                end_frame = int(annotation['end_frame'])
                
                # Skip invalid frame ranges
                if end_frame <= start_frame:
                    continue
                
                # Convert video path to joints path: remove the segment .mp4 extension
                # "003/video_name.mp4/segment_id.mp4" -> "003/video_name.mp4/segment_id.npy"
                # Split at the last '/' to separate video and segment
                video_dir, segment_file = video_path.rsplit('/', 1)
                segment_id = segment_file.replace('.mp4', '.npy')
                joints_file = os.path.join(self.bold_root, "joints", video_dir, segment_id)
                
                if not os.path.exists(joints_file):
                    if idx < 5:  # Only print first few warnings
                        print(f"⚠️  Joints file not found: {joints_file}")
                    continue
                
                # Load joint data: (N, 56) where N is number of frames
                joints_data = np.load(joints_file)
                
                # Debug: print available person_ids and frame numbers
                if idx < 5:
                    print(f"[DEBUG] {joints_file}: available person_ids = {np.unique(joints_data[:,1])}, requested = {person_id}")
                    print(f"[DEBUG] {joints_file}: available frames = {np.min(joints_data[:,0])} to {np.max(joints_data[:,0])}, requested = {start_frame} to {end_frame}")
                
                if len(joints_data) == 0:
                    continue
                
                # Filter by person_id (column 1) and extract pose data
                person_mask = joints_data[:, 1] == person_id
                person_joints = joints_data[person_mask]
                
                if len(person_joints) == 0:
                    if idx < 5:
                        print(f"⚠️  No joints found for person {person_id} in {video_path}")
                    continue
                
                # Get frame numbers (column 0) and pose data (columns 2-55, reshape to 18x3)
                frame_numbers = person_joints[:, 0].astype(int)
                pose_data = person_joints[:, 2:56].reshape(-1, 18, 3)  # 54 values -> 18 joints x 3 (x,y,conf)

                # Use all available pose data for this person/segment
                if len(pose_data) == 0:
                    continue
                segment_poses = pose_data

                # Normalize sequence length
                segment_poses = self.normalize_sequence_length(segment_poses)

                # Extract features
                segment_features = self.feature_extractor.extract_pose_features(
                    segment_poses, input_format='bold18'
                )

                # Compute stress score from annotations
                annotation_dict = {
                    'categorical': annotation['categorical'],
                    'valence': annotation['valence'],
                    'arousal': annotation['arousal'],
                    'dominance': annotation['dominance']
                }
                stress_score = self.stress_calculator.compute_stress_score(annotation_dict)

                X_segments.append(segment_features)
                y_segments.append(stress_score)
                processed_count += 1

                if processed_count % 50 == 0:
                    print(f"✅ Processed {processed_count} valid segments...")
                
            except Exception as e:
                if idx < 5:  # Only print first few errors
                    print(f"❌ Error processing row {idx}: {e}")
                continue
        
        print(f"🎉 Successfully loaded {len(X_segments)} segments from {split} split")
        if len(X_segments) == 0:
            print("⚠️  No valid segments found. Check if:")
            print("   - Joint files exist in the joints directory")
            print("   - Person IDs in CSV match those in joint files")
            print("   - Frame ranges are valid")
        
        return np.array(X_segments), np.array(y_segments)
    
    def normalize_sequence_length(self, segment_poses: np.ndarray) -> np.ndarray:
        """
        Pad or truncate segment to fixed sequence length.
        
        Args:
            segment_poses: Array of shape (num_frames, 18, 3)
            
        Returns:
            Normalized segment of shape (sequence_length, 18, 3)
        """
        if len(segment_poses) < self.sequence_length:
            # Pad with last frame
            last_frame = segment_poses[-1:] if len(segment_poses) > 0 else np.zeros((1, 18, 3))
            padding_needed = self.sequence_length - len(segment_poses)
            padding = np.repeat(last_frame, padding_needed, axis=0)
            return np.concatenate([segment_poses, padding], axis=0)
        else:
            # Truncate to sequence length
            return segment_poses[:self.sequence_length]
    
    def fit_scaler(self, X: np.ndarray):
        """
        Fit the feature scaler on training data.
        
        Args:
            X: Training features of shape (num_samples, sequence_length, num_features)
        """
        if len(X) == 0:
            print("Warning: Empty training data for scaler fitting")
            return
        
        # Reshape to (num_samples * sequence_length, num_features) for fitting
        X_reshaped = X.reshape(-1, X.shape[-1])
        self.scaler.fit(X_reshaped)
        self._scaler_fitted = True
        print(f"Fitted scaler on {X_reshaped.shape[0]} feature vectors")
    
    def transform_features(self, X: np.ndarray) -> np.ndarray:
        """
        Transform features using fitted scaler.
        
        Args:
            X: Features of shape (num_samples, sequence_length, num_features)
            
        Returns:
            Scaled features of same shape
        """
        if not self._scaler_fitted:
            raise ValueError("Scaler not fitted. Call fit_scaler() first.")
        
        if len(X) == 0:
            return X
        
        original_shape = X.shape
        X_reshaped = X.reshape(-1, X.shape[-1])
        X_scaled = self.scaler.transform(X_reshaped)
        return X_scaled.reshape(original_shape)
    
    # ...existing code...

# Add these missing methods to the end of the BOLDDataset class:

    def compute_velocity_features(self, features: np.ndarray) -> np.ndarray:
        """
        Compute velocity features from consecutive frames.
        
        Args:
            features: Feature array of shape (sequence_length, num_features)
            
        Returns:
            Velocity features of shape (sequence_length-1, num_features)
        """
        return self.feature_extractor.compute_velocity_features(features)
    
    def extract_features_from_coco17(self, coco17_sequence: np.ndarray) -> np.ndarray:
        """
        Extract features from a sequence of COCO-17 keypoints.
        
        Args:
            coco17_sequence: Shape (sequence_length, 17, 3)
            
        Returns:
            Feature array of shape (sequence_length, num_features)
        """
        return self.feature_extractor.extract_pose_features(
            coco17_sequence, input_format='coco17'
        )
    
    def compute_stress_score_realtime(self, coco17_keypoints: np.ndarray) -> float:
        """
        Compute real-time stress score from pose keypoints.
        
        Args:
            coco17_keypoints: COCO-17 format keypoints
            
        Returns:
            Estimated stress score between 0 and 5
        """
        return self.stress_calculator.compute_stress_score_realtime(coco17_keypoints)
    
    def convert_bold18_to_coco17(self, bold_keypoints: np.ndarray) -> np.ndarray:
        """Convert BOLD-18 to COCO-17 format."""
        return self.pose_converter.convert_bold18_to_coco17(bold_keypoints)
    
    def convert_coco17_to_bold18(self, coco17_keypoints: np.ndarray) -> np.ndarray:
        """Convert COCO-17 to BOLD-18 format."""
        return self.pose_converter.convert_coco17_to_bold18(coco17_keypoints)
    
    # Add these properties for backward compatibility
    @property
    def joint_names(self):
        """Backward compatibility for joint names."""
        return self.pose_converter.bold_joint_names
    
    @property
    def bold_joint_names(self):
        """BOLD joint names."""
        return self.pose_converter.bold_joint_names
    
    @property
    def coco17_joint_names(self):
        """COCO-17 joint names."""
        return self.pose_converter.coco17_joint_names
    
    def create_tf_dataset(self, X: np.ndarray, y: np.ndarray, 
                         batch_size: int, shuffle: bool = True) -> tf.data.Dataset:
        """
        Create TensorFlow dataset for training.
        
        Args:
            X: Features array
            y: Labels array
            batch_size: Batch size
            shuffle: Whether to shuffle data
            
        Returns:
            TensorFlow dataset
        """
        if len(X) == 0 or len(y) == 0:
            # Return empty dataset
            return tf.data.Dataset.from_tensor_slices((
                tf.zeros((0, self.sequence_length, X.shape[-1] if len(X.shape) > 2 else 1)),
                tf.zeros((0,))
            )).batch(batch_size)
        
        dataset = tf.data.Dataset.from_tensor_slices((X.astype(np.float32), y.astype(np.float32)))
        
        if shuffle:
            dataset = dataset.shuffle(buffer_size=min(1000, len(X)))
        
        dataset = dataset.batch(batch_size)
        dataset = dataset.prefetch(tf.data.AUTOTUNE)
        
        return dataset
    
    def create_training_dataset(self, train_split: str = 'train', 
                               val_split: str = 'val',
                               test_split: str = 'test') -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
        """
        Create complete training dataset with train/validation/test splits.
        
        Args:
            train_split: Name of training split
            val_split: Name of validation split  
            test_split: Name of test split
            
        Returns:
            Dictionary containing datasets for each split
        """
        datasets = {}
        
        for split_name in [train_split, val_split, test_split]:
            try:
                X, y = self.load_dataset(split_name)
                print(f"Loaded {split_name} split: {X.shape[0]} segments")
                datasets[split_name] = (X, y)
            except Exception as e:
                print(f"Warning: Could not load {split_name} split: {e}")
                datasets[split_name] = (np.array([]), np.array([]))
        
        return datasets
    
    def get_data_stats(self, X: np.ndarray, y: np.ndarray) -> Dict:
        """
        Get statistics about the loaded dataset.
        
        Args:
            X: Feature array
            y: Stress scores
            
        Returns:
            Dictionary with dataset statistics
        """
        if len(X) == 0 or len(y) == 0:
            return {
                'num_segments': 0,
                'sequence_length': self.sequence_length,
                'num_features': 0,
                'stress_score_min': 0,
                'stress_score_max': 0,
                'stress_score_mean': 0,
                'stress_score_std': 0
            }
        
        return {
            'num_segments': X.shape[0],
            'sequence_length': X.shape[1],
            'num_features': X.shape[2],
            'stress_score_min': float(np.min(y)),
            'stress_score_max': float(np.max(y)),
            'stress_score_mean': float(np.mean(y)),
            'stress_score_std': float(np.std(y))
        }
    
    def get_format_info(self) -> Dict:
        """Get information about the keypoint formats."""
        return self.pose_converter.get_format_info()