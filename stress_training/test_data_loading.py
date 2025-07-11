"""
Quick test script to verify BOLD dataset loading and feature extraction.
Run this before starting full training to ensure everything is working correctly.
"""

import os
import sys
import numpy as np

# Add the current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from data_loader import BOLDDataLoader
    print("✅ Successfully imported BOLDDataLoader")
except ImportError as e:
    print(f"❌ Failed to import BOLDDataLoader: {e}")
    sys.exit(1)


def test_data_loading():
    """Test basic data loading functionality."""
    print("\n" + "="*50)
    print("TESTING BOLD DATASET LOADING")
    print("="*50)
    
    # Initialize data loader
    bold_root = "/Users/RyoSeah/Downloads/BOLD_public"
    
    if not os.path.exists(bold_root):
        print(f"❌ BOLD dataset not found at: {bold_root}")
        print("Please download and extract the BOLD dataset first.")
        return False
    
    data_loader = BOLDDataLoader(
        bold_root=bold_root,
        sequence_length=10,  # Use shorter sequence for testing
        overlap_ratio=0.5,
        min_confidence=0.3
    )
    
    # Test annotation loading
    print("\n1. Testing annotation loading...")
    try:
        train_annotations = data_loader.load_annotations('train')
        val_annotations = data_loader.load_annotations('val')
        
        print(f"   ✅ Train annotations: {len(train_annotations)} samples")
        print(f"   ✅ Val annotations: {len(val_annotations)} samples")
        
        # Show sample annotation
        if len(train_annotations) > 0:
            sample = train_annotations.iloc[0]
            print(f"   📋 Sample annotation:")
            print(f"      Video: {sample['video_file']}")
            print(f"      Person ID: {sample['person_id']}")
            print(f"      Frames: {sample['start_frame']}-{sample['end_frame']}")
            print(f"      Valence: {sample['valence']:.3f}")
            print(f"      Arousal: {sample['arousal']:.3f}")
            print(f"      Dominance: {sample['dominance']:.3f}")
    except Exception as e:
        print(f"   ❌ Error loading annotations: {e}")
        return False
    
    # Test joints data loading
    print("\n2. Testing joints data loading...")
    try:
        sample = train_annotations.iloc[0]
        video_file = sample['video_file']
        print(f"   📋 Video file: {video_file}")
        
        # Parse video file path: format is "003/video_name.mp4/clip_id.mp4"
        # We need to extract the directory (003) and the clip_id
        if isinstance(video_file, str):
            parts = video_file.split('/')
            if len(parts) == 3:
                video_dir = parts[0]  # e.g., "003"
                clip_id = parts[2].replace('.mp4', '')  # e.g., "0114"
                
                print(f"   📋 Video dir: {video_dir}, Clip ID: {clip_id}")
                
                joints_data = data_loader.load_joints_data(f"{video_dir}/{parts[1]}", clip_id)
                if joints_data is not None:
                    print(f"   ✅ Joints data shape: {joints_data.shape}")
                    print(f"   📋 Sample joints data (first 2 rows):")
                    print(f"      {joints_data[:2]}")
                else:
                    print(f"   ❌ Failed to load joints data for {video_dir}/{parts[1]}/{clip_id}")
                    return False
            else:
                print(f"   ❌ Unexpected video file format: {video_file}")
                return False
        else:
            print(f"   ❌ Video file is not a string: {type(video_file)} = {video_file}")
            return False
    except Exception as e:
        print(f"   ❌ Error loading joints data: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Test pose sequence extraction
    print("\n3. Testing pose sequence extraction...")
    try:
        print(f"   📋 Extracting sequence for person {sample['person_id']} from frames {sample['start_frame']}-{sample['end_frame']}")
        print(f"   📋 Joints data shape: {joints_data.shape}")
        print(f"   📋 Unique person IDs in data: {np.unique(joints_data[:, 1])}")
        print(f"   📋 Frame range in data: {joints_data[:, 0].min():.0f} - {joints_data[:, 0].max():.0f}")
        
        pose_sequence = data_loader.extract_person_sequence(
            joints_data, sample['person_id'], 
            sample['start_frame'], sample['end_frame']
        )
        
        if pose_sequence is not None:
            print(f"   ✅ Pose sequence shape: {pose_sequence.shape}")
            print(f"   📋 Expected shape: (num_frames, 18, 3)")
        else:
            print(f"   ❌ Failed to extract pose sequence")
            
            # Debug information
            person_mask = joints_data[:, 1] == sample['person_id']
            person_data = joints_data[person_mask]
            print(f"   📋 Person data found: {len(person_data)} frames")
            
            if len(person_data) > 0:
                frame_mask = (person_data[:, 0] >= sample['start_frame']) & (person_data[:, 0] <= sample['end_frame'])
                frame_data = person_data[frame_mask]
                print(f"   📋 Frames in range: {len(frame_data)}")
                print(f"   📋 Required minimum frames: {data_loader.sequence_length // 2}")
                
                if len(frame_data) > 0:
                    print(f"   📋 Available frame numbers: {frame_data[:5, 0]}")  # Show first 5 frame numbers
            return False
    except Exception as e:
        print(f"   ❌ Error extracting pose sequence: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Test feature extraction
    print("\n4. Testing feature extraction...")
    try:
        # Use a smaller sequence for testing
        test_sequence = pose_sequence[:data_loader.sequence_length] if len(pose_sequence) >= data_loader.sequence_length else pose_sequence
        
        features = data_loader.extract_pose_features(test_sequence)
        print(f"   ✅ Features shape: {features.shape}")
        print(f"   📋 Features per frame: {features.shape[1]}")
        
        # Test velocity features
        velocities = data_loader.compute_velocity_features(features)
        print(f"   ✅ Velocities shape: {velocities.shape}")
        
    except Exception as e:
        print(f"   ❌ Error extracting features: {e}")
        return False
    
    # Test stress score computation
    print("\n5. Testing stress score computation...")
    try:
        annotations_dict = {
            'valence': sample['valence'],
            'arousal': sample['arousal'],
            'dominance': sample['dominance'],
            'categorical': sample['categorical']
        }
        stress_score = data_loader.compute_stress_score(annotations_dict)
        print(f"   ✅ Stress score: {stress_score:.3f}")
        print(f"   📋 Score range should be [0, 1]")
        
    except Exception as e:
        print(f"   ❌ Error computing stress score: {e}")
        return False
    
    print("\n" + "="*50)
    print("✅ ALL TESTS PASSED!")
    print("✅ Data loading system is working correctly.")
    print("✅ Ready to start training!")
    print("="*50)
    
    return True


def test_small_dataset_loading():
    """Test loading a small subset of the dataset."""
    print("\n" + "="*50)
    print("TESTING SMALL DATASET LOADING")
    print("="*50)
    
    bold_root = "/Users/RyoSeah/Downloads/BOLD_public"
    
    data_loader = BOLDDataLoader(
        bold_root=bold_root,
        sequence_length=10,  # Short sequence for testing
        overlap_ratio=0.5,
        min_confidence=0.3
    )
    
    # Load a small amount of training data
    print("Loading small training dataset...")
    
    # Temporarily modify the load_dataset method to load only first 10 samples
    original_load_annotations = data_loader.load_annotations
    
    def load_limited_annotations(split):
        df = original_load_annotations(split)
        return df.head(10)  # Only first 10 samples
    
    data_loader.load_annotations = load_limited_annotations
    
    try:
        X_train, y_train = data_loader.load_dataset('train')
        
        if len(X_train) > 0:
            print(f"✅ Successfully loaded {len(X_train)} training sequences")
            print(f"✅ Feature shape: {X_train.shape}")
            print(f"✅ Stress scores range: {y_train.min():.3f} - {y_train.max():.3f}")
            
            # Test scaling
            data_loader.fit_scaler(X_train)
            X_scaled = data_loader.transform_features(X_train)
            print(f"✅ Feature scaling completed")
            print(f"✅ Scaled features range: {X_scaled.min():.3f} - {X_scaled.max():.3f}")
            
            return True
        else:
            print("❌ No training sequences loaded")
            return False
            
    except Exception as e:
        print(f"❌ Error loading dataset: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all tests."""
    print("🔍 BOLD Dataset Testing Suite")
    print("This script tests the data loading pipeline before training.")
    
    # Test 1: Basic functionality
    success1 = test_data_loading()
    
    if success1:
        # Test 2: Small dataset loading
        success2 = test_small_dataset_loading()
        
        if success2:
            print("\n🎉 ALL TESTS COMPLETED SUCCESSFULLY!")
            print("🚀 You can now run the full training pipeline with:")
            print("   python train.py --model_type gru --epochs 50")
        else:
            print("\n❌ Dataset loading test failed.")
    else:
        print("\n❌ Basic functionality test failed.")
        print("🔧 Please check:")
        print("   1. BOLD dataset is properly downloaded and extracted")
        print("   2. All required packages are installed")
        print("   3. Dataset paths are correct")


if __name__ == "__main__":
    main()
