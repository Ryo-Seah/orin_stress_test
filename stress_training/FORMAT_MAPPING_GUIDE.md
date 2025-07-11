# BOLD to COCO-17 Format Mapping Implementation

## 🎯 **Overview**

This implementation provides seamless conversion between BOLD-18 and COCO-17 keypoint formats, enabling:
1. **Training** on BOLD dataset (18 keypoints with neck)
2. **Inference** with MoveNet output (17 keypoints, no neck)
3. **Feature compatibility** between training and deployment

## 📊 **Format Specifications**

### **BOLD-18 Format (Training Data)**
```python
BOLD_18_KEYPOINTS = [
    "Nose",      # 0
    "Neck",      # 1  ← EXTRA keypoint not in COCO-17
    "RShoulder", # 2
    "RElbow",    # 3
    "RWrist",    # 4
    "LShoulder", # 5
    "LElbow",    # 6
    "LWrist",    # 7
    "RHip",      # 8
    "RKnee",     # 9
    "RAnkle",    # 10
    "LHip",      # 11
    "LKnee",     # 12
    "LAnkle",    # 13
    "REye",      # 14
    "LEye",      # 15
    "REar",      # 16
    "LEar"       # 17
]
```

### **COCO-17 Format (MoveNet Output)**
```python
COCO_17_KEYPOINTS = [
    "nose",          # 0
    "left_eye",      # 1
    "right_eye",     # 2
    "left_ear",      # 3
    "right_ear",     # 4
    "left_shoulder", # 5
    "right_shoulder",# 6
    "left_elbow",    # 7
    "right_elbow",   # 8
    "left_wrist",    # 9
    "right_wrist",   # 10
    "left_hip",      # 11
    "right_hip",     # 12
    "left_knee",     # 13
    "right_knee",    # 14
    "left_ankle",    # 15
    "right_ankle"    # 16
]
```

## 🔄 **Mapping Implementation**

### **BOLD-18 → COCO-17 Mapping**
```python
bold_to_coco17_mapping = {
    0: 0,   # Nose → nose
    1: None, # Neck → None (not in COCO-17)
    2: 6,   # RShoulder → right_shoulder
    3: 8,   # RElbow → right_elbow
    4: 10,  # RWrist → right_wrist
    5: 5,   # LShoulder → left_shoulder
    6: 7,   # LElbow → left_elbow
    7: 9,   # LWrist → left_wrist
    8: 12,  # RHip → right_hip
    9: 14,  # RKnee → right_knee
    10: 16, # RAnkle → right_ankle
    11: 11, # LHip → left_hip
    12: 13, # LKnee → left_knee
    13: 15, # LAnkle → left_ankle
    14: 2,  # REye → right_eye
    15: 1,  # LEye → left_eye
    16: 4,  # REar → right_ear
    17: 3,  # LEar → left_ear
}
```

### **COCO-17 → BOLD-18 Mapping**
```python
coco17_to_bold_mapping = {
    0: 0,   # nose → Nose
    1: 15,  # left_eye → LEye
    2: 14,  # right_eye → REye
    3: 17,  # left_ear → LEar
    4: 16,  # right_ear → REar
    5: 5,   # left_shoulder → LShoulder
    6: 2,   # right_shoulder → RShoulder
    7: 6,   # left_elbow → LElbow
    8: 3,   # right_elbow → RElbow
    9: 7,   # left_wrist → LWrist
    10: 4,  # right_wrist → RWrist
    11: 11, # left_hip → LHip
    12: 8,  # right_hip → RHip
    13: 12, # left_knee → LKnee
    14: 9,  # right_knee → RKnee
    15: 13, # left_ankle → LAnkle
    16: 10, # right_ankle → RAnkle
}
```

## 🛠️ **Key Methods**

### **1. BOLD-18 to COCO-17 Conversion**
```python
def convert_bold18_to_coco17(self, bold_keypoints: np.ndarray) -> np.ndarray:
    """
    Convert BOLD 18-keypoint format to COCO-17 format.
    - Input: (seq_len, 18, 3) or (18, 3)
    - Output: (seq_len, 17, 3) or (17, 3)
    - Neck keypoint is dropped (not in COCO-17)
    """
```

### **2. COCO-17 to BOLD-18 Conversion**
```python
def convert_coco17_to_bold18(self, coco17_keypoints: np.ndarray) -> np.ndarray:
    """
    Convert COCO-17 format to BOLD-18 format with estimated neck.
    - Input: (seq_len, 17, 3) or (17, 3)
    - Output: (seq_len, 18, 3) or (18, 3)
    - Neck is estimated as midpoint of shoulders
    """
```

### **3. Real-time Stress Detection**
```python
def compute_stress_score_realtime(self, coco17_keypoints: np.ndarray) -> float:
    """
    Compute stress score from COCO-17 keypoints for real-time inference.
    - Input: Single frame COCO-17 keypoints (17, 3)
    - Output: Stress score [0, 5] (compatible with existing movenet.py)
    - Uses same feature engineering as training pipeline
    """
```

### **4. Feature Extraction from COCO-17**
```python
def extract_features_from_coco17(self, coco17_sequence: np.ndarray) -> np.ndarray:
    """
    Extract features from COCO-17 sequence for model prediction.
    - Input: (sequence_length, 17, 3)
    - Output: (sequence_length, num_features)
    - Automatically converts to BOLD-18 internally
    """
```

## 🎯 **Neck Estimation Algorithm**

Since COCO-17 doesn't have a neck keypoint, we estimate it from shoulders:

```python
# Only estimate if both shoulders are confident
if left_shoulder[2] > min_confidence and right_shoulder[2] > min_confidence:
    neck_x = (left_shoulder[0] + right_shoulder[0]) / 2
    neck_y = (left_shoulder[1] + right_shoulder[1]) / 2
    neck_conf = min(left_shoulder[2], right_shoulder[2])
    bold_keypoints[i, 1] = [neck_x, neck_y, neck_conf]
```

## 🚀 **Integration Examples**

### **Training Pipeline (BOLD-18)**
```python
# Load BOLD dataset
data_loader = BOLDDataLoader(bold_root="/path/to/BOLD_public")
X_train, y_train = data_loader.load_dataset('train')  # Automatic BOLD-18 processing

# Train model
model.fit(X_train, y_train)
```

### **Real-time Inference (COCO-17)**
```python
# MoveNet detection (returns COCO-17)
keypoints = detect_pose_movenet(frame)  # Shape: (17, 3)

# Method 1: Simple stress score
stress_score = data_loader.compute_stress_score_realtime(keypoints)

# Method 2: Sequence-based prediction (more accurate)
coco17_sequence = collect_keypoints_over_time()  # Shape: (30, 17, 3)
features = data_loader.extract_features_from_coco17(coco17_sequence)
prediction = trained_model.predict(features)
```

### **Format Conversion**
```python
# Convert between formats
bold18_kp = data_loader.convert_coco17_to_bold18(coco17_kp)
coco17_kp = data_loader.convert_bold18_to_coco17(bold18_kp)

# Get format information
info = data_loader.get_format_info()
print(f"BOLD joints: {info['bold18_joints']}")
print(f"COCO joints: {info['coco17_joints']}")
```

## ✅ **Validation Results**

The format conversion has been tested and validated:

- ✅ **BOLD-18 → COCO-17**: Correctly maps 17 out of 18 joints (neck dropped)
- ✅ **COCO-17 → BOLD-18**: Successfully estimates neck from shoulders
- ✅ **Feature extraction**: Produces identical 54-dimensional features
- ✅ **Real-time detection**: Compatible with existing movenet.py (0-5 scale)
- ✅ **Sequence processing**: Handles both single frames and sequences

## **Achieved the Following:**

1. **Seamless Training**: Use full BOLD dataset without modification
2. **MoveNet Compatible**: Direct integration with existing inference code
3. **Flexible Input**: Handles both single frames and sequences
4. **Neck Estimation**: Intelligent neck position estimation from shoulders

This implementation ensures that your trained stress detection model will work seamlessly with MoveNet's COCO-17 output while maintaining the rich feature engineering developed for the BOLD-18 training data!
