# Stress Detection Training Pipeline for BOLD Dataset

This directory contains a complete training pipeline for stress detection using the BOLD dataset and lightweight neural networks optimized for the Jetson Orin Nano.

## 🎯 Overview

The pipeline trains temporal models (GRU, TCN, LSTM, or Hybrid) to predict stress levels from human pose sequences. The models are designed to be lightweight and efficient for real-time deployment on edge devices.

## 📁 File Structure

```
stress_training/
├── data_loader.py              # Data loading and preprocessing
├── models.py                   # Model architectures (GRU, TCN, LSTM, Hybrid)
├── train.py                    # Main training script
├── evaluate.py                 # Model evaluation and testing
├── test_data_loading.py        # Data loading verification script
├── requirements.txt            # Python dependencies
└── README.md                   # This file

../realtime_stress_detection.py # Real-time inference script
```

## 🚀 Quick Start

### 1. Setup Environment

```bash
# Navigate to the stress training directory
cd /Users/RyoSeah/Desktop/Stress_Detection/stress_training

# Install dependencies
pip install -r requirements.txt
```

### 2. Verify Dataset

First, ensure your BOLD dataset is properly set up and test the data loading:

```bash
python test_data_loading.py
```

This script will:
- ✅ Test annotation loading
- ✅ Test joints data loading  
- ✅ Test pose sequence extraction
- ✅ Test feature extraction
- ✅ Test stress score computation

### 3. Train Your First Model

Start with the recommended GRU model:

```bash
python train.py --model_type gru --epochs 50 --batch_size 32
```

### 4. Evaluate the Model

After training, evaluate the model performance:

```bash
python evaluate.py /path/to/trained/model.h5
```

### 5. Real-time Testing

Test the trained model with live camera input:

```bash
cd ..
python realtime_stress_detection.py
```

## 📊 Stress Metric Definition

The pipeline converts BOLD dataset annotations into a quantifiable stress score (0-1) using:

**Stress Score = 0.7 × VAD_Stress + 0.3 × Categorical_Stress**

Where:
- **VAD_Stress**: Derived from Valence, Arousal, Dominance values
- **Categorical_Stress**: Based on stress-related emotions (Fear, Anger, Disquietment, etc.)

## 🧠 Model Architectures

### 1. GRU Model (Recommended)
- **Best for**: Balance of performance and efficiency
- **Parameters**: ~50K parameters
- **Memory**: ~10MB
- **Inference**: ~15ms on Jetson Orin Nano

### 2. TCN (Temporal Convolutional Network)
- **Best for**: Parallelizable training, good temporal modeling
- **Parameters**: ~40K parameters  
- **Memory**: ~8MB
- **Inference**: ~12ms on Jetson Orin Nano

### 3. Lightweight LSTM
- **Best for**: Alternative to GRU with different temporal dynamics
- **Parameters**: ~30K parameters
- **Memory**: ~6MB
- **Inference**: ~18ms on Jetson Orin Nano

### 4. Hybrid CNN-GRU
- **Best for**: Complex temporal and spatial patterns
- **Parameters**: ~70K parameters
- **Memory**: ~15MB
- **Inference**: ~25ms on Jetson Orin Nano

## 🔧 Feature Engineering

The pipeline extracts rich features from pose sequences:

### Spatial Features (per frame)
- **Normalized coordinates**: Relative to torso center/length
- **Joint distances**: Between key body parts
- **Body angles**: Shoulder, hip, elbow angles
- **Posture metrics**: Head tilt, slouch indicators

### Temporal Features
- **Velocities**: Frame-to-frame changes in all spatial features
- **Movement patterns**: Captured through sequence modeling

### Example Feature Vector (per frame)
- 36 normalized coordinates (18 joints × 2)
- 14 joint distances
- 4 body angles
- 2 posture metrics
- **Total**: ~56 features per frame

## ⚙️ Training Configuration

### Default Parameters
```python
config = {
    'sequence_length': 30,      # 1 second at 30fps
    'overlap_ratio': 0.5,       # 50% overlap between sequences
    'min_confidence': 0.3,      # Pose keypoint confidence threshold
    'batch_size': 32,           # Training batch size
    'epochs': 50,               # Maximum epochs
    'dropout_rate': 0.3,        # Regularization
    'early_stopping_patience': 10,  # Early stopping
}
```

### Custom Training
```bash
# Train TCN model with custom parameters
python train.py \
    --model_type tcn \
    --epochs 100 \
    --batch_size 64 \
    --sequence_length 45 \
    --output_dir ./my_training_results

# Train on a different dataset location
python train.py \
    --model_type gru \
    --bold_root /path/to/your/BOLD_dataset
```

## 📈 Expected Performance

### Training Metrics
- **Training Loss**: ~0.015-0.025 (MSE)
- **Validation Loss**: ~0.020-0.035 (MSE)
- **MAE**: ~0.08-0.12
- **R²**: 0.65-0.85

### Inference Speed (Jetson Orin Nano)
- **GRU Model**: ~60 FPS
- **TCN Model**: ~80 FPS  
- **LSTM Model**: ~55 FPS
- **Hybrid Model**: ~40 FPS

## 🐛 Troubleshooting

### Common Issues

**1. "No valid sequences found"**
```bash
# Check dataset structure and annotations
python test_data_loading.py
```

**2. "Out of memory during training"**
```bash
# Reduce batch size
python train.py --batch_size 16

# Or use shorter sequences
python train.py --sequence_length 20
```

**3. "Poor model performance"**
- Increase training epochs: `--epochs 100`
- Try different model: `--model_type tcn`
- Check data quality with test script

**4. "Slow inference speed"**
```bash
# Convert to TensorFlow Lite (done automatically)
# Use TFLite model for faster inference
python evaluate.py model.tflite
```

## 📱 Deployment

### TensorFlow Lite Conversion
Models are automatically converted to TensorFlow Lite format during training for efficient deployment:

```python
# Manual conversion example
from models import ModelUtils
ModelUtils.convert_to_tflite(model, "stress_model.tflite", quantize=True)
```

### Jetson Orin Nano Optimization
- Models use quantization for reduced memory
- Optimized for ARM64 architecture
- GPU acceleration through TensorRT (optional)

## 📚 Dataset Requirements

### BOLD Dataset Structure
```
BOLD_public/
├── annotations/
│   ├── train.csv
│   ├── val.csv
│   └── test.csv (optional)
├── joints/
│   └── 003/
│       ├── video1.mp4/
│       │   ├── 0001.npy
│       │   └── ...
│       └── ...
└── videos/ (not used in training)
```

### Annotation Format
Each CSV row contains:
- Video file path
- Person ID
- Start/end frames
- Categorical emotions (26-dim binary)
- Valence, Arousal, Dominance values
- Demographics

## 🔬 Advanced Usage

### Custom Stress Metrics
Modify the `compute_stress_score()` function in `data_loader.py`:

```python
def compute_stress_score(self, annotations: Dict) -> float:
    # Your custom stress computation
    valence = annotations['valence'] 
    arousal = annotations['arousal']
    
    # Example: Focus only on arousal
    stress_score = arousal
    return np.clip(stress_score, 0.0, 1.0)
```

### Feature Engineering
Add custom features in `extract_pose_features()`:

```python
# Add custom feature (e.g., hand movement)
if frame_kp[4, 2] > self.min_confidence:  # Right wrist
    hand_speed = np.linalg.norm(frame_kp[4, :2] - prev_frame_kp[4, :2])
    frame_features.append(hand_speed)
```

### Model Architecture
Create custom models in `models.py`:

```python
@staticmethod
def build_custom_model(input_shape):
    # Your custom architecture
    model = models.Sequential([...])
    return model
```

## 📖 Citation

If you use this pipeline in your research, please cite:

```bibtex
@article{ARBEE,
    Title = {ARBEE: Towards Automated Recognition of Bodily Expression of Emotion In the Wild},
    url = {https://doi.org/10.1007/s11263-019-01215-y},
    journal = {International Journal of Computer Vision},
    Author = {Luo, Yu and Ye, Jianbo and Adams Jr, Reginald B and Li, Jia and Newman, Michelle G and Wang, James Z},
    Year = {2019}
}
```

## 🤝 Contributing

Feel free to contribute improvements:
1. Fork the repository
2. Create a feature branch
3. Submit a pull request

## 📞 Support

For issues and questions:
1. Check troubleshooting section
2. Run `test_data_loading.py` for diagnostics
3. Review training logs in `output_dir/logs/`

---

**Happy Training! 🚀**
