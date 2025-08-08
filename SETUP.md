# NVIDIA Jetson Orin Setup Guide for Stress Detection System

## 📦 Hardware Components Required

### Essential Components
- **NVIDIA Jetson Orin Nano Developer Kit**
- **Power supply** (USB-C PD, 15W minimum)
- **HDMI to DisplayPort cable** (or HDMI to HDMI if monitor supports it)
- **USB-C Hub** (for multiple USB connections)
- **MicroSD card** (64GB minimum, Class 10 or higher)

### Peripherals (All USB-C Compatible)
- **USB Camera** with microphone capability
- **USB Microphone** (dedicated, higher quality than camera mic)
- **USB Mouse**
- **USB Keyboard**
- **Monitor** with DisplayPort or HDMI input
- **Ethernet cable** (optional, for stable internet)

### Optional Components
- **Cooling fan** (recommended for sustained workloads)
- **WiFi adapter** (if not using built-in WiFi)

---

## 🔌 Hardware Setup

### Step 1: Initial Connections
1. **Power OFF** - Ensure Jetson Orin is powered off
2. **Connect HDMI** - Plug HDMI to DisplayPort cable from Jetson to monitor
3. **Connect USB Hub** - Attach USB-C hub to Jetson Orin
4. **Connect Peripherals** to USB hub:
   - USB Mouse
   - USB Keyboard
   - USB Camera
   - USB Microphone
5. **Connect Power** - Plug in USB-C power supply (**DO NOT POWER ON YET**)

### Step 2: Network Setup
**Option A: Ethernet (Recommended)**
- Connect ethernet cable for stable internet connection

**Option B: WiFi**
- Ensure WiFi adapter is connected (if using external)

### Step 3: Power On
1. **Power on the monitor** first
2. **Power on Jetson Orin** by connecting power supply
3. **Wait for boot** (first boot may take 2-3 minutes)

---

## 🔐 Initial Login

### Default Credentials
- **Username**: `orin_nano`
- **Password**: `jetson`

### First Boot Setup
1. **Follow Ubuntu setup wizard** if prompted
2. **Connect to WiFi** (if not using ethernet)
3. **Update system** (recommended):
   ```bash
   sudo apt update && sudo apt upgrade -y
   ```

---

## 🛠️ Software Installation

### Step 1: Install Python Dependencies (only for new setup)
```bash
# Update package manager
sudo apt update

# Install Python 3 and pip
sudo apt install python python-pip python-venv -y

# Install system dependencies
sudo apt install build-essential cmake git -y
sudo apt install portaudio19-dev python-pyaudio -y
sudo apt install libsndfile1-dev -y
```

<!-- ### Step 2: Install JetPack Components
```bash
# Install JetPack SDK components (if not already installed)
sudo apt install nvidia-jetpack -y

# Verify CUDA installation
nvcc --version
```

### Step 3: Install TensorFlow Lite Runtime
```bash
# Install TensorFlow Lite for ARM64
pip3 install --index-url https://google-coral.github.io/py-repo/ tflite_runtime
``` -->

---

## 🎥 Hardware Testing

### Test USB Camera
```bash
# List video devices
ls /dev/video*

# Test camera with simple capture
python -c "
import cv2
cap = cv2.VideoCapture(0)
ret, frame = cap.read()
if ret:
    print('✅ Camera working')
    print(f'Resolution: {frame.shape}')
else:
    print('❌ Camera not working')
cap.release()
"
```

### Test Audio Devices
```bash
# Install audio testing tool
pip3 install sounddevice numpy

# Create and run audio test script
python test_audio_devices.py
```

---

## 📁 Project Setup

### Step 1: Clone Repository
```bash
# Navigate to home directory
cd ~

# Clone the stress detection project
git clone https://github.com/your-repo/orin_stress_test.git (Should already be in)
cd orin_stress_test
```

### Step 2: Create Virtual Environment
```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
source venv/bin/activate

# Install project dependencies
pip install -r requirements.txt
pip install -r stress_training/requirements.txt
```

### Step 3: Download Models (not required for existing set up)
```bash
# Create models directory
mkdir -p models training_results/models

# Download MoveNet model (example)
# wget https://tfhub.dev/google/movenet/singlepose/thunder/4?tf-hub-format=compressed -O movenet_thunder.tflite

# Place your trained models in training_results/models/
```

---

## ⚙️ System Configuration

### Optimize for Performance
```bash
# Set performance mode
sudo nvpmodel -m 0
sudo jetson_clocks

# Check current performance mode
sudo nvpmodel -q
```

### Configure Audio Permissions
```bash
# Add user to audio group
sudo usermod -a -G audio $USER

# Logout and login again for changes to take effect
```

---

## 🧪 Testing & Verification

### Step 1: Test Audio Setup
```bash
cd orin_stress_test
python test_audio_devices.py
```
**Expected Output:**
- Lists all audio devices
- Tests each input device
- Recommends best device ID
- Shows amplitude levels

### Step 2: Test Video Processing
```bash
# Test MoveNet pose detection
python -c "
from working.video_stress_detector import VideoStressDetector
print('✅ Video processing modules imported successfully')
"
```

### Step 3: Test Multimodal System
```bash
# Activate virtual environment
source venv/bin/activate

# Run multimodal detector
cd working
python multimodal_detector.py
```

**Expected Behavior:**
- Camera window opens
- Pose keypoints detected and drawn
- Audio VAD scores printed in terminal
- No error messages

---

## 🔧 Troubleshooting

### Common Issues

**Camera not detected:**
```bash
# Check USB devices
lsusb

# Check video devices
ls -la /dev/video*

# Test different USB ports
```

**Audio device issues:**
```bash
# List audio devices
python -c "import sounddevice as sd; print(sd.query_devices())"

# Check ALSA devices
aplay -l
arecord -l
```

**Performance issues:**
```bash
# Check system resources
htop

# Monitor GPU usage
sudo tegrastats

# Check temperature
cat /sys/class/thermal/thermal_zone*/temp
```

**Permission errors:**
```bash
# Fix camera permissions
sudo chmod 666 /dev/video*

# Fix audio permissions
sudo usermod -a -G audio $USER
```

---

## 🚀 Quick Start Commands

```bash
# Complete setup in one go
cd ~/orin_stress_test
source venv/bin/activate
python test_audio_devices.py  # Note the recommended device ID
# Edit multimodal_detector.py with correct audio device ID
cd working
python multimodal_detector.py
```

**System Information:**
- **OS**: Ubuntu 20.04 (JetPack)
- **Python**: 3.8+
- **CUDA**: 11.4+
- **TensorRT**: 8.2+