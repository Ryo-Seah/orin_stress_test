#!/usr/bin/env python3
"""
Audio Device Tester for Jetson Orin
Tests all available audio input devices to help identify your USB microphone.
"""

import sounddevice as sd
import numpy as np
import time

def list_audio_devices():
    """List all available audio devices."""
    print("🎤 Available Audio Devices:")
    print("=" * 60)
    
    devices = sd.query_devices()
    input_devices = []
    
    for idx, device in enumerate(devices):
        input_ch = device['max_input_channels']
        output_ch = device['max_output_channels']
        sample_rate = device['default_samplerate']
        
        device_type = []
        if input_ch > 0:
            device_type.append(f"INPUT({input_ch}ch)")
            input_devices.append(idx)
        if output_ch > 0:
            device_type.append(f"OUTPUT({output_ch}ch)")
        
        type_str = " + ".join(device_type) if device_type else "NO I/O"
        
        print(f"[{idx:2d}] {device['name']}")
        print(f"     Type: {type_str}")
        print(f"     Sample Rate: {sample_rate:.0f} Hz")
        print()
    
    return input_devices

def test_audio_device(device_id, duration=2):
    """Test recording from a specific audio device."""
    try:
        devices = sd.query_devices()
        device = devices[device_id]
        
        print(f"🎙️  Testing Device [{device_id}]: {device['name']}")
        print(f"   Recording {duration} seconds...")
        
        # Record audio
        sample_rate = int(device['default_samplerate'])
        audio = sd.rec(
            int(duration * sample_rate),
            samplerate=sample_rate,
            channels=1,
            device=device_id,
            dtype='float32'
        )
        sd.wait()  # Wait for recording to complete
        
        # Analyze the recording
        audio_squeezed = np.squeeze(audio)
        amplitude = np.abs(audio_squeezed).mean()
        max_amplitude = np.abs(audio_squeezed).max()
        
        print(f"   ✅ Recording successful!")
        print(f"   📊 Average amplitude: {amplitude:.6f}")
        print(f"   📈 Peak amplitude: {max_amplitude:.6f}")
        
        if amplitude > 0.001:
            print(f"   🔊 GOOD - Device is picking up audio!")
        elif amplitude > 0.0001:
            print(f"   🔉 WEAK - Very low audio signal")
        else:
            print(f"   🔇 SILENT - No audio detected")
        
        return True, amplitude
        
    except Exception as e:
        print(f"   ❌ Error testing device: {e}")
        return False, 0.0

def main():
    """Main function to test audio devices."""
    print("🔧 Audio Device Tester for Jetson Orin")
    print("=" * 50)
    print()
    
    # List all devices
    input_devices = list_audio_devices()
    
    if not input_devices:
        print("❌ No input devices found!")
        return
    
    print(f"📋 Found {len(input_devices)} input device(s): {input_devices}")
    print()
    
    # Test each input device
    print("🧪 Testing Input Devices...")
    print("=" * 40)
    print("💡 Make some noise (tap, speak) while testing each device!")
    print()
    
    best_device = None
    best_amplitude = 0.0
    
    for device_id in input_devices:
        print(f"Testing device {device_id}...")
        input("   Press ENTER when ready to record (make noise!): ")
        
        success, amplitude = test_audio_device(device_id, duration=3)
        
        if success and amplitude > best_amplitude:
            best_device = device_id
            best_amplitude = amplitude
        
        print()
        time.sleep(1)
    
    # Recommendations
    print("🎯 RECOMMENDATIONS:")
    print("=" * 30)
    
    if best_device is not None:
        devices = sd.query_devices()
        print(f"✅ Best device: [{best_device}] {devices[best_device]['name']}")
        print(f"   Amplitude: {best_amplitude:.6f}")
        print()
        print(f"🔧 To use this device in your code, set:")
        print(f"   device_id = {best_device}")
        print()
        print(f"   Or in multimodal_detector.py line ~56:")
        print(f"   audio_device_id = {best_device}")
    else:
        print("❌ No working audio input devices found!")

if __name__ == "__main__":
    main()