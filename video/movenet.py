import cv2
import numpy as np
import tensorflow as tf  # Use tflite_runtime if full TensorFlow is too heavy
import time

# Load MoveNet TFLite model
interpreter = tf.lite.Interpreter(model_path="movenet_thunder.tflite")  # or "movenet_lightning.tflite"
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
input_size = input_details[0]['shape'][2]

# GStreamer camera string for Jetson
gst_str = (
    "nvarguscamerasrc sensor-id=0 ! "
    "nvvidconv ! "
    "videoconvert ! "
    "video/x-raw, format=BGR ! appsink"
)

# Pose estimation from image
def detect_pose_movenet(frame):
    img = cv2.resize(frame, (input_size, input_size))
    input_data = np.expand_dims(img, axis=0).astype(np.uint8)
    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()
    keypoints = interpreter.get_tensor(output_details[0]['index'])[0][0]  # shape: (17, 3)
    return keypoints

# Stress scoring logic
def compute_stress_score(keypoints):
    kp = keypoints[:, :2]  # (x, y)
    LEFT_SHOULDER, RIGHT_SHOULDER = kp[5], kp[6]
    LEFT_HIP, RIGHT_HIP = kp[11], kp[12]
    NOSE = kp[0]

    shoulder_avg = (LEFT_SHOULDER + RIGHT_SHOULDER) / 2
    hip_avg = (LEFT_HIP + RIGHT_HIP) / 2
    torso_length = np.linalg.norm(shoulder_avg - hip_avg)
    head_tilt = np.linalg.norm(NOSE - shoulder_avg) / (torso_length + 1e-6)
    slouch = (LEFT_SHOULDER[1] - LEFT_HIP[1] + RIGHT_SHOULDER[1] - RIGHT_HIP[1]) / 2
    score = 2.0 * slouch + 3.0 * head_tilt
    return float(np.clip(score, 0, 5))

# Run loop
cap = cv2.VideoCapture(gst_str, cv2.CAP_GSTREAMER)
if not cap.isOpened():
    print("❌ Failed to open camera.")
    exit()

print("✅ Press 'q' to quit.")
while True:
    ret, frame = cap.read()
    if not ret: break

    keypoints = detect_pose_movenet(frame)
    score = compute_stress_score(keypoints)

    # Draw keypoints
    for i, (x, y, c) in enumerate(keypoints):
        if c > 0.3:
            cx, cy = int(x * frame.shape[1]), int(y * frame.shape[0])
            cv2.circle(frame, (cx, cy), 4, (0, 255, 0), -1)

    # Display stress score
    cv2.putText(frame, f"Stress Score: {score:.2f}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.imshow("MoveNet Stress Estimation", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'): break

cap.release()
cv2.destroyAllWindows()
