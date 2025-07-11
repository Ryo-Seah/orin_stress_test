import os
import cv2
import numpy as np
import tensorflow as tf
import requests # More robust for downloading than os.system('wget')

# Choose your model
model_name = "movenet_lightning_int8"  # Change to e.g. "movenet_thunder_f16" if needed

# --- Model Download and Preparation ---
input_size = 192 if "lightning" in model_name else 256
model_filename = "model.tflite" # Define the consistent filename for the downloaded model

# URLs for the MoveNet models
model_urls = {
    "movenet_lightning_f16": "https://tfhub.dev/google/lite-model/movenet/singlepose/lightning/tflite/float16/4?lite-format=tflite",
    "movenet_thunder_f16": "https://tfhub.dev/google/lite-model/movenet/singlepose/thunder/tflite/float16/4?lite-format=tflite",
    "movenet_lightning_int8": "https://tfhub.dev/google/lite-model/movenet/singlepose/lightning/tflite/int8/4?lite-format=tflite",
    "movenet_thunder_int8": "https://tfhub.dev/google/lite-model/movenet/singlepose/thunder/tflite/int8/4?lite-format=tflite"
}

if model_name not in model_urls:
    raise ValueError(f"Unsupported model name: {model_name}. Choose from {list(model_urls.keys())}")

# Check if model already exists, otherwise download
if not os.path.exists(model_filename):
    print(f"Downloading {model_name} model...")
    try:
        response = requests.get(model_urls[model_name])
        response.raise_for_status() # Raise an exception for bad status codes
        with open(model_filename, 'wb') as f:
            f.write(response.content)
        print(f"Model downloaded to {model_filename}")
    except requests.exceptions.RequestException as e:
        print(f"❌ Error downloading model: {e}")
        exit()
else:
    print(f"Model '{model_filename}' already exists. Skipping download.")

# Initialize interpreter
try:
    interpreter = tf.lite.Interpreter(model_path=model_filename) # FIXED: Use model_filename
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    print("✅ TensorFlow Lite interpreter initialized.")
except Exception as e:
    print(f"❌ Error initializing TensorFlow Lite interpreter: {e}")
    exit()

def movenet(input_image):
    """Run pose estimation on input image."""
    # Ensure input_image is a tf.Tensor
    if not isinstance(input_image, tf.Tensor):
        input_image = tf.convert_to_tensor(input_image)

    input_image = tf.image.resize_with_pad(input_image, input_size, input_size)
    input_image = tf.expand_dims(input_image, axis=0)
    input_image = tf.cast(input_image, dtype=input_details[0]['dtype']) # Cast to model's expected dtype
    
    # Check input shape before setting tensor
    expected_shape = input_details[0]['shape']
    if list(input_image.shape) != list(expected_shape):
        raise ValueError(f"Input image shape mismatch. Expected {expected_shape}, got {input_image.shape}")

    interpreter.set_tensor(input_details[0]['index'], input_image.numpy())
    interpreter.invoke()
    keypoints_with_scores = interpreter.get_tensor(output_details[0]['index'])
    return keypoints_with_scores

# Live camera feed
cap = cv2.VideoCapture(0) # 0 is typically the default webcam
if not cap.isOpened():
    print("❌ Failed to open camera. Check if another application is using it or if the camera index is correct.")
    # Try other indices if 0 fails, e.g., cap = cv2.VideoCapture(1)
    exit()

print("✅ Press 'q' to quit.")
while True:
    ret, frame = cap.read()
    if not ret:
        print("❌ Failed to grab frame from camera. Exiting.")
        break

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    try:
        keypoints_with_scores = movenet(rgb)
    except Exception as e:
        print(f"❌ Error during MoveNet inference: {e}")
        # Optionally, you can continue or break depending on how critical the error is
        continue 

    # Draw keypoints
    height, width, _ = frame.shape
    # Ensure keypoints_with_scores has the expected structure before accessing
    # It should be a numpy array of shape (1, 1, 17, 3)
    if keypoints_with_scores.shape == (1, 1, 17, 3):
        for idx in range(17):
            y, x, c = keypoints_with_scores[0][0][idx]
            # Draw only if confidence is above threshold
            if c > 0.3: # This threshold is important for filtering out bad predictions (e.g., when camera is blocked)
                cx, cy = int(x * width), int(y * height)
                cv2.circle(frame, (cx, cy), 4, (0, 255, 0), -1)
    else:
        print(f"Warning: Unexpected keypoints_with_scores shape: {keypoints_with_scores.shape}. Expected (1, 1, 17, 3)")


    cv2.imshow("MoveNet Live", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()