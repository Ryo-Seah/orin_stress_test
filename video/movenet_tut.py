import os
import cv2
import numpy as np
import tensorflow as tf

# Choose your model
model_name = "movenet_lightning_int8"  # Change to e.g. "movenet_thunder_f16" if needed

# Download and prepare model
input_size = 192 if "lightning" in model_name else 256
model_path = "place.tflite"

if not os.path.exists(model_path):
    if "lightning_f16" in model_name:
        os.system("wget -q -O model.tflite https://tfhub.dev/google/lite-model/movenet/singlepose/lightning/tflite/float16/4?lite-format=tflite")
    elif "thunder_f16" in model_name:
        os.system("wget -q -O model.tflite https://tfhub.dev/google/lite-model/movenet/singlepose/thunder/tflite/float16/4?lite-format=tflite")
    elif "lightning_int8" in model_name:
        os.system("wget -q -O model.tflite https://tfhub.dev/google/lite-model/movenet/singlepose/lightning/tflite/int8/4?lite-format=tflite")
    elif "thunder_int8" in model_name:
        os.system("wget -q -O model.tflite https://tfhub.dev/google/lite-model/movenet/singlepose/thunder/tflite/int8/4?lite-format=tflite")
    else:
        raise ValueError("Unsupported model name.")

# Initialize interpreter
interpreter = tf.lite.Interpreter(model_path=model_path)
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

def movenet(input_image):
    """Run pose estimation on input image."""
    input_image = tf.image.resize_with_pad(input_image, input_size, input_size)
    input_image = tf.expand_dims(input_image, axis=0)
    input_image = tf.cast(input_image, dtype=tf.uint8)
    interpreter.set_tensor(input_details[0]['index'], input_image.numpy())
    interpreter.invoke()
    keypoints_with_scores = interpreter.get_tensor(output_details[0]['index'])
    return keypoints_with_scores

# Live camera feed
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("❌ Failed to open camera.")
    exit()

print("✅ Press 'q' to quit.")
while True:
    ret, frame = cap.read()
    if not ret:
        break

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    keypoints_with_scores = movenet(tf.convert_to_tensor(rgb))

    # Draw keypoints
    height, width, _ = frame.shape
    for idx in range(17):
        y, x, c = keypoints_with_scores[0][0][idx]
        if c > 0.3:
            cx, cy = int(x * width), int(y * height)
            cv2.circle(frame, (cx, cy), 4, (0, 255, 0), -1)

    cv2.imshow("MoveNet Live", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
