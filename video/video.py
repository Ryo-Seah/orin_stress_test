import cv2
import mediapipe as mp
import numpy as np
import time

# initialize person detector: lightweight option
hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

# initialize mediapipe face landmarks
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1)

# util: Simple stress score
def estimate_stress(landmarks):
    # Simple: measure eye aspect ratio & mouth opening
    LEFT_EYE = [33, 133]
    MOUTH = [13, 14]
    left_eye = np.linalg.norm(landmarks[LEFT_EYE[0]] - landmarks[LEFT_EYE[1]])
    mouth = np.linalg.norm(landmarks[MOUTH[0]] - landmarks[MOUTH[1]])
    score = mouth / (left_eye + 1e-6)  # Normalize to avoid div0
    return np.clip(score, 0, 5)

# Init video stream (default camera)
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Camera not accessible")
    exit()

print("Press 'q' to quit")
while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Resize for faster inference
    resized = cv2.resize(frame, (320, 240))
    gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)

    # 1. Detect person
    boxes, _ = hog.detectMultiScale(gray, winStride=(8, 8))
    person_detected = len(boxes) > 0

    stress_score = None
    if person_detected:
        # 2. Detect face + facial landmarks
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(rgb)
        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                h, w, _ = resized.shape
                landmarks = np.array([[lm.x * w, lm.y * h] for lm in face_landmarks.landmark])
                # 3. Estimate stress
                stress_score = estimate_stress(landmarks)
                cv2.putText(resized, f"Stress Score: {stress_score:.2f}", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    # Draw person box
    for (x, y, w, h) in boxes:
        cv2.rectangle(resized, (x, y), (x + w, y + h), (255, 0, 0), 2)

    # Display frame
    cv2.imshow("Stress Monitor", resized)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
