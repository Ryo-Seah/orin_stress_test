import cv2
import mediapipe as mp
import numpy as np

# pose estimation
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=False)
drawing = mp.solutions.drawing_utils

def estimate_body_stress(landmarks):
    # Calculate simple stress indicators (e.g. shoulder slump, head tilt)
    LEFT_SHOULDER = mp_pose.PoseLandmark.LEFT_SHOULDER.value
    RIGHT_SHOULDER = mp_pose.PoseLandmark.RIGHT_SHOULDER.value
    LEFT_HIP = mp_pose.PoseLandmark.LEFT_HIP.value
    RIGHT_HIP = mp_pose.PoseLandmark.RIGHT_HIP.value

    # Shoulder to hip distance as slouch indicator
    dy_l = landmarks[LEFT_SHOULDER].y - landmarks[LEFT_HIP].y
    dy_r = landmarks[RIGHT_SHOULDER].y - landmarks[RIGHT_HIP].y
    slouch_score = (dy_l + dy_r) / 2

    return np.clip(slouch_score * 10, 0, 5)  # Scaled stress score


# # FaceMesh for cropping
# mp_face_mesh = mp.solutions.face_mesh
# face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1)

# GStreamer pipeline (adjust if needed)
gst_str = (
    "nvarguscamerasrc sensor-id=0 ! "
    "nvvidconv ! "
    "videoconvert ! "
    "video/x-raw, format=BGR ! appsink"
)
cap = cv2.VideoCapture(gst_str, cv2.CAP_GSTREAMER)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret: break
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(rgb)

    if results.pose_landmarks:
        landmarks = results.pose_landmarks.landmark
        stress_score = estimate_body_stress(landmarks)
        cv2.putText(frame, f"Stress Score: {stress_score:.2f}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

    cv2.imshow("Body Stress Monitor", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'): break

cap.release()
cv2.destroyAllWindows()
