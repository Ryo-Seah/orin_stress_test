import cv2
import mediapipe as mp
import numpy as np

# Load ONNX emotion model
net = cv2.dnn.readNetFromONNX("EmotionFERPlus.onnx")

# Face detection
mp_face_detection = mp.solutions.face_detection
face_det = mp_face_detection.FaceDetection(model_selection=0)

def predict_emotion(face_img):
    blob = cv2.dnn.blobFromImage(face_img, scalefactor=1/255, size=(64,64), mean=(0,0,0), swapRB=False)
    net.setInput(blob)
    preds = net.forward()[0]  # shape (8,)
    emotions = ["neutral","happiness","surprise","sadness","anger","disgust","fear","contempt"]
    scores = dict(zip(emotions, preds))
    # Map to stress: negatives increase stress
    stress = scores["sadness"] + scores["anger"]*0.8 + scores["fear"]*1.2 + scores["disgust"]*0.8
    return stress, scores

# FaceMesh for cropping
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1)

# Camera setup (same gst pipeline)
cap = cv2.VideoCapture(gst_str, cv2.CAP_GSTREAMER)

while True:
    ret, frame = cap.read()
    if not ret: break

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    fd = face_det.process(rgb)
    if fd.detections:
        fm = face_mesh.process(rgb)
        if fm.multi_face_landmarks:
            pts = fm.multi_face_landmarks[0].landmark
            h, w, _ = frame.shape
            xs = [int(p.x * w) for p in pts]
            ys = [int(p.y * h) for p in pts]
            x0, y0 = max(min(xs), 0), max(min(ys), 0)
            x1, y1 = min(max(xs), w), min(max(ys), h)
            face = frame[y0:y1, x0:x1]
            face_small = cv2.resize(face, (64,64))
            stress, emotions = predict_emotion(face_small)
            cv2.putText(frame, f"Stress: {stress:.2f}", (10,30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0),2)

    cv2.imshow("Stress Monitor", frame)
    if cv2.waitKey(1)&0xFF==ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
