import cv2

# GStreamer pipeline
gst_str = (
    "nvarguscamerasrc sensor-id=0 ! "
    "nvvidconv ! "
    "videoconvert ! "
    "video/x-raw, format=BGR ! appsink"
)

# Open video stream
cap = cv2.VideoCapture(gst_str, cv2.CAP_GSTREAMER)

if not cap.isOpened():
    print("❌ Failed to open camera via GStreamer.")
    exit()

print("✅ Press 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        print("❌ Failed to read frame.")
        break

    cv2.imshow("IMX219 Camera Test", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
