import cv2

# Open the default USB camera (usually index 0 or 1)
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("❌ Failed to open USB camera.")
    exit()

print("✅ USB camera opened successfully. Press 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        print("❌ Failed to read frame from USB camera.")
        break

    cv2.imshow("USB Camera Test", frame)

    # Exit loop when 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()