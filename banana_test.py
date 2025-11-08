import cv2
import numpy as np

# Try to load a banana image
banana = cv2.imread('banana.png', cv2.IMREAD_UNCHANGED)

# Create a blank background
cap = cv2.VideoCapture(0)

ok, frame = cap.read()

if banana is not None:
    # Resize and overlay banana
    banana = cv2.resize(banana, (100, 100))  # smaller
    x, y = 270, 190  # center position
    h, w = banana.shape[:2]
    if banana.shape[2] == 4:  # if RGBA (has alpha)
        alpha = banana[:, :, 3] / 255.0
        for c in range(3):
            frame[y:y+h, x:x+w, c] = (alpha * banana[:, :, c] +
                                      (1 - alpha) * frame[y:y+h, x:x+w, c])
    else:
        frame[y:y+h, x:x+w] = banana
else:
    # Fallback: draw a yellow circle instead
    cv2.circle(frame, (320, 240), 50, (0, 255, 255), -1)

# Show the frame
cv2.imshow("Banana Test", frame)

# Wait until user presses 'q' to quit
while True:
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()
