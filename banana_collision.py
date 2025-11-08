import os
# (optional) quiet logs
os.environ["GLOG_minloglevel"] = "2"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import cv2
import mediapipe as mp
import numpy as np
import time
import random

# --- settings ---
CAM_INDEX = 0           # try 0, if no video try 1 or 2
BANANA_SCALE = 0.08  # sprite size
GRAVITY = 900.0         # pixels / s^2
GROUND_OFFSET = 40      # pixels from bottom for "ground" line

# --- load banana sprite (RGBA) or fall back to a circle ---
banana_rgba = cv2.imread("banana.png", cv2.IMREAD_UNCHANGED)
use_sprite = banana_rgba is not None and banana_rgba.shape[2] == 4

def overlay_rgba(bg_bgr, fg_rgba, cx, cy, scale=1.0):
    """Overlay RGBA sprite centered at (cx, cy) onto BGR frame."""
    h, w = bg_bgr.shape[:2]
    fh, fw = fg_rgba.shape[:2]
    nw, nh = max(1, int(fw * scale)), max(1, int(fh * scale))
    fg = cv2.resize(fg_rgba, (nw, nh), interpolation=cv2.INTER_AREA)

    x0, y0 = int(cx - nw // 2), int(cy - nh // 2)
    x1, y1 = x0 + nw, y0 + nh

    # clip to frame
    if x1 <= 0 or y1 <= 0 or x0 >= w or y0 >= h:
        return int(min(nw, nh) * 0.35)  # off-screen; still return approx radius
    x0c, y0c = max(0, x0), max(0, y0)
    x1c, y1c = min(w, x1), min(h, y1)

    fx0, fy0 = x0c - x0, y0c - y0
    fx1, fy1 = fx0 + (x1c - x0c), fy0 + (y1c - y0c)
    fg_crop = fg[fy0:fy1, fx0:fx1]
    if fg_crop.size == 0:
        return int(min(nw, nh) * 0.35)

    alpha = (fg_crop[:, :, 3] / 255.0)[:, :, None]
    fg_rgb = fg_crop[:, :, :3].astype(np.float32)
    roi = bg_bgr[y0c:y1c, x0c:x1c].astype(np.float32)
    blended = (alpha * fg_rgb + (1 - alpha) * roi).astype(np.uint8)
    bg_bgr[y0c:y1c, x0c:x1c] = blended

    return int(min(nw, nh) * 0.35)  # approx radius (not used for collisions here)

# --- mediapipe (optional: still draws landmarks; comment out if you don’t want them) ---
mp_drawing  = mp.solutions.drawing_utils
mp_holistic = mp.solutions.holistic

# --- camera ---
cap = cv2.VideoCapture(CAM_INDEX)
if not cap.isOpened():
    raise RuntimeError(f"Could not open camera index {CAM_INDEX}. Try 0/1/2 and check permissions.")

cap.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

# --- banana state ---
vy = 0.0
x = 320
y = 60

def reset_banana(frame_w):
    """Respawn banana at top with random x; zero vertical speed."""
    global x, y, vy
    x = random.randint(60, max(61, frame_w - 60))
    y = 60
    vy = 0.0

last_t = time.time()
win = "Banana Falling (no collisions) — q=quit, space=reset"
cv2.namedWindow(win, cv2.WINDOW_AUTOSIZE)

with mp_holistic.Holistic(min_detection_confidence=0.5,
                          min_tracking_confidence=0.5) as holistic:

    reset_banana(640)

    while True:
        ok, frame = cap.read()
        if not ok:
            print("Failed to read frame. Is another app using the camera?")
            break

        H, W = frame.shape[:2]
        ground_y = H - GROUND_OFFSET

        # (optional) run holistic and draw landmarks (for reference)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = holistic.process(rgb)
        image = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

        if results.pose_landmarks is not None:
            mp_drawing.draw_landmarks(
                image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS,
                mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=3),
                mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2)
            )
        if results.left_hand_landmarks is not None:
            mp_drawing.draw_landmarks(
                image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
                mp_drawing.DrawingSpec(color=(121,22,76), thickness=2, circle_radius=3),
                mp_drawing.DrawingSpec(color=(121,44,250), thickness=2, circle_radius=2)
            )
        if results.right_hand_landmarks is not None:
            mp_drawing.draw_landmarks(
                image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
                mp_drawing.DrawingSpec(color=(80,22,10), thickness=2, circle_radius=3),
                mp_drawing.DrawingSpec(color=(80,44,121), thickness=2, circle_radius=2)
            )

        # --- physics update (gravity only) ---
        now = time.time()
        dt = max(0.001, min(0.05, now - last_t))  # clamp dt for stability
        last_t = now

        vy += GRAVITY * dt
        y  += vy * dt

        # --- draw ground and banana ---
        cv2.line(image, (0, ground_y), (W, ground_y), (60, 60, 60), 2)

        if use_sprite:
            overlay_rgba(image, banana_rgba, int(x), int(y), scale=BANANA_SCALE)
        else:
            # yellow circle fallback
            radius = 28
            cv2.circle(image, (int(x), int(y)), radius, (0,255,255), -1)

        # if banana reaches ground, reset (still no collisions)
        # we use 20px as a rough "sprite half-height" pad
        if y >= ground_y - 20:
            reset_banana(W)

        cv2.imshow(win, image)
        key = cv2.waitKey(16) & 0xFF
        if key == ord('q'):
            break
        elif key == ord(' '):
            reset_banana(W)

cap.release()
cv2.destroyAllWindows()
