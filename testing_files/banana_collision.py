import os
# Quiet TF/mediapipe logs (optional)
os.environ["GLOG_minloglevel"] = "2"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import cv2
import mediapipe as mp
import numpy as np
import time
import random

# === Settings you can tweak ===
CAM_INDEX = 0
BANANA_SCALE = 0.08      # relative sprite scale (tune if your sprite is huge/small)
GRAVITY = 200.0          # pixels / s^2
GROUND_OFFSET = 70       # pixels up from bottom for the ground line

# === Load optional banana sprite (RGBA). Fallback is a circle. ===
banana_rgba = cv2.imread("banana.png", cv2.IMREAD_UNCHANGED)
use_sprite = banana_rgba is not None and banana_rgba.shape[2] == 4

def overlay_rgba_centered(bg_bgr, fg_rgba, cx, cy, scale):
    """
    Draw an RGBA sprite centered at (cx, cy) on a BGR background.
    Returns the approximate 'banana radius' in pixels for collisions.
    """
    h, w = bg_bgr.shape[:2]
    fh, fw = fg_rgba.shape[:2]
    # Scale sprite
    nw = max(1, int(fw * scale))
    nh = max(1, int(fh * scale))
    fg = cv2.resize(fg_rgba, (nw, nh), interpolation=cv2.INTER_AREA)

    x0, y0 = int(cx - nw // 2), int(cy - nh // 2)
    x1, y1 = x0 + nw, y0 + nh

    # If completely off-screen, just return a radius estimate and skip
    if x1 <= 0 or y1 <= 0 or x0 >= w or y0 >= h:
        return int(0.5 * max(nw, nh) * 0.5)

    # Clip to frame
    x0c, y0c = max(0, x0), max(0, y0)
    x1c, y1c = min(w, x1), min(h, y1)

    fx0, fy0 = x0c - x0, y0c - y0
    fx1, fy1 = fx0 + (x1c - x0c), fy0 + (y1c - y0c)

    fg_crop = fg[fy0:fy1, fx0:fx1]
    if fg_crop.size == 0:
        return int(0.5 * max(nw, nh) * 0.5)

    alpha = (fg_crop[:, :, 3].astype(np.float32) / 255.0)[:, :, None]
    fg_rgb = fg_crop[:, :, :3].astype(np.float32)
    roi = bg_bgr[y0c:y1c, x0c:x1c].astype(np.float32)

    blended = (alpha * fg_rgb + (1 - alpha) * roi).astype(np.uint8)
    bg_bgr[y0c:y1c, x0c:x1c] = blended

    # Return a collision radius (roughly half the max dimension * 0.5 to be a bit conservative)
    return int(0.5 * max(nw, nh) * 0.5)

# === Mediapipe setup (full-body + hands drawn so you can see yourself) ===
mp_drawing  = mp.solutions.drawing_utils
mp_holistic = mp.solutions.holistic

# === Camera ===
cap = cv2.VideoCapture(CAM_INDEX)
if not cap.isOpened():
    raise RuntimeError(f"Could not open camera index {CAM_INDEX}. Try 0/1/2 and check permissions.")

cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1080)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

# === Banana state and counters ===
x, y = 320.0, 60.0
vy = 0.0
collected = 0
missed = 0

def reset_banana(frame_w):
    """Respawn banana near top with random x and zero vertical speed."""
    global x, y, vy
    x = random.randint(60, max(61, frame_w - 60))
    y = 60.0
    vy = 0.0

def hand_circle_from_landmarks(hand_landmarks, width, height):
    """
    Build a simple collision circle for a hand.
    Center: average of wrist (0) and middle MCP (9).
    Radius: distance(wrist, middle_mcp) * 0.8 (tunable).
    Returns (cx, cy, r) in pixels.
    """
    # Indexes from MediaPipe Hands spec
    WRIST = 0
    MIDDLE_MCP = 9

    lm = hand_landmarks.landmark
    wx, wy = lm[WRIST].x * width, lm[WRIST].y * height
    mx, my = lm[MIDDLE_MCP].x * width, lm[MIDDLE_MCP].y * height

    cx = 0.5 * (wx + mx)
    cy = 0.5 * (wy + my)
    r  = 0.8 * np.hypot(mx - wx, my - wy)  # scale factor to cover palm area
    r  = max(12, r)  # a reasonable minimum so it’s not too small
    return cx, cy, r

last_t = time.time()
win = "Banana Catch — q=quit, space=reset"
cv2.namedWindow(win, cv2.WINDOW_AUTOSIZE)

with mp_holistic.Holistic(min_detection_confidence=0.5,
                          min_tracking_confidence=0.5) as holistic:

    # Prime banana position
    reset_banana(640)

    while True:
        ok, frame = cap.read()
        if not ok:
            print("Failed to read frame. Is another app using the camera?")
            break
        frame = cv2.flip(frame, 1)
        H, W = frame.shape[:2]
        ground_y = H - GROUND_OFFSET

        # --- Run Mediapipe Holistic and keep the landmarks visible ---
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = holistic.process(rgb)
        image = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

        # Draw pose and hands so you can see full-body tracking
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

        # --- Physics step (gravity only) ---
        now = time.time()
        dt = max(0.001, min(0.05, now - last_t))  # clamp dt for stability
        last_t = now

        vy += GRAVITY * dt
        y  += vy * dt

        # --- Draw ground ---
        cv2.line(image, (0, int(ground_y)), (W, int(ground_y)), (60, 60, 60), 2)

        # --- Draw banana & get its radius for collision ---
        if use_sprite:
            banana_radius = overlay_rgba_centered(image, banana_rgba, int(x), int(y), scale=BANANA_SCALE)
        else:
            banana_radius = 28  # fallback circle radius
            cv2.circle(image, (int(x), int(y)), banana_radius, (0,255,255), -1)

        # --- Build hand collision circles (either hand can catch) ---
        hand_circles = []
        if results.left_hand_landmarks is not None:
            hand_circles.append(hand_circle_from_landmarks(results.left_hand_landmarks, W, H))
        if results.right_hand_landmarks is not None:
            hand_circles.append(hand_circle_from_landmarks(results.right_hand_landmarks, W, H))

        # Optional: visualize the hand circles (helps you tune)
        for (hc_x, hc_y, hc_r) in hand_circles:
            cv2.circle(image, (int(hc_x), int(hc_y)), int(hc_r), (0, 180, 255), 2)

        # --- Check catch (collision banana↔hand) ---
        caught = False
        for (hc_x, hc_y, hc_r) in hand_circles:
            d = np.hypot(x - hc_x, y - hc_y)
            if d <= (banana_radius + hc_r):
                caught = True
                break

        if caught:
            collected += 1
            reset_banana(W)

        # --- Check ground (missed) ---
        elif (y + banana_radius) >= ground_y:
            missed += 1
            if missed == 3:
                break
            reset_banana(W)

        # --- HUD text ---
        cv2.putText(image, f"Collected: {collected}", (10, 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (30, 200, 30), 2, cv2.LINE_AA)
        cv2.putText(image, f"Missed:    {missed}", (10, 56),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (40, 40, 220), 2, cv2.LINE_AA)
        cv2.putText(image, "q = quit, space = reset banana", (10, H - 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (180, 180, 180), 1, cv2.LINE_AA)

        # --- Show ---
        cv2.imshow(win, image)
        key = cv2.waitKey(16) & 0xFF
        if key == ord('q'):
            break
        elif key == ord(' '):
            reset_banana(W)

cap.release()
cv2.destroyAllWindows()
