import os
os.environ["GLOG_minloglevel"] = "2"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import cv2
import mediapipe as mp
import numpy as np
import time
import random

# --- camera + window settings ---
CAM_INDEX = 0
WIN_NAME = "Banana Falling — q=quit, space=reset"

# --- stage / banana settings ---
BANANA_SCALE = 0.08
BANANAS_TO_STAGE2 = 10  # after this many misses, go to stage 2

# Stage 1: two playable bananas
STAGE1_FAST_GRAVITY = 175.0      # px/s^2
STAGE1_GAP_MIN = 170             # vertical gap between bananas (pixels)
STAGE1_GAP_MAX = 230

# Stage 2: two fast bananas + one slower banana
STAGE2_FAST_GRAVITY = 200.0      # px/s^2 (a bit faster, still playable)
STAGE2_SLOW_GRAVITY = 160.0      # clearly slower one
STAGE2_GAP_MIN = 160
STAGE2_GAP_MAX = 230

# --- load banana sprite (RGBA) or fallback circle ---
banana_rgba = cv2.imread("banana.png", cv2.IMREAD_UNCHANGED)
use_sprite = banana_rgba is not None and banana_rgba.shape[2] == 4

epic_rgba = cv2.imread("epicbanana.png", cv2.IMREAD_UNCHANGED)
use_epic_sprite = epic_rgba is not None and epic_rgba.shape[2] == 4


def overlay_rgba(bg_bgr, fg_rgba, cx, cy, scale=1.0):
    """Overlay RGBA sprite centered at (cx, cy) onto BGR frame."""
    h, w = bg_bgr.shape[:2]
    fh, fw = fg_rgba.shape[:2]

    # resize sprite
    nw = max(1, int(fw * scale))
    nh = max(1, int(fh * scale))
    fg = cv2.resize(fg_rgba, (nw, nh), interpolation=cv2.INTER_AREA)

    # top-left / bottom-right in background coords
    x0 = int(cx - nw // 2)
    y0 = int(cy - nh // 2)
    x1 = x0 + nw
    y1 = y0 + nh

    # completely off-screen
    if x1 <= 0 or y1 <= 0 or x0 >= w or y0 >= h:
        return

    # clamp to visible region
    x0c, y0c = max(0, x0), max(0, y0)
    x1c, y1c = min(w, x1), min(h, y1)

    # corresponding crop in the sprite
    fx0, fy0 = x0c - x0, y0c - y0
    fx1, fy1 = fx0 + (x1c - x0c), fy0 + (y1c - y0c)

    fg_crop = fg[fy0:fy1, fx0:fx1]
    if fg_crop.size == 0:
        return

    alpha = (fg_crop[:, :, 3] / 255.0)[:, :, None]          # (h,w,1)
    fg_rgb = fg_crop[:, :, :3].astype(np.float32)           # (h,w,3)
    roi = bg_bgr[y0c:y1c, x0c:x1c].astype(np.float32)       # (h,w,3)

    blended = (alpha * fg_rgb + (1.0 - alpha) * roi).astype(np.uint8)
    bg_bgr[y0c:y1c, x0c:x1c] = blended


# --- Mediapipe setup ---
mp_drawing = mp.solutions.drawing_utils
mp_holistic = mp.solutions.holistic

# --- banana class ---
class Banana:
    def __init__(self, W, gravity, label="", epic=False):
        self.W = W
        self.gravity = gravity
        self.label = label
        self.epic = epic        # special banana?
        self.x = random.randint(60, max(61, W - 60))
        self.y = -60
        self.vy = 0.0

    def update(self, dt):
        self.vy += self.gravity * dt
        self.y += self.vy * dt

    def respawn_above(self, bananas, gap_min, gap_max):
        """Respawn this banana above the highest other banana with a gap."""
        # find highest (smallest y) among others; if none, use -60
        others = [b for b in bananas if b is not self]
        if others:
            highest_y = min(b.y for b in others)
        else:
            highest_y = -60

        gap = random.randint(gap_min, gap_max)
        self.x = random.randint(60, max(61, self.W - 60))
        self.y = min(highest_y, -40) - gap
        self.vy = 0.0


# --- camera setup ---
cap = cv2.VideoCapture(CAM_INDEX)
if not cap.isOpened():
    raise RuntimeError(f"Could not open camera index {CAM_INDEX}. Try 0/1/2 and check permissions.")

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
cv2.namedWindow(WIN_NAME, cv2.WINDOW_AUTOSIZE)


def make_stage1_bananas(W):
    """Two bananas with nice vertical spacing."""
    b1 = Banana(W, STAGE1_FAST_GRAVITY, "s1_fast1")
    b2 = Banana(W, STAGE1_FAST_GRAVITY, "s1_fast2")

    # stack upward so they enter one after another
    gap = random.randint(STAGE1_GAP_MIN, STAGE1_GAP_MAX)
    b1.y = -60
    b2.y = b1.y - gap

    return [b1, b2]


def make_stage2_bananas(W):
    """Two fast bananas + one slower banana, all spaced."""
    b1 = Banana(W, STAGE2_FAST_GRAVITY, "s2_fast1")
    b2 = Banana(W, STAGE2_FAST_GRAVITY, "s2_fast2")
    b3 = Banana(W, STAGE2_SLOW_GRAVITY, "s2_slow")

    gap1 = random.randint(STAGE2_GAP_MIN, STAGE2_GAP_MAX)
    gap2 = random.randint(STAGE2_GAP_MIN, STAGE2_GAP_MAX)

    b1.y = -60
    b2.y = b1.y - gap1
    b3.y = b2.y - gap2

    return [b1, b2, b3]


def reset_game(W):
    bananas_passed = 0
    stage = 1
    bananas = make_stage1_bananas(W)
    return bananas, stage, bananas_passed


with mp_holistic.Holistic(
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
) as holistic:

    ok, first_frame = cap.read()
    if not ok:
        raise RuntimeError("Failed to read initial frame from camera.")
    H, W = first_frame.shape[:2]

    bananas, stage, bananas_passed = reset_game(W)
    last_t = time.time()
    # Epic banana scheduling (stage 2 only)
    epic_next_at = random.randint(3, 5)  # after this many normal bananas
    normal_since_epic = 0

    while True:
        ok, frame = cap.read()
        if not ok:
            print("Failed to read frame. Is another app using the camera?")
            break

        H, W = frame.shape[:2]

        # --- Mediapipe (visual only) ---
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = holistic.process(rgb)
        image = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

        if results.pose_landmarks:
            mp_drawing.draw_landmarks(
                image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS,
                mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=3),
                mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2)
            )
        if results.left_hand_landmarks:
            mp_drawing.draw_landmarks(
                image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
                mp_drawing.DrawingSpec(color=(121, 22, 76), thickness=2, circle_radius=3),
                mp_drawing.DrawingSpec(color=(121, 44, 250), thickness=2, circle_radius=2)
            )
        if results.right_hand_landmarks:
            mp_drawing.draw_landmarks(
                image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
                mp_drawing.DrawingSpec(color=(80, 22, 10), thickness=2, circle_radius=3),
                mp_drawing.DrawingSpec(color=(80, 44, 121), thickness=2, circle_radius=2)
            )

        # --- timing ---
        now = time.time()
        dt = max(0.001, min(0.05, now - last_t))
        last_t = now

        # --- stage transition ---
        if stage == 1 and bananas_passed >= BANANAS_TO_STAGE2:
            stage = 2
            bananas = make_stage2_bananas(W)
            normal_since_epic = 0
            epic_next_at = random.randint(3, 5)

        # --- update + draw bananas ---
        for b in bananas:
            b.update(dt)

            if b.epic and use_epic_sprite:
                overlay_rgba(image, epic_rgba, int(b.x), int(b.y), scale=BANANA_SCALE)
            elif use_sprite:
                overlay_rgba(image, banana_rgba, int(b.x), int(b.y), scale=BANANA_SCALE)
            else:
                cv2.circle(image, (int(b.x), int(b.y)), 20,
                           (0, 255, 255) if not b.epic else (0, 165, 255), -1)

        # --- check for misses & respawn with spacing ---
        if stage == 1:
            for b in bananas:
                if b.y >= H:
                    bananas_passed += 1
                    b.respawn_above(bananas, STAGE1_GAP_MIN, STAGE1_GAP_MAX)
        else:  # stage 2
            for b in bananas:
                if b.y >= H:
                    if b.epic:
                        # epic banana counts as 3
                        bananas_passed += 3
                        b.epic = False
                        normal_since_epic = 0
                        epic_next_at = random.randint(3, 5)
                    else:
                        bananas_passed += 1
                        normal_since_epic += 1

                    # respawn above stack
                    b.respawn_above(bananas, STAGE2_GAP_MIN, STAGE2_GAP_MAX)

                    # decide if this respawned banana becomes the next epic
                    if not b.epic and normal_since_epic >= epic_next_at:
                        b.epic = True
                        normal_since_epic = 0
                        epic_next_at = random.randint(3, 5)

        # --- HUD text ---
        info = f"Stage: {stage} | On screen: {len(bananas)} | Missed: {bananas_passed}"
        cv2.putText(image, info, (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2, cv2.LINE_AA)

        cv2.imshow(WIN_NAME, image)
        key = cv2.waitKey(16) & 0xFF

        if key == ord('q'):
            break
        elif key == ord(' '):
            bananas, stage, bananas_passed = reset_game(W)

cap.release()
cv2.destroyAllWindows()
