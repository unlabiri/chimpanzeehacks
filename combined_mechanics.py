import os

# --- sound setup (catch / miss) ---
import pygame
try:
    pygame.mixer.init()
    catch_sound = pygame.mixer.Sound("pop.wav")
    miss_sound = pygame.mixer.Sound("error.wav")
    SOUND_ENABLED = True
except Exception as e:
    print("Audio disabled:", e)
    SOUND_ENABLED = False

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
GROUND_OFFSET = 70

# --- stage / banana settings ---
BANANA_SCALE = 0.08

# NOTE: This is the number of BANANAS COLLECTED to move to stage 2 (not misses).
BANANAS_TO_STAGE2 = 10

# Stage 1: two playable bananas
STAGE1_FAST_GRAVITY = 175.0      # px/s^2
STAGE1_GAP_MIN = 170             # vertical gap between bananas (pixels)
STAGE1_GAP_MAX = 230

# Stage 2: two fast bananas + one slower banana
STAGE2_FAST_GRAVITY = 200.0      # px/s^2
STAGE2_SLOW_GRAVITY = 160.0      # px/s^2
STAGE2_GAP_MIN = 160
STAGE2_GAP_MAX = 230

# --- load banana sprite(s) ---
banana_rgba = cv2.imread("banana.png", cv2.IMREAD_UNCHANGED)
use_sprite = banana_rgba is not None and banana_rgba.shape[2] == 4

epic_rgba = cv2.imread("epicbanana.png", cv2.IMREAD_UNCHANGED)
use_epic_sprite = epic_rgba is not None and epic_rgba.shape[2] == 4


def overlay_rgba_centered(bg_bgr, fg_rgba, cx, cy, scale):
    """
    Draw an RGBA sprite centered at (cx, cy) on a BGR background.
    Returns an approximate collision radius in pixels.
    """
    h, w = bg_bgr.shape[:2]
    fh, fw = fg_rgba.shape[:2]

    # Scale sprite
    nw = max(1, int(fw * scale))
    nh = max(1, int(fh * scale))
    fg = cv2.resize(fg_rgba, (nw, nh), interpolation=cv2.INTER_AREA)

    x0, y0 = int(cx - nw // 2), int(cy - nh // 2)
    x1, y1 = x0 + nw, y0 + nh

    # Off-screen: just return a sensible radius
    if x1 <= 0 or y1 <= 0 or x0 >= w or y0 >= h:
        return int(0.25 * max(nw, nh))

    # Clip to frame
    x0c, y0c = max(0, x0), max(0, y0)
    x1c, y1c = min(w, x1), min(h, y1)

    fx0, fy0 = x0c - x0, y0c - y0
    fx1, fy1 = fx0 + (x1c - x0c), fy0 + (y1c - y0c)

    fg_crop = fg[fy0:fy1, fx0:fx1]
    if fg_crop.size == 0:
        return int(0.25 * max(nw, nh))

    alpha = (fg_crop[:, :, 3].astype(np.float32) / 255.0)[:, :, None]
    fg_rgb = fg_crop[:, :, :3].astype(np.float32)
    roi = bg_bgr[y0c:y1c, x0c:x1c].astype(np.float32)

    blended = (alpha * fg_rgb + (1 - alpha) * roi).astype(np.uint8)
    bg_bgr[y0c:y1c, x0c:x1c] = blended

    # Reasonable collision radius for circular hit-test
    return int(0.25 * max(nw, nh))


# --- Mediapipe setup ---
mp_drawing = mp.solutions.drawing_utils
mp_holistic = mp.solutions.holistic


# --- banana class ---
class Banana:
    def __init__(self, W, gravity, label="", epic=False):
        self.W = W
        self.gravity = gravity
        self.label = label
        self.epic = epic  # special banana?
        self.x = random.randint(60, max(61, W - 60))
        self.y = -60
        self.vy = 0.0
        self.radius = 20  # updated when drawn

    def update(self, dt):
        self.vy += self.gravity * dt
        self.y += self.vy * dt

    def draw_and_update_radius(self, image):
        """Draw banana (sprite or circle) and update self.radius for collisions."""
        if self.epic and use_epic_sprite:
            r = overlay_rgba_centered(image, epic_rgba, int(self.x), int(self.y), scale=BANANA_SCALE)
            self.radius = r if r is not None else 20
        elif use_sprite:
            r = overlay_rgba_centered(image, banana_rgba, int(self.x), int(self.y), scale=BANANA_SCALE)
            self.radius = r if r is not None else 20
        else:
            # Fallback circle
            self.radius = 20 if not self.epic else 22
            cv2.circle(
                image,
                (int(self.x), int(self.y)),
                int(self.radius),
                (0, 255, 255) if not self.epic else (0, 165, 255),
                -1
            )

    def respawn_above(self, bananas, gap_min, gap_max):
        """Respawn this banana above the highest other banana with a vertical gap."""
        others = [b for b in bananas if b is not self]
        highest_y = min((b.y for b in others), default=-60)
        gap = random.randint(gap_min, gap_max)
        self.x = random.randint(60, max(61, self.W - 60))
        self.y = min(highest_y, -40) - gap
        self.vy = 0.0


# --- helpers to build stages ---
def make_stage1_bananas(W):
    b1 = Banana(W, STAGE1_FAST_GRAVITY, "s1_fast1")
    b2 = Banana(W, STAGE1_FAST_GRAVITY, "s1_fast2")
    gap = random.randint(STAGE1_GAP_MIN, STAGE1_GAP_MAX)
    b1.y = -60
    b2.y = b1.y - gap
    return [b1, b2]


def make_stage2_bananas(W):
    b1 = Banana(W, STAGE2_FAST_GRAVITY, "s2_fast1")
    b2 = Banana(W, STAGE2_FAST_GRAVITY, "s2_fast2")
    b3 = Banana(W, STAGE2_SLOW_GRAVITY, "s2_slow")
    gap1 = random.randint(STAGE2_GAP_MIN, STAGE2_GAP_MAX)
    gap2 = random.randint(STAGE2_GAP_MIN, STAGE2_GAP_MAX)
    b1.y = -60
    b2.y = b1.y - gap1
    b3.y = b2.y - gap2
    return [b1, b2, b3]


def hand_circle_from_landmarks(hand_landmarks, width, height):
    # WRIST=0, MIDDLE_MCP=9
    lm = hand_landmarks.landmark
    wx, wy = lm[0].x * width, lm[0].y * height
    mx, my = lm[9].x * width, lm[9].y * height
    cx = 0.5 * (wx + mx)
    cy = 0.5 * (wy + my)
    r = 0.8 * np.hypot(mx - wx, my - wy)
    r = max(12, r)
    return cx, cy, r


# --- camera setup ---
cap = cv2.VideoCapture(CAM_INDEX)
if not cap.isOpened():
    raise RuntimeError(f"Could not open camera index {CAM_INDEX}. Try 0/1/2 and check permissions.")

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1080)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
cv2.namedWindow(WIN_NAME, cv2.WINDOW_AUTOSIZE)


def reset_game(W):
    """Reset counters and return a fresh stage-1 setup."""
    collected = 0
    missed = 0
    stage = 1
    bananas = make_stage1_bananas(W)
    # Epic scheduling (stage 2)
    normal_since_epic = 0
    epic_next_at = random.randint(3, 5)
    return bananas, stage, collected, missed, normal_since_epic, epic_next_at


with mp_holistic.Holistic(
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
) as holistic:

    ok, first_frame = cap.read()
    if not ok:
        raise RuntimeError("Failed to read initial frame from camera.")
    H, W = first_frame.shape[:2]

    bananas, stage, collected, missed, normal_since_epic, epic_next_at = reset_game(W)
    last_t = time.time()

    while True:
        ok, frame = cap.read()
        if not ok:
            print("Failed to read frame. Is another app using the camera?")
            break

        # Mirror so it feels natural
        frame = cv2.flip(frame, 1)
        H, W = frame.shape[:2]
        ground_y = H - GROUND_OFFSET

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

        # Build hand hit-circles
        hand_circles = []
        if results.left_hand_landmarks is not None:
            hand_circles.append(hand_circle_from_landmarks(results.left_hand_landmarks, W, H))
        if results.right_hand_landmarks is not None:
            hand_circles.append(hand_circle_from_landmarks(results.right_hand_landmarks, W, H))

        # Optional: visualize hand circles
        for (hc_x, hc_y, hc_r) in hand_circles:
            cv2.circle(image, (int(hc_x), int(hc_y)), int(hc_r), (0, 180, 255), 2)

        # Ground guide
        cv2.line(image, (0, int(ground_y)), (W, int(ground_y)), (60, 60, 60), 2)

        # --- timing ---
        now = time.time()
        dt = max(0.001, min(0.05, now - last_t))
        last_t = now

        # --- stage transition (based on collected) ---
        if stage == 1 and collected >= BANANAS_TO_STAGE2:
            stage = 2
            bananas = make_stage2_bananas(W)
            normal_since_epic = 0
            epic_next_at = random.randint(3, 5)

        # --- update positions ---
        for b in bananas:
            b.update(dt)

        # --- draw bananas & update radius ---
        for b in bananas:
            b.draw_and_update_radius(image)

        # --- collision + ground logic ---
        for b in bananas:
            # Hand collision
            caught = False
            for (hc_x, hc_y, hc_r) in hand_circles:
                d = np.hypot(b.x - hc_x, b.y - hc_y)
                if d <= (b.radius + hc_r):
                    caught = True
                    break

            if caught:
                # play catch sfx
                if SOUND_ENABLED and pygame.mixer.get_init():
                    catch_sound.play()

                # Score: epic worth 2 (you can change to 3 if you want)
                collected += 2 if b.epic else 1

                # Respawn above stack
                if stage == 1:
                    b.respawn_above(bananas, STAGE1_GAP_MIN, STAGE1_GAP_MAX)
                else:
                    b.respawn_above(bananas, STAGE2_GAP_MIN, STAGE2_GAP_MAX)

                    # Epic scheduling on respawn (only count normals)
                    if b.epic:
                        # caught an epic: reset epic cadence
                        b.epic = False
                        normal_since_epic = 0
                        epic_next_at = random.randint(3, 5)
                    else:
                        normal_since_epic += 1
                        if normal_since_epic >= epic_next_at:
                            b.epic = True
                            normal_since_epic = 0
                            epic_next_at = random.randint(3, 5)

                continue  # done with this banana

            # Miss (hits ground)
            if (b.y + b.radius) >= ground_y:
                missed += 1

                # play miss sfx
                if SOUND_ENABLED and pygame.mixer.get_init():
                    miss_sound.play()

                if stage == 1:
                    b.respawn_above(bananas, STAGE1_GAP_MIN, STAGE1_GAP_MAX)
                else:
                    b.respawn_above(bananas, STAGE2_GAP_MIN, STAGE2_GAP_MAX)

                    # Same epic cadence rules apply on any respawn
                    if b.epic:
                        b.epic = False
                        normal_since_epic = 0
                        epic_next_at = random.randint(3, 5)
                    else:
                        normal_since_epic += 1
                        if normal_since_epic >= epic_next_at:
                            b.epic = True
                            normal_since_epic = 0
                            epic_next_at = random.randint(3, 5)

        # --- HUD ---
        info = f"Stage: {stage} | On screen: {len(bananas)} | Collected: {collected} | Missed: {missed}"
        cv2.putText(image, info, (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2, cv2.LINE_AA)

        cv2.imshow(WIN_NAME, image)
        key = cv2.waitKey(16) & 0xFF

        if missed > 2:
            print("Game over — too many missed bananas!")
            break

        if key == ord('q'):
            break
        elif key == ord(' '):
            bananas, stage, collected, missed, normal_since_epic, epic_next_at = reset_game(W)

cap.release()
cv2.destroyAllWindows()
