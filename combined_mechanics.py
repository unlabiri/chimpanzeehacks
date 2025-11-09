import os
import pygame
import cv2
import mediapipe as mp
import numpy as np
import time
import random

try:
    pygame.mixer.init()
    catch_sound = pygame.mixer.Sound("audio/pop.wav")
    miss_sound = pygame.mixer.Sound("audio/error.wav")
    bomb_sound = pygame.mixer.Sound("audio/bomb.wav")
    
    SOUND_ENABLED = True
except Exception as e:
    print("Audio disabled:", e)
    SOUND_ENABLED = False

os.environ["GLOG_minloglevel"] = "2"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"


# --- camera + window settings ---
CAM_INDEX = 0
WIN_NAME = "Banana Falling — q=quit, space=reset"
GROUND_OFFSET = 70

# --- stage / banana settings ---
BANANA_SCALE = 0.08
BANANAS_TO_STAGE2 = 10    # collected bananas to go to stage 2
BANANAS_TO_STAGE4 = 25    # collected bananas to go to stage 4 (tweak as you like)
MISSED_LIMIT = 3          # misses before GAME OVER

# Stage 1
STAGE1_FAST_GRAVITY = 175.0
STAGE1_GAP_MIN = 170
STAGE1_GAP_MAX = 230

# Stage 2
STAGE2_FAST_GRAVITY = 200.0
STAGE2_SLOW_GRAVITY = 160.0
STAGE2_GAP_MIN = 160
STAGE2_GAP_MAX = 230

# Bomb / Stage 4
BOMB_SCALE = 0.09
STAGE4_BOMB_GRAVITY = 230.0
BOMB_GAP_MIN = 150
BOMB_GAP_MAX = 220
NUM_STAGE4_BOMBS = 2      # how many bombs in stage 4


# --- load banana sprite(s) ---
banana_rgba = cv2.imread("img/banana.png", cv2.IMREAD_UNCHANGED)
use_sprite = banana_rgba is not None and banana_rgba.shape[2] == 4

epic_rgba = cv2.imread("img/epicbanana.png", cv2.IMREAD_UNCHANGED)
use_epic_sprite = epic_rgba is not None and epic_rgba.shape[2] == 4

# --- load bomb sprite (optional) ---
bomb_rgba = cv2.imread("img/coconut.png", cv2.IMREAD_UNCHANGED)
use_bomb_sprite = bomb_rgba is not None and bomb_rgba.shape[2] == 4

# --- load horse sprite for GAME OVER screen ---
horse_rgba = cv2.imread("img/horse.png", cv2.IMREAD_UNCHANGED)
use_horse_sprite = horse_rgba is not None and horse_rgba.shape[2] == 4

start_img = cv2.imread("img/Start.png", cv2.IMREAD_UNCHANGED)
title_img = cv2.imread("img/Title.png", cv2.IMREAD_UNCHANGED)


def overlay_rgba_centered(bg_bgr, fg_rgba, cx, cy, scale):
    """Draw RGBA centered on bg_bgr and return an approximate radius."""
    h, w = bg_bgr.shape[:2]
    fh, fw = fg_rgba.shape[:2]

    # scale
    nw = max(1, int(fw * scale))
    nh = max(1, int(fh * scale))
    fg = cv2.resize(fg_rgba, (nw, nh), interpolation=cv2.INTER_AREA)

    # position
    x0, y0 = int(cx - nw // 2), int(cy - nh // 2)
    x1, y1 = x0 + nw, y0 + nh

    # off screen
    if x1 <= 0 or y1 <= 0 or x0 >= w or y0 >= h:
        return int(0.25 * max(nw, nh))

    # clamp to screen
    x0c, y0c = max(0, x0), max(0, y0)
    x1c, y1c = min(w, x1), min(h, y1)

    # corresponding crop in fg
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

    return int(0.25 * max(nw, nh))


# --- mediapipe ---
mp_drawing = mp.solutions.drawing_utils
mp_holistic = mp.solutions.holistic


class Banana:
    def __init__(self, W, gravity, label="", epic=False):
        self.W = W
        self.gravity = gravity
        self.label = label
        self.epic = epic
        self.x = random.randint(60, max(61, W - 60))
        self.y = -60
        self.vy = 0.0
        self.radius = 20

    def update(self, dt):
        self.vy += self.gravity * dt
        self.y += self.vy * dt

    def draw_and_update_radius(self, image):
        if self.epic and use_epic_sprite:
            r = overlay_rgba_centered(image, epic_rgba,
                                      int(self.x), int(self.y),
                                      scale=BANANA_SCALE)
            self.radius = r if r is not None else 22
        elif use_sprite:
            r = overlay_rgba_centered(image, banana_rgba,
                                      int(self.x), int(self.y),
                                      scale=BANANA_SCALE)
            self.radius = r if r is not None else 20
        else:
            self.radius = 22 if self.epic else 20
            cv2.circle(
                image,
                (int(self.x), int(self.y)),
                int(self.radius),
                (0, 165, 255) if self.epic else (0, 255, 255),
                -1
            )

    def respawn_above(self, bananas, gap_min, gap_max):
        others = [b for b in bananas if b is not self]
        highest_y = min((b.y for b in others), default=-60)
        gap = random.randint(gap_min, gap_max)
        self.x = random.randint(60, max(61, self.W - 60))
        self.y = min(highest_y, -40) - gap
        self.vy = 0.0


class Bomb:
    """Stage 4: falling bomb that causes GAME OVER if it hits head/torso/hands."""
    def __init__(self, W, gravity, label="bomb"):
        self.W = W
        self.gravity = gravity
        self.label = label
        self.x = random.randint(60, max(61, W - 60))
        self.y = -60
        self.vy = 0.0
        self.radius = 24  # will be updated by draw

    def update(self, dt):
        self.vy += self.gravity * dt
        self.y += self.vy * dt

    def draw_and_update_radius(self, image):
        if use_bomb_sprite:
            r = overlay_rgba_centered(image, bomb_rgba,
                                      int(self.x), int(self.y),
                                      scale=BOMB_SCALE)
            self.radius = r if r is not None else 24
        else:
            # fallback: dark circle
            self.radius = 24
            cv2.circle(
                image,
                (int(self.x), int(self.y)),
                int(self.radius),
                (0, 0, 180),
                -1
            )

    def respawn_above(self, bombs, gap_min, gap_max):
        others = [b for b in bombs if b is not self]
        highest_y = min((b.y for b in others), default=-60)
        gap = random.randint(gap_min, gap_max)
        self.x = random.randint(60, max(61, self.W - 60))
        self.y = min(highest_y, -40) - gap
        self.vy = 0.0


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
    lm = hand_landmarks.landmark
    wx, wy = lm[0].x * width, lm[0].y * height
    mx, my = lm[9].x * width, lm[9].y * height
    cx = 0.5 * (wx + mx)
    cy = 0.5 * (wy + my)
    r = 0.8 * np.hypot(mx - wx, my - wy)
    r = max(12, r)
    return cx, cy, r


def get_stage4_hit_areas(results, W, H, vis_thresh=0.5):
    """
    Hit areas for Stage 4 bombs:
      - head: circle
      - torso: rectangle
      - both hands: circles
    No forearms or legs.
    Returns list of:
      ("circle", cx, cy, r) or ("rect", x0, y0, x1, y1)
    """
    areas = []

    # Pose landmarks for head & torso
    if results.pose_landmarks:
        lm = results.pose_landmarks.landmark
        PL = mp_holistic.PoseLandmark

        # Head circle around nose; size based on ears if available
        nose = lm[PL.NOSE.value]
        l_ear = lm[PL.LEFT_EAR.value]
        r_ear = lm[PL.RIGHT_EAR.value]

        if nose.visibility > vis_thresh:
            cx = int(nose.x * W)
            cy = int(nose.y * H)

            if l_ear.visibility > vis_thresh and r_ear.visibility > vis_thresh:
                ex1 = l_ear.x * W
                ex2 = r_ear.x * W
                base_r = int(abs(ex2 - ex1) * 0.6)
            else:
                base_r = 40

            head_r = max(25, base_r)
            areas.append(("circle", cx, cy, head_r))

        # Torso rectangle from shoulders to hips
        ls = lm[PL.LEFT_SHOULDER.value]
        rs = lm[PL.RIGHT_SHOULDER.value]
        lh = lm[PL.LEFT_HIP.value]
        rh = lm[PL.RIGHT_HIP.value]

        if (ls.visibility > vis_thresh and rs.visibility > vis_thresh and
            lh.visibility > vis_thresh and rh.visibility > vis_thresh):

            x_min = int(min(ls.x, rs.x) * W)
            x_max = int(max(ls.x, rs.x) * W)
            torso_w = x_max - x_min

            # pad a bit; keep this region to torso only
            x_min -= int(torso_w * 0.20)
            x_max += int(torso_w * 0.20)

            y_min = int(min(ls.y, rs.y) * H)
            y_max = int(max(lh.y, rh.y) * H)

            areas.append(("rect", x_min, y_min, x_max, y_max))

    # Hands: from hand landmarks only (already no forearms)
    if results.left_hand_landmarks:
        cx, cy, r = hand_circle_from_landmarks(results.left_hand_landmarks, W, H)
        areas.append(("circle", int(cx), int(cy), int(r)))

    if results.right_hand_landmarks:
        cx, cy, r = hand_circle_from_landmarks(results.right_hand_landmarks, W, H)
        areas.append(("circle", int(cx), int(cy), int(r)))

    return areas


def bomb_hits_player(bomb, hit_areas):
    """Check if bomb circle overlaps any Stage 4 hit area."""
    bx, by, br = bomb.x, bomb.y, bomb.radius

    for area in hit_areas:
        shape = area[0]

        if shape == "circle":
            _, cx, cy, r = area
            d = np.hypot(bx - cx, by - cy)
            if d <= br + r:
                return True

        elif shape == "rect":
            _, x0, y0, x1, y1 = area
            closest_x = np.clip(bx, x0, x1)
            closest_y = np.clip(by, y0, y1)
            d = np.hypot(bx - closest_x, by - closest_y)
            if d <= br:
                return True

    return False


# --- buttons ---
def draw_start_button(image, cx, cy, r):
    # --- colors (BGR) ---
    baby_pink   = (193, 182, 255)
    soft_yellow = (186, 248, 255)
    outline     = (160, 170, 220)
    text_main   = (255, 255, 255)
    text_shadow = (140, 140, 160)
    hint_color  = (225, 225, 230)

    overlay = image.copy()

    # soft outer glow
    cv2.circle(overlay, (cx, cy), int(r * 1.12), (200, 190, 250), -1)
    cv2.addWeighted(overlay, 0.25, image, 0.75, 0, image)

    # button base
    overlay = image.copy()
    cv2.circle(overlay, (cx, cy), r, baby_pink, -1)
    cv2.addWeighted(overlay, 0.85, image, 0.15, 0, image)

    # inner ring
    overlay = image.copy()
    cv2.circle(overlay, (cx, cy), int(r * 0.86), soft_yellow, -1)
    cv2.addWeighted(overlay, 0.65, image, 0.35, 0, image)

    # outline
    cv2.circle(image, (cx, cy), r, outline, 3)

    # glossy highlight
    overlay = image.copy()
    gloss_color = (255, 255, 255)
    cv2.ellipse(
        overlay,
        (cx, cy - int(r * 0.22)),
        (int(r * 0.85), int(r * 0.5)),
        0, 0, 360,
        gloss_color, -1
    )
    cv2.addWeighted(overlay, 0.12, image, 0.88, 0, image)

    # START text
    label = "START"
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1.4
    thick_outline = 5
    thick_fill = 3

    (tw, th), baseline = cv2.getTextSize(label, font, font_scale, thick_fill)
    text_x = cx - tw // 2
    text_y = cy + th // 3

    cv2.putText(image, label, (text_x + 2, text_y + 2), font, font_scale, text_shadow, thick_outline, cv2.LINE_AA)
    cv2.putText(image, label, (text_x, text_y), font, font_scale, text_main, thick_fill, cv2.LINE_AA)

    # hint text
    hint = "Touch with hand or press S / Space"
    hint_scale = 0.62
    hint_thick = 2
    (hw, hh), hbase = cv2.getTextSize(hint, font, hint_scale, hint_thick)
    hint_x = cx - hw // 2
    hint_y = cy + r + 48

    cv2.putText(image, hint, (hint_x + 1, hint_y + 1), font, hint_scale, (120, 120, 130), hint_thick, cv2.LINE_AA)
    cv2.putText(image, hint, (hint_x, hint_y), font, hint_scale, hint_color, hint_thick, cv2.LINE_AA)


def draw_replay_button(image, cx, cy, r):
    pastel_yellow = (180, 255, 255)
    soft_outline = (160, 220, 220)

    cv2.circle(image, (cx, cy), r, pastel_yellow, -1)
    cv2.circle(image, (cx, cy), r, soft_outline, 8)

    overlay = image.copy()
    top_shine = (200, 255, 255)
    cv2.ellipse(
        overlay,
        (cx, cy - int(r * 0.2)),
        (int(r * 0.8), int(r * 0.5)),
        0, 0, 360,
        top_shine,
        -1
    )
    cv2.addWeighted(overlay, 0.3, image, 0.7, 0)

    text = "REPLAY?"
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 1.2, 3)
    tx = cx - tw // 2
    ty = cy + th // 2
    cv2.putText(image, text, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, 1.2,
                (120, 120, 120), 4, cv2.LINE_AA)
    cv2.putText(image, text, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, 1.2,
                (255, 255, 255), 2, cv2.LINE_AA)


# --- camera setup ---
cap = cv2.VideoCapture(CAM_INDEX)
if not cap.isOpened():
    raise RuntimeError(f"Could not open camera index {CAM_INDEX}. Try 0/1/2 and check permissions.")
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1080)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
cv2.namedWindow(WIN_NAME, cv2.WINDOW_AUTOSIZE)


def reset_game(W):
    collected = 0
    missed = 0
    stage = 1
    bananas = make_stage1_bananas(W)
    bombs = []  # no bombs until Stage 4
    normal_since_epic = 0
    epic_next_at = random.randint(3, 5)
    return bananas, bombs, stage, collected, missed, normal_since_epic, epic_next_at


with mp_holistic.Holistic(
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
) as holistic:

    ok, first_frame = cap.read()
    if not ok:
        raise RuntimeError("Failed to read initial frame from camera.")
    H, W = first_frame.shape[:2]

    # --- game state ---
    GAME_STATE = "menu"  # "menu", "playing", "game_over"

    start_btn = {"cx": W // 2, "cy": H // 2, "r": 110}
    replay_btn = {"cx": W // 2, "cy": H // 2 + 130, "r": 90}

    bananas, bombs, stage, collected, missed, normal_since_epic, epic_next_at = reset_game(W)
    last_t = time.time()
    game_over_start = None  # for horse bounce timing

    while True:
        ok, frame = cap.read()
        if not ok:
            print("Failed to read frame. Is another app using the camera?")
            break

        frame = cv2.flip(frame, 1)
        H, W = frame.shape[:2]
        ground_y = H - GROUND_OFFSET

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = holistic.process(rgb)
        image = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

        # Landmarks (for feedback)
        # if results.pose_landmarks:
        #     mp_drawing.draw_landmarks(
        #         image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS,
        #         mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=3),
        #         mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2)
        #     )
        # if results.left_hand_landmarks:
        #     mp_drawing.draw_landmarks(
        #         image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
        #         mp_drawing.DrawingSpec(color=(121, 22, 76), thickness=2, circle_radius=3),
        #         mp_drawing.DrawingSpec(color=(121, 44, 250), thickness=2, circle_radius=2)
        #     )
        # if results.right_hand_landmarks:
        #     mp_drawing.draw_landmarks(
        #         image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
        #         mp_drawing.DrawingSpec(color=(80, 22, 10), thickness=2, circle_radius=3),
        #         mp_drawing.DrawingSpec(color=(80, 44, 121), thickness=2, circle_radius=2)
        #     )

        # Hand hit-circles for banana catching
        hand_circles = []
        if results.left_hand_landmarks is not None:
            hand_circles.append(hand_circle_from_landmarks(results.left_hand_landmarks, W, H))
        if results.right_hand_landmarks is not None:
            hand_circles.append(hand_circle_from_landmarks(results.right_hand_landmarks, W, H))

        # Timing
        now = time.time()
        dt = max(0.001, min(0.05, now - last_t))
        last_t = now

        # ---------- MENU ----------
        if GAME_STATE == "menu":
            # title
            if title_img is not None and title_img.shape[2] == 4:
                overlay_rgba_centered(image, title_img,
                                      start_btn["cx"], start_btn["cy"] - 180, scale=1.0)
            else:
                cv2.putText(
                    image, "MONKEYING AROUND",
                    (start_btn["cx"] - 220, start_btn["cy"] - 180),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2,
                    (0, 255, 255), 3, cv2.LINE_AA
                )

            # start button
            if start_img is not None and start_img.shape[2] == 4:
                overlay_rgba_centered(image, start_img,
                                      start_btn["cx"], start_btn["cy"], scale=1.0)
            else:
                draw_start_button(image, start_btn["cx"], start_btn["cy"], start_btn["r"])

            # show hands
            for (hc_x, hc_y, hc_r) in hand_circles:
                cv2.circle(image, (int(hc_x), int(hc_y)), int(hc_r), (0, 180, 255), 2)

            # Hand touch start?
            hand_touched_start = False
            for (hc_x, hc_y, hc_r) in hand_circles:
                d = np.hypot(hc_x - start_btn["cx"], hc_y - start_btn["cy"])
                if d <= (hc_r + start_btn["r"]):
                    hand_touched_start = True
                    if SOUND_ENABLED and pygame.mixer.get_init():
                        catch_sound.play()
                    break

            cv2.putText(
                image, "q = quit",
                (10, H - 12),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                (180, 180, 180), 1, cv2.LINE_AA
            )

            cv2.imshow(WIN_NAME, image)
            key = cv2.waitKey(16) & 0xFF

            if key == ord('q'):
                break
            if key in (ord('s'), ord(' ')) or hand_touched_start:
                bananas, bombs, stage, collected, missed, normal_since_epic, epic_next_at = reset_game(W)
                GAME_STATE = "playing"
                game_over_start = None

                if SOUND_ENABLED and pygame.mixer.get_init():
                    try:
                        pygame.mixer.music.load("audio/game_song.wav")
                        pygame.mixer.music.set_volume(0.4)
                        pygame.mixer.music.play(-1)
                    except Exception as e:
                        print("Could not play background music:", e)
            continue

        # ---------- GAME OVER ----------
        if GAME_STATE == "game_over":
            overlay = image.copy()
            cv2.rectangle(overlay, (0, 0), (W, H), (0, 0, 0), -1)
            image = cv2.addWeighted(overlay, 0.55, image, 0.45, 0)

            font = cv2.FONT_HERSHEY_SIMPLEX

            title = "GAME OVER!"
            (tw, th), _ = cv2.getTextSize(title, font, 2.0, 4)
            title_x = (W - tw) // 2
            title_y = H // 2 - 160
            cv2.putText(
                image, title,
                (title_x, title_y),
                font, 2.0,
                (0, 255, 255), 4, cv2.LINE_AA
            )

            quote = "For what the monkey considered business, the Horse considered play."
            q_scale = 0.7
            (qw, qh), _ = cv2.getTextSize(quote, font, q_scale, 2)
            quote_x = int(max(10, W * 0.12))
            quote_y = title_y + 80
            cv2.putText(
                image, quote,
                (quote_x, quote_y),
                font, q_scale,
                (255, 255, 255), 2, cv2.LINE_AA
            )

            draw_replay_button(image, replay_btn["cx"], replay_btn["cy"], replay_btn["r"])

            for (hc_x, hc_y, hc_r) in hand_circles:
                cv2.circle(image, (int(hc_x), int(hc_y)), int(hc_r), (0, 180, 255), 2)

            hand_replay = False
            for (hc_x, hc_y, hc_r) in hand_circles:
                d = np.hypot(hc_x - replay_btn["cx"], hc_y - replay_btn["cy"])
                if d <= (hc_r + replay_btn["r"]):
                    hand_replay = True
                    break

            # bouncing horse
            if use_horse_sprite and game_over_start is not None:
                t = now - game_over_start
                amplitude = 25
                period = 1.2
                offset = int(np.sin(2 * np.pi * t / period) * amplitude)

                horse_x = int(W * 0.83)
                horse_y = H // 2 + offset
                overlay_rgba_centered(image, horse_rgba, horse_x, horse_y, scale=0.7)
            elif not use_horse_sprite:
                cv2.circle(image, (int(W * 0.83), H // 2), 40, (50, 150, 255), -1)

            cv2.putText(
                image, "q = quit",
                (10, H - 12),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                (200, 200, 200), 1, cv2.LINE_AA
            )

            cv2.imshow(WIN_NAME, image)
            key = cv2.waitKey(16) & 0xFF

            if key == ord('q'):
                break
            if key in (ord('s'), ord(' ')) or hand_replay:
                bananas, bombs, stage, collected, missed, normal_since_epic, epic_next_at = reset_game(W)
                GAME_STATE = "playing"
                game_over_start = None
                if SOUND_ENABLED and pygame.mixer.get_init():
                    try:
                        pygame.mixer.music.load("audio/game_song.wav")
                        pygame.mixer.music.set_volume(0.4)
                        pygame.mixer.music.play(-1)
                    except Exception as e:
                        print("Could not play background music:", e)
            continue

        # ---------- PLAYING ----------
        # show hand circles (for banana catching)
        for (hc_x, hc_y, hc_r) in hand_circles:
            cv2.circle(image, (int(hc_x), int(hc_y)), int(hc_r), (0, 180, 255), 2)

        # ground line
        cv2.line(image, (0, int(ground_y)), (W, int(ground_y)), (60, 60, 60), 2)

        # stage transitions
        if stage == 1 and collected >= BANANAS_TO_STAGE2:
            stage = 2
            bananas = make_stage2_bananas(W)
            bombs = []
            normal_since_epic = 0
            epic_next_at = random.randint(3, 5)
        elif stage == 2 and collected >= BANANAS_TO_STAGE4:
            # jump to Stage 4: bananas + bombs
            stage = 4
            bombs = [
                Bomb(W, STAGE4_BOMB_GRAVITY, "bomb1"),
                Bomb(W, STAGE4_BOMB_GRAVITY, "bomb2")
            ]

        # update bananas
        for b in bananas:
            b.update(dt)
        for b in bananas:
            b.draw_and_update_radius(image)

        # banana collisions and misses (same behavior all stages)
        for b in bananas:
            caught = False
            for (hc_x, hc_y, hc_r) in hand_circles:
                d = np.hypot(b.x - hc_x, b.y - hc_y)
                if d <= (b.radius + hc_r):
                    caught = True
                    break

            if caught:
                if SOUND_ENABLED and pygame.mixer.get_init():
                    catch_sound.play()
                collected += 2 if b.epic else 1

                if stage == 1:
                    b.respawn_above(bananas, STAGE1_GAP_MIN, STAGE1_GAP_MAX)
                else:
                    b.respawn_above(bananas, STAGE2_GAP_MIN, STAGE2_GAP_MAX)
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
                continue

            if (b.y + b.radius) >= ground_y:
                missed += 1
                if SOUND_ENABLED and pygame.mixer.get_init():
                    miss_sound.play()

                if stage == 1:
                    b.respawn_above(bananas, STAGE1_GAP_MIN, STAGE1_GAP_MAX)
                else:
                    b.respawn_above(bananas, STAGE2_GAP_MIN, STAGE2_GAP_MAX)
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

        # Stage 4: bombs + torso/head/hands hitboxes
        if stage >= 4 and bombs:
            hit_areas = get_stage4_hit_areas(results, W, H)

            for bomb in bombs:
                bomb.update(dt)
                bomb.draw_and_update_radius(image)

                # if bomb falls past screen, respawn above
                if (bomb.y - bomb.radius) > H:
                    bomb.respawn_above(bombs, BOMB_GAP_MIN, BOMB_GAP_MAX)

                if bomb_hits_player(bomb, hit_areas):
                    # bomb hit = 1 miss, not instant game over
                    missed += 1
                    bomb_sound.play()

                    # respawn this bomb
                    bomb.respawn_above(bombs, BOMB_GAP_MIN, BOMB_GAP_MAX)

                    # check if that miss ends the game
                    if missed >= MISSED_LIMIT:
                        GAME_STATE = "game_over"
                        game_over_start = now
                        if SOUND_ENABLED and pygame.mixer.get_init():
                            pygame.mixer.music.stop()
                    break


        # Trigger GAME OVER on misses
        if missed >= MISSED_LIMIT and GAME_STATE != "game_over":
            GAME_STATE = "game_over"
            game_over_start = now
            if SOUND_ENABLED and pygame.mixer.get_init():
                pygame.mixer.music.stop()
            cv2.imshow(WIN_NAME, image)
            cv2.waitKey(16)
            continue

        # HUD
        info = f"Stage: {stage} | On screen: {len(bananas)} | Collected: {collected} | Missed: {missed}"
        cv2.putText(
            image, info,
            (10, 25),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6,
            (0, 255, 255), 2, cv2.LINE_AA
        )
        cv2.putText(
            image, "q = quit, space = reset",
            (10, H - 12),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6,
            (180, 180, 180), 1, cv2.LINE_AA
        )

        cv2.imshow(WIN_NAME, image)
        key = cv2.waitKey(16) & 0xFF

        if key == ord('q'):
            break
        elif key == ord(' '):
            bananas, bombs, stage, collected, missed, normal_since_epic, epic_next_at = reset_game(W)
            GAME_STATE = "menu"
            game_over_start = None

cap.release()
cv2.destroyAllWindows()
