"""
Sign-Bot: Live Webcam Test  (Mac pre-flight check)
=====================================================
Run this the night before deployment to test the full real-time pipeline
end to end on your Mac's built-in camera — MediaPipe extraction,
normalization, sliding-window buffering, and TFLite inference — before
touching the Raspberry Pi tomorrow.

This mirrors the Pi's sign_engine.py logic as closely as possible so a
successful test here is a strong signal (not a guarantee) that the Pi
version will behave the same way, since it's running the same model,
the same preprocessing.py normalization, and the same MediaPipe settings.

What this also checks for free
--------------------------------
Handedness/mirroring: this was flagged earlier as unverified. The on-screen
overlay shows which hand MediaPipe labels "Left" vs "Right" in real time.
Hold up your physical left hand and confirm the overlay says "Left". If it
says "Right", your camera feed is mirrored relative to training and needs
cv2.flip(frame, 1) added before MediaPipe processing (both here and in the
Pi's sign_engine.py).

Controls
--------
    q — quit

Dependencies
------------
    pip install arabic-reshaper python-bidi pillow

cv2.putText cannot render Arabic script — it has no right-to-left support
and no contextual letter shaping. Predictions are Arabic class names, so
this script renders all overlay text through PIL with arabic-reshaper and
python-bidi instead, using macOS's built-in Geeza Pro font.

Usage
-----
    conda activate action_detection
    python test_webcam_live.py
"""

import os
import time
from collections import deque

import cv2
import numpy as np
import tensorflow as tf
import mediapipe as mp
import arabic_reshaper
from bidi.algorithm import get_display
from PIL import Image, ImageDraw, ImageFont

from preprocessing import normalize_frame

# ─────────────────────────────────────────────────────────────────────────────
# Config — matches training pipeline and Pi's config.py exactly
# ─────────────────────────────────────────────────────────────────────────────
MODEL_PATH = os.path.expanduser(
    "~/Desktop/signbot_pipeline/model/signbot_production.tflite"
)
LABELS_PATH = os.path.expanduser(
    "~/Desktop/signbot_pipeline/output/dataset/class_names.npy"
)

# macOS ships Geeza Pro by default — supports Arabic script out of the box.
# If this path doesn't exist on your system, find an alternative with:
#   find /System/Library/Fonts -iname "*geeza*" -o -iname "*arabic*"
ARABIC_FONT_PATH = "/System/Library/Fonts/GeezaPro.ttc"
FONT_SIZE = 28

CAMERA_INDEX      = 0
CAMERA_WIDTH      = 640
CAMERA_HEIGHT     = 480
PROCESS_WIDTH     = 320   # must match training FRAME_WIDTH
PROCESS_HEIGHT    = 240   # must match training FRAME_HEIGHT

NUM_HAND_POINTS   = 21
MAX_SEQUENCE_LENGTH = 20
PREDICT_EVERY_N_FRAMES = 3
CONFIDENCE_THRESHOLD    = 0.80
CONFIDENCE_REJECT_BELOW = 0.60

MODEL_COMPLEXITY         = 0
MIN_DETECTION_CONFIDENCE = 0.6
MIN_TRACKING_CONFIDENCE  = 0.5

# ─────────────────────────────────────────────────────────────────────────────
# Arabic text rendering
# cv2.putText cannot render Arabic at all — no right-to-left support, no
# contextual letter shaping (the same letter takes different glyph forms
# depending on its position in the word). This routes text through
# arabic_reshaper (fixes letter shaping) + python-bidi (fixes RTL ordering)
# + PIL (actually draws Unicode glyphs, which OpenCV cannot do), then
# converts the result back to an OpenCV-compatible frame.
# ─────────────────────────────────────────────────────────────────────────────
if not os.path.exists(ARABIC_FONT_PATH):
    raise FileNotFoundError(
        f"Arabic font not found at: {ARABIC_FONT_PATH}\n"
        f"Run: find /System/Library/Fonts -iname '*geeza*' -o -iname '*arabic*'\n"
        f"and update ARABIC_FONT_PATH at the top of this script."
    )
_arabic_font = ImageFont.truetype(ARABIC_FONT_PATH, FONT_SIZE)


def _contains_arabic(text):
    """Return True if the string contains any Arabic Unicode characters."""
    return any('\u0600' <= ch <= '\u06FF' for ch in text)


def draw_text(frame_bgr, text, position, color_bgr, font_size=FONT_SIZE):
    """Draw text (Arabic or Latin) onto an OpenCV BGR frame via PIL.

    Arabic text is reshaped + bidi-processed before rendering so glyphs
    appear in the correct visual order. Latin/digit text bypasses that
    pipeline entirely to avoid mis-rendering (e.g. numbers shown as boxes).

    Wrapped defensively: if PIL rendering throws, falls back to
    cv2.putText with the raw ASCII text so the overlay never disappears
    silently.
    """
    try:
        if _contains_arabic(text):
            reshaped = arabic_reshaper.reshape(text)
            render_text = get_display(reshaped)
        else:
            render_text = text

        img_pil = Image.fromarray(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(img_pil)
        color_rgb = (color_bgr[2], color_bgr[1], color_bgr[0])

        font = _arabic_font if font_size == FONT_SIZE else ImageFont.truetype(ARABIC_FONT_PATH, font_size)
        draw.text(position, render_text, font=font, fill=color_rgb)

        return cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
    except Exception as e:
        print(f"[draw_text ERROR] Failed to render {text!r}: {e}")
        # Fallback: plain cv2.putText, ASCII-only.
        safe_text = text.encode("ascii", errors="replace").decode("ascii")
        cv2.putText(frame_bgr, safe_text, position,
                    cv2.FONT_HERSHEY_SIMPLEX, font_size / 40, color_bgr, 1)
        return frame_bgr

# ─────────────────────────────────────────────────────────────────────────────
# Load model and labels
# ─────────────────────────────────────────────────────────────────────────────
print(f"Loading model: {MODEL_PATH}")
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Model not found: {MODEL_PATH}")

interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)
interpreter.allocate_tensors()
input_details  = interpreter.get_input_details()
output_details = interpreter.get_output_details()
print(f"  Input shape : {input_details[0]['shape']}")
print(f"  Output shape: {output_details[0]['shape']}")

print(f"Loading labels: {LABELS_PATH}")
class_names = np.load(LABELS_PATH, allow_pickle=True)
print(f"  {len(class_names)} classes loaded")

# Diagnostic: check for type inconsistencies. If the original label sheet
# was exported from Excel/JSON with digit-only entries auto-converted to
# actual numbers instead of strings, some entries here could be non-string
# types, which is a plausible cause of a rendering failure specific to the
# numeric sign classes.
_types_seen = set(type(c).__name__ for c in class_names)
if len(_types_seen) > 1:
    print(f"  WARNING: class_names contains mixed types: {_types_seen}")
    print("  This may explain rendering failures specific to certain classes.")
    for i, c in enumerate(class_names[:35]):
        print(f"    [{i}] {c!r}  ({type(c).__name__})")
else:
    print(f"  All entries are type: {_types_seen.pop()}")

# ─────────────────────────────────────────────────────────────────────────────
# Feature extraction — matches sign_engine.py exactly
# ─────────────────────────────────────────────────────────────────────────────

def extract_and_normalize_hands(results):
    """Extract a 126-dim normalized feature vector from MediaPipe Hands results.

    Left hand goes in the first 63 values, right hand in the last 63,
    matching how the training dataset was built. Also returns raw
    handedness labels for the on-screen debug overlay.
    """
    lh = np.zeros(NUM_HAND_POINTS * 3, dtype=np.float32)
    rh = np.zeros(NUM_HAND_POINTS * 3, dtype=np.float32)
    detected_labels = []

    if results.multi_hand_landmarks and results.multi_handedness:
        for hand_lms, handedness in zip(results.multi_hand_landmarks,
                                         results.multi_handedness):
            label = handedness.classification[0].label
            detected_labels.append(label)
            row = np.array(
                [[lm.x, lm.y, lm.z] for lm in hand_lms.landmark],
                dtype=np.float32,
            ).flatten()
            if label == "Left":
                lh = row
            else:
                rh = row

    raw = np.concatenate([lh, rh])
    normalized = normalize_frame(raw, NUM_HAND_POINTS)
    return normalized, detected_labels


# ─────────────────────────────────────────────────────────────────────────────
# Main loop
# ─────────────────────────────────────────────────────────────────────────────
mp_hands   = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    model_complexity=MODEL_COMPLEXITY,
    min_detection_confidence=MIN_DETECTION_CONFIDENCE,
    min_tracking_confidence=MIN_TRACKING_CONFIDENCE,
)

cap = cv2.VideoCapture(CAMERA_INDEX)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)

if not cap.isOpened():
    raise RuntimeError("Could not open webcam. Check CAMERA_INDEX.")

print("\nCamera opened. Press 'q' to quit.")
print("Hold up your LEFT hand first and confirm the overlay says 'Left'.\n")

frame_buffer = deque(maxlen=MAX_SEQUENCE_LENGTH)
frame_count = 0
hands_visible_prev = False
last_prediction_text = "Listening..."
last_confidence = 0.0
last_labels_seen = []
fps = 0.0  # EMA FPS (matches Pi's vision_thread)

while True:
    loop_start = time.time()  # For EMA FPS (same pattern as Pi's vision_thread)

    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 1
    display_frame = frame.copy()

    small = cv2.resize(frame, (PROCESS_WIDTH, PROCESS_HEIGHT))
    rgb_small = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
    rgb_small.flags.writeable = False
    results = hands.process(rgb_small)
    rgb_small.flags.writeable = True

    hands_visible = bool(results.multi_hand_landmarks)
    if hands_visible and not hands_visible_prev:
        frame_buffer.clear()
    hands_visible_prev = hands_visible

    normalized_frame, detected_labels = extract_and_normalize_hands(results)
    if detected_labels:
        last_labels_seen = detected_labels
    frame_buffer.append(normalized_frame)

    # Landmark drawing intentionally removed — no squares/dots on the video feed.

    # Run inference every N frames once buffer is full
    if (len(frame_buffer) == MAX_SEQUENCE_LENGTH
            and frame_count % PREDICT_EVERY_N_FRAMES == 0):
        sequence = np.expand_dims(
            np.array(list(frame_buffer), dtype=np.float32), axis=0
        )
        interpreter.set_tensor(input_details[0]["index"], sequence)
        interpreter.invoke()

        output = interpreter.get_tensor(output_details[0]["index"])
        pred_idx = int(np.argmax(output[0]))
        confidence = float(output[0][pred_idx])
        last_confidence = confidence

        if confidence >= CONFIDENCE_THRESHOLD:
            last_prediction_text = str(class_names[pred_idx])
        elif confidence >= CONFIDENCE_REJECT_BELOW:
            last_prediction_text = f"~{str(class_names[pred_idx])}"
        else:
            last_prediction_text = "..."

    # ── EMA FPS (same formula as Pi's vision_thread) ───────────────────────
    loop_dt = time.time() - loop_start
    fps = 0.9 * fps + 0.1 * (1.0 / max(loop_dt, 1e-6))

    # ── On-screen overlay (mirrors Pi dashboard style) ────────────────────
    h, w = display_frame.shape[:2]

    # Dark header bar
    cv2.rectangle(display_frame, (0, 0), (w, 110), (20, 20, 20), -1)

    # ── Big prediction sign (Arabic gets PIL; numbers/Latin get PIL too) ──
    # No "Prediction:" prefix — just the sign name, large and centered.
    pred_display = str(last_prediction_text)
    is_listening = pred_display in ("Listening...", "...")
    pred_color = (200, 200, 200) if is_listening else (0, 212, 170)  # teal when active
    display_frame = draw_text(
        display_frame, pred_display,
        (12, 6), pred_color, font_size=42
    )

    # ── Confidence pill ────────────────────────────────────────────────────
    conf_pct = f"{last_confidence:.0%}" if not is_listening else "—"
    display_frame = draw_text(
        display_frame, f"conf: {conf_pct}",
        (12, 62), (160, 160, 160), font_size=20
    )

    # ── FPS + Hands (pure ASCII — cv2.putText is fine) ────────────────────
    handedness_str = ", ".join(last_labels_seen) if last_labels_seen else "—"
    status_str = f"FPS: {fps:.0f}   hands: {handedness_str}"
    cv2.putText(display_frame, status_str,
                (w - 300, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (100, 220, 100), 1)

    # ── Buffer fill bar (matches Pi's buffer_fill indicator) ──────────────
    buffer_fill = len(frame_buffer) / MAX_SEQUENCE_LENGTH
    bar_x0, bar_y, bar_w, bar_h = w - 300, 50, 280, 10
    cv2.rectangle(display_frame, (bar_x0, bar_y), (bar_x0 + bar_w, bar_y + bar_h),
                  (60, 60, 60), -1)
    filled_w = int(bar_w * buffer_fill)
    bar_color = (0, 200, 150) if buffer_fill >= 1.0 else (80, 150, 220)
    if filled_w > 0:
        cv2.rectangle(display_frame, (bar_x0, bar_y),
                      (bar_x0 + filled_w, bar_y + bar_h), bar_color, -1)
    cv2.putText(display_frame, f"buf: {buffer_fill:.0%}",
                (bar_x0, bar_y + bar_h + 14),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (150, 150, 150), 1)

    cv2.imshow("Sign-Bot Live Test (press q to quit)", display_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
hands.close()

print("Session ended.")
