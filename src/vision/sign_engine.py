"""
Sign-language detection engine: USB camera -> MediaPipe Holistic ->
normalized hand landmarks -> 20-frame sequence buffer -> TFLite inference.
Ported from the Sign-bot V2 vision_thread, wrapped as a QThread that can be
cleanly started/stopped by the Sign switch in the UI rather than running
for the lifetime of the app.
"""
import logging
import time
from collections import deque

import cv2
import numpy as np
from PyQt5.QtCore import QThread, pyqtSignal
from PyQt5.QtGui import QImage

import config

logger = logging.getLogger(__name__)

try:
    import tensorflow as tf
    Interpreter = tf.lite.Interpreter
except ImportError as e:
    raise RuntimeError(
        "TensorFlow not found. On the Pi run:\n"
        "  pip install tensorflow==2.13.0"
    ) from e

import mediapipe as mp


def normalize_hand_landmarks(hand_row, num_points=config.NUM_HAND_POINTS):
    """Wrist-relative + scale-by-max-distance normalization (must match training)."""
    if np.all(hand_row == 0):
        return hand_row
    coords = hand_row.reshape(num_points, 3)
    wrist = coords[0].copy()
    coords = coords - wrist
    distances = np.linalg.norm(coords, axis=1)
    max_dist = distances.max()
    if max_dist > 1e-6:
        coords = coords / max_dist
    return coords.flatten().astype(np.float32)


def extract_and_normalize(results):
    """Extract a 126-dim normalized feature vector (both hands) from MediaPipe results."""
    lh = np.zeros(config.NUM_HAND_POINTS * 3, dtype=np.float32)
    if results.left_hand_landmarks:
        for i, lm in enumerate(results.left_hand_landmarks.landmark):
            lh[i * 3], lh[i * 3 + 1], lh[i * 3 + 2] = lm.x, lm.y, lm.z

    rh = np.zeros(config.NUM_HAND_POINTS * 3, dtype=np.float32)
    if results.right_hand_landmarks:
        for i, lm in enumerate(results.right_hand_landmarks.landmark):
            rh[i * 3], rh[i * 3 + 1], rh[i * 3 + 2] = lm.x, lm.y, lm.z

    return np.concatenate([normalize_hand_landmarks(lh), normalize_hand_landmarks(rh)])


class SignEngine(QThread):
    frameReady = pyqtSignal(QImage)
    predictionReady = pyqtSignal(str, float)    # label ("" if below threshold), confidence
    stableWordDetected = pyqtSignal(str)         # debounced — fires once per held sign, for TTS
    errorOccurred = pyqtSignal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._running = False

    def stop(self):
        self._running = False
        self.wait(2000)

    def run(self):
        self._running = True
        try:
            interpreter = Interpreter(model_path=config.MODEL_PATH)
            interpreter.allocate_tensors()
            input_details = interpreter.get_input_details()
            output_details = interpreter.get_output_details()
            class_names = np.load(config.LABELS_PATH, allow_pickle=True)
        except Exception as e:
            self.errorOccurred.emit(f"Model load failed: {e}")
            return

        cap = cv2.VideoCapture(config.CAMERA_INDEX)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, config.CAMERA_WIDTH)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.CAMERA_HEIGHT)
        if not cap.isOpened():
            self.errorOccurred.emit("Could not open USB camera")
            return

        mp_holistic = mp.solutions.holistic
        mp_drawing = mp.solutions.drawing_utils
        holistic = mp_holistic.Holistic(
            static_image_mode=False, model_complexity=0,
            min_detection_confidence=0.5, min_tracking_confidence=0.5,
        )

        frame_buffer = deque(maxlen=config.MAX_SEQUENCE_LENGTH)
        frame_count = 0
        last_spoken_word = ""
        last_spoken_ts = 0.0
        hands_visible_prev = False

        while self._running:
            ret, frame = cap.read()
            if not ret:
                time.sleep(0.05)
                continue

            frame = cv2.resize(frame, (config.CAMERA_WIDTH, config.CAMERA_HEIGHT))
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            rgb.flags.writeable = False
            results = holistic.process(rgb)
            rgb.flags.writeable = True

            hands_visible = bool(results.left_hand_landmarks or results.right_hand_landmarks)
            if hands_visible and not hands_visible_prev:
                frame_buffer.clear()  # reset on gesture boundary
            hands_visible_prev = hands_visible

            frame_buffer.append(extract_and_normalize(results))
            frame_count += 1

            if results.left_hand_landmarks:
                mp_drawing.draw_landmarks(
                    frame, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
                    landmark_drawing_spec=mp_drawing.DrawingSpec(color=(121, 22, 76), thickness=1, circle_radius=2),
                )
            if results.right_hand_landmarks:
                mp_drawing.draw_landmarks(
                    frame, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
                    landmark_drawing_spec=mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=1, circle_radius=2),
                )

            if (len(frame_buffer) == config.MAX_SEQUENCE_LENGTH
                    and frame_count % config.PREDICT_EVERY_N_FRAMES == 0):
                sequence = np.expand_dims(np.array(list(frame_buffer), dtype=np.float32), axis=0)
                interpreter.set_tensor(input_details[0]["index"], sequence)
                interpreter.invoke()
                output = interpreter.get_tensor(output_details[0]["index"])
                pred_idx = int(np.argmax(output[0]))
                confidence = float(output[0][pred_idx])

                if confidence >= config.CONFIDENCE_THRESHOLD:
                    current_word = str(class_names[pred_idx])
                    self.predictionReady.emit(current_word, confidence)

                    now = time.time()
                    same_word_recently = (
                        current_word == last_spoken_word
                        and now - last_spoken_ts < config.STABLE_PREDICTION_REPEAT_GUARD
                    )
                    if not same_word_recently:
                        self.stableWordDetected.emit(current_word)
                        last_spoken_word = current_word
                        last_spoken_ts = now
                else:
                    self.predictionReady.emit("", confidence)

            rgb_for_qt = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h, w, ch = rgb_for_qt.shape
            qimg = QImage(rgb_for_qt.data, w, h, ch * w, QImage.Format_RGB888).copy()
            self.frameReady.emit(qimg)

        cap.release()
        holistic.close()
