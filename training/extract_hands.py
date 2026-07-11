"""
Sign-Bot: MediaPipe Hands Landmark Extraction  (Corrected V3)
==============================================================
Extracts hand-only landmarks from all KArSL videos.

Key corrections from original extract_holistic.py
--------------------------------------------------
1. Uses mp.solutions.hands instead of mp.solutions.holistic.
   Reason: deployment (sign_engine.py) uses MediaPipe Hands.
   Training and deployment must use the same detector to avoid
   a train-deploy domain gap in landmark coordinates.

2. Detector is reinitialized per video using a context manager.
   Reason: a shared Holistic/Hands instance carries tracker state
   between videos. Stale state from video N can corrupt frame 1 of
   video N+1 before the detector fires a clean detection.

3. Feature vector is 126 per frame (hands only, no pose).
   Reason: pose landmarks are discarded during training anyway.
   Extracting them wastes time and disk space.

4. Frame resized to FRAME_WIDTH x FRAME_HEIGHT (320x240) before
   MediaPipe. Must match Pi deployment PROCESS_WIDTH / PROCESS_HEIGHT.

5. Handedness assignment uses multi_handedness label ("Left"/"Right")
   — same convention as sign_engine.py on the Pi.

Output
------
Each video → one .npy file of shape (num_frames, 126).
Path: output/npy/{split}/{signer}/{sign_id_str}/{filename}.npy
Train/test boundary preserved from source folder structure.

Usage
-----
    python extract_hands.py
"""

import os
import sys
import time
import traceback

import cv2
import mediapipe as mp
import numpy as np
from tqdm import tqdm

from config_training import (
    SSD_ROOT,
    NPY_DIR,
    NUM_CLASSES,
    SIGNERS,
    NUM_HAND_POINTS,
    NUM_FEATURES,
    FRAME_WIDTH,
    FRAME_HEIGHT,
    HANDS_MODEL_COMPLEXITY,
    MIN_DETECTION_CONFIDENCE,
    MIN_TRACKING_CONFIDENCE,
    MAX_NUM_HANDS,
)


# ─────────────────────────────────────────────────────────────────────────────
# Video discovery
# ─────────────────────────────────────────────────────────────────────────────

def find_videos():
    """
    Walk the SSD folder structure and collect all video paths for the
    first NUM_CLASSES sign classes across all signers and splits.

    Returns a list of dicts:
        {
            "video_path" : str,
            "signer"     : str,   # "01", "02", "03"
            "ssd_split"  : str,   # "train" or "test"
            "sign_id"    : int,   # 1–100
            "sign_id_str": str,   # "0001"–"0100"
            "npy_path"   : str,   # output path for .npy
        }
    """
    videos = []

    for signer in SIGNERS:
        for ssd_split in ["train", "test"]:
            split_dir = os.path.join(SSD_ROOT, signer, ssd_split)
            if not os.path.isdir(split_dir):
                print(f"  Warning: Missing directory: {split_dir}")
                continue

            for range_folder in sorted(os.listdir(split_dir)):
                range_path = os.path.join(split_dir, range_folder)
                if not os.path.isdir(range_path):
                    continue

                for sign_folder in sorted(os.listdir(range_path)):
                    sign_path = os.path.join(range_path, sign_folder)
                    if not os.path.isdir(sign_path):
                        continue

                    try:
                        sign_id = int(sign_folder)
                    except ValueError:
                        continue

                    if sign_id < 1 or sign_id > NUM_CLASSES:
                        continue

                    sign_id_str = f"{sign_id:04d}"

                    for fname in sorted(os.listdir(sign_path)):
                        if not fname.lower().endswith(".mp4"):
                            continue
                        if fname.startswith("._"):   # macOS metadata files
                            continue

                        video_path   = os.path.join(sign_path, fname)
                        npy_filename = os.path.splitext(fname)[0] + ".npy"
                        npy_path     = os.path.join(
                            NPY_DIR, ssd_split, signer, sign_id_str, npy_filename
                        )

                        videos.append({
                            "video_path" : video_path,
                            "signer"     : signer,
                            "ssd_split"  : ssd_split,
                            "sign_id"    : sign_id,
                            "sign_id_str": sign_id_str,
                            "npy_path"   : npy_path,
                        })

    return videos


# ─────────────────────────────────────────────────────────────────────────────
# Per-video extraction
# ─────────────────────────────────────────────────────────────────────────────

def extract_landmarks_from_video(video_path, hands_detector):
    """
    Process one video file and return hand landmark features.

    Args:
        video_path    : path to .mp4 file
        hands_detector: initialized mp.solutions.hands.Hands instance
                        (fresh instance per video — see main loop)

    Returns:
        np.ndarray of shape (num_frames, 126)  or  None if no frames could be read.

    Feature layout per frame (126 values):
        [left_hand_x0, left_hand_y0, left_hand_z0, … × 21]   (63 values)
        [right_hand_x0, right_hand_y0, right_hand_z0, … × 21] (63 values)

    Missing hand → that hand's 63 values remain zero.
    Handedness determined by results.multi_handedness label ("Left" / "Right").
    This matches the convention used in sign_engine.py on the Raspberry Pi.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None

    frames_data = []
    hand_size   = NUM_HAND_POINTS * 3   # 63

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Resize to match Pi deployment processing resolution
            frame     = cv2.resize(frame, (FRAME_WIDTH, FRAME_HEIGHT))
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            rgb_frame.flags.writeable = False

            results = hands_detector.process(rgb_frame)

            # Build 126-dim feature vector: [LH(63) | RH(63)]
            lh_row = np.zeros(hand_size, dtype=np.float32)
            rh_row = np.zeros(hand_size, dtype=np.float32)

            if results.multi_hand_landmarks and results.multi_handedness:
                for hand_lms, handedness in zip(
                        results.multi_hand_landmarks,
                        results.multi_handedness):

                    label = handedness.classification[0].label   # "Left" or "Right"
                    row   = np.array(
                        [[lm.x, lm.y, lm.z] for lm in hand_lms.landmark],
                        dtype=np.float32,
                    ).flatten()

                    if label == "Left":
                        lh_row = row
                    else:
                        rh_row = row

            frame_features = np.concatenate([lh_row, rh_row])   # (126,)
            frames_data.append(frame_features)

    finally:
        cap.release()

    if not frames_data:
        return None

    return np.array(frames_data, dtype=np.float32)   # (num_frames, 126)


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("Sign-Bot: MediaPipe Hands Landmark Extraction  (V3)")
    print("=" * 60)
    print(f"  SSD source      : {SSD_ROOT}")
    print(f"  NPY output      : {NPY_DIR}")
    print(f"  Classes         : 1–{NUM_CLASSES}")
    print(f"  Signers         : {SIGNERS}")
    print(f"  Features/frame  : {NUM_FEATURES}  (hands only, no pose)")
    print(f"  Frame resize    : {FRAME_WIDTH}×{FRAME_HEIGHT}")
    print(f"  Model complexity: {HANDS_MODEL_COMPLEXITY}")
    print(f"  Min detection   : {MIN_DETECTION_CONFIDENCE}")
    print(f"  Min tracking    : {MIN_TRACKING_CONFIDENCE}")
    print()

    # ── Discover videos ──────────────────────────────────────────────────────
    print("Scanning for videos …")
    videos = find_videos()
    print(f"  Found {len(videos)} video files.\n")

    if not videos:
        print("No videos found. Check SSD_ROOT and folder structure.")
        sys.exit(1)

    already_done = sum(1 for v in videos if os.path.exists(v["npy_path"]))
    to_process   = [v for v in videos if not os.path.exists(v["npy_path"])]

    print(f"  Already extracted : {already_done}")
    print(f"  Remaining         : {len(to_process)}\n")

    if not to_process:
        print("All videos already extracted.")
        return

    # ── Process videos ────────────────────────────────────────────────────────
    success_count = 0
    skip_count    = 0
    error_count   = 0
    total_frames  = 0
    start_time    = time.time()

    pbar = tqdm(to_process, desc="Extracting", unit="video")

    for video_info in pbar:
        video_path = video_info["video_path"]
        npy_path   = video_info["npy_path"]

        # ── Fresh detector per video (prevents tracker state bleed) ──────────
        # This is the key correction from the original script.
        # A shared detector carries internal tracking state between videos.
        # Reinitializing ensures video N+1 starts with a clean detection pass.
        try:
            with mp.solutions.hands.Hands(
                static_image_mode=False,
                max_num_hands=MAX_NUM_HANDS,
                model_complexity=HANDS_MODEL_COMPLEXITY,
                min_detection_confidence=MIN_DETECTION_CONFIDENCE,
                min_tracking_confidence=MIN_TRACKING_CONFIDENCE,
            ) as hands_detector:

                landmarks = extract_landmarks_from_video(video_path, hands_detector)

            if landmarks is None or landmarks.shape[0] == 0:
                skip_count += 1
                continue

            os.makedirs(os.path.dirname(npy_path), exist_ok=True)
            np.save(npy_path, landmarks)

            success_count += 1
            total_frames  += landmarks.shape[0]

        except Exception as e:
            error_count += 1
            tqdm.write(f"  Error: {video_path}: {e}")

        elapsed = time.time() - start_time
        rate    = (success_count + skip_count + error_count) / max(elapsed, 1)
        pbar.set_postfix({
            "ok"  : success_count,
            "skip": skip_count,
            "err" : error_count,
            "v/s" : f"{rate:.1f}",
        })

    # ── Summary ───────────────────────────────────────────────────────────────
    elapsed = time.time() - start_time
    print()
    print("=" * 60)
    print("Extraction Complete")
    print("=" * 60)
    print(f"  Videos processed : {success_count}")
    print(f"  Videos skipped   : {skip_count}  (no landmarks detected)")
    print(f"  Errors           : {error_count}")
    print(f"  Total frames     : {total_frames}")
    print(f"  Time elapsed     : {elapsed / 3600:.1f} hours")

    total_size = sum(
        os.path.getsize(os.path.join(r, f))
        for r, _, files in os.walk(NPY_DIR)
        for f in files if f.endswith(".npy")
    )
    print(f"  Total .npy size  : {total_size / (1024**2):.1f} MB")


if __name__ == "__main__":
    main()
