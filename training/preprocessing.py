"""
Sign-Bot Shared Preprocessing
================================
Single source of truth for landmark normalization.

Imported by:
  - build_dataset.py          (training pipeline)
  - train_production_model.py (production training)
  - loso_cross_validation.py  (LOSO evaluation)

On the Raspberry Pi, copy this file to your project directory and import it
from sign_engine.py instead of duplicating the normalization logic.

Normalization algorithm
-----------------------
Per frame, per hand independently:
  1. Reshape flat (63,) vector to (21, 3)
  2. Subtract wrist (landmark 0) from all landmarks  → translation invariant
  3. Divide by max Euclidean distance from wrist      → scale invariant

Output range: approximately [-1, 1]
No global statistics used → zero leakage risk between train and test.
"""

import numpy as np

NUM_HAND_POINTS = 21


def normalize_hand_landmarks(hand_row, num_points=NUM_HAND_POINTS):
    """
    Normalize a single hand's landmarks: wrist-relative + scale by max distance.

    Args:
        hand_row  : np.ndarray of shape (num_points * 3,)  flat [x0,y0,z0, x1,y1,z1 …]
        num_points: number of landmarks (21 for MediaPipe hand)

    Returns:
        Normalized flat array of same shape.
        Returns the original zero array unchanged if the hand was not detected.
    """
    if np.all(hand_row == 0):
        return hand_row   # hand not detected — preserve zero fill

    coords = hand_row.reshape(num_points, 3)
    wrist  = coords[0].copy()

    coords    = coords - wrist                          # translate to wrist origin
    distances = np.linalg.norm(coords, axis=1)
    max_dist  = distances.max()

    if max_dist > 1e-6:
        coords = coords / max_dist                      # scale invariant

    return coords.flatten().astype(np.float32)


def normalize_frame(frame_hands, num_hand_points=NUM_HAND_POINTS):
    """
    Normalize both hands in a single frame independently.

    Args:
        frame_hands: np.ndarray of shape (126,)  [left_hand(63) | right_hand(63)]

    Returns:
        Normalized frame of same shape.
    """
    hand_size = num_hand_points * 3          # 63
    lh = frame_hands[:hand_size]
    rh = frame_hands[hand_size:]
    return np.concatenate([
        normalize_hand_landmarks(lh, num_hand_points),
        normalize_hand_landmarks(rh, num_hand_points),
    ])


def normalize_sequence(seq_hands):
    """
    Apply per-frame normalization to an entire variable-length sequence.

    Args:
        seq_hands: np.ndarray of shape (num_frames, 126)

    Returns:
        Normalized array of same shape.
    """
    return np.array(
        [normalize_frame(frame) for frame in seq_hands],
        dtype=np.float32,
    )
