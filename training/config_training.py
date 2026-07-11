"""
Sign-Bot Training Configuration
=================================
Single source of truth for all training pipeline parameters.
These values must match deployment config exactly for:
  - model_complexity
  - min_detection_confidence
  - min_tracking_confidence
  - FRAME_WIDTH / FRAME_HEIGHT  (must equal Pi PROCESS_WIDTH / PROCESS_HEIGHT)
"""

# ─────────────────────────────────────────────────────────────────────────────
# Paths  (edit to match your local SSD and output directories)
# ─────────────────────────────────────────────────────────────────────────────
SSD_ROOT   = "/Volumes/CrucialX9/KArSL"          # root of original video dataset
NPY_DIR    = "output/npy"                          # extracted .npy landmark files
DATASET_DIR = "output/dataset"                     # final consolidated arrays
LABELS_DIR  = "labels"                      # class_names.json, label_map.json
OUTPUT_ROOT = "output"

# ─────────────────────────────────────────────────────────────────────────────
# Dataset
# ─────────────────────────────────────────────────────────────────────────────
NUM_CLASSES = 100
SIGNERS     = ["03"]

# ─────────────────────────────────────────────────────────────────────────────
# MediaPipe Hands  — MUST match deployment sign_engine.py exactly
# ─────────────────────────────────────────────────────────────────────────────
HANDS_MODEL_COMPLEXITY      = 0    # 0 = lite/fastest; matches Pi deployment
MIN_DETECTION_CONFIDENCE    = 0.6  # matches Pi deployment
MIN_TRACKING_CONFIDENCE     = 0.5  # matches Pi deployment
MAX_NUM_HANDS               = 2

# ─────────────────────────────────────────────────────────────────────────────
# Frame resize before MediaPipe  — MUST match Pi PROCESS_WIDTH / PROCESS_HEIGHT
# ─────────────────────────────────────────────────────────────────────────────
FRAME_WIDTH  = 320   # matches config.PROCESS_WIDTH  on Pi
FRAME_HEIGHT = 240   # matches config.PROCESS_HEIGHT on Pi

# ─────────────────────────────────────────────────────────────────────────────
# Features
# ─────────────────────────────────────────────────────────────────────────────
NUM_HAND_POINTS   = 21
NUM_HANDS         = 2
NUM_FEATURES      = NUM_HAND_POINTS * NUM_HANDS * 3   # 126  (no pose)

# ─────────────────────────────────────────────────────────────────────────────
# Sequence
# ─────────────────────────────────────────────────────────────────────────────
MAX_SEQUENCE_LENGTH = 20   # frames per sample

# ─────────────────────────────────────────────────────────────────────────────
# Training
# ─────────────────────────────────────────────────────────────────────────────
RANDOM_SEED       = 42
EPOCHS            = 100
BATCH_SIZE        = 32
LEARNING_RATE     = 0.001
DROPOUT_RATE      = 0.3
LSTM_UNITS_1      = 128
LSTM_UNITS_2      = 64
DENSE_UNITS       = 64
VALIDATION_SPLIT  = 0.15   # fraction of train data held out as validation

# ─────────────────────────────────────────────────────────────────────────────
# Callbacks
# ─────────────────────────────────────────────────────────────────────────────
LR_FACTOR        = 0.5
LR_PATIENCE      = 5
LR_MIN           = 1e-6
EARLY_STOP_PATIENCE = 10
