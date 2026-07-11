"""
Sign-Bot: Dataset Builder  (Corrected V3)
==========================================
Converts per-video .npy landmark files into consolidated train/val/test arrays
and packages them into a ZIP for Google Colab.

Key corrections from original build_dataset.py
-----------------------------------------------
1. No pose stripping. Extracted .npy files are already 126 features (hands only).
   The original stripped pose columns (0:33) from 159-feature files.
   V3 extraction produces 126-feature files directly.

2. Proper validation split created here from training data only.
   The original script had no validation split. Training used X_test as
   validation_data, which is test set leakage. This script produces
   X_val / y_val from 15% of training data (stratified, seeded).

3. Imports normalization from shared preprocessing.py module.
   The original duplicated the normalization function. Any divergence
   between the two copies would silently corrupt train-deploy consistency.

Output ZIP contents
-------------------
    signbot_colab_package.zip
    ├── X_train.npy    (N_train, 20, 126)
    ├── y_train.npy    (N_train,)
    ├── X_val.npy      (N_val,   20, 126)   ← new
    ├── y_val.npy      (N_val,)              ← new
    ├── X_test.npy     (N_test,  20, 126)
    ├── y_test.npy     (N_test,)
    ├── class_names.npy
    └── label_map.json

Usage
-----
    python build_dataset.py
"""

import json
import os
import shutil
import sys
import zipfile
from collections import Counter

import numpy as np
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from config_training import (
    NPY_DIR,
    DATASET_DIR,
    LABELS_DIR,
    OUTPUT_ROOT,
    NUM_CLASSES,
    NUM_FEATURES,
    MAX_SEQUENCE_LENGTH,
    SIGNERS,
    RANDOM_SEED,
    VALIDATION_SPLIT,
)
from preprocessing import normalize_sequence   # shared module — single source of truth


# ─────────────────────────────────────────────────────────────────────────────
# Sequence utilities
# ─────────────────────────────────────────────────────────────────────────────

def pad_or_truncate(sequence, max_len):
    """
    Pad with zeros or truncate a sequence to exactly max_len frames.

    Truncation keeps the first max_len frames.
    Padding appends zero frames at the end.

    Known limitation: front-truncation may discard the sign nucleus in
    long videos where preparation phase exceeds max_len frames.
    Center-crop would be more robust but requires empirical validation.
    This is documented as a limitation in the thesis.
    """
    num_frames = sequence.shape[0]
    if num_frames >= max_len:
        return sequence[:max_len]
    padding = np.zeros((max_len - num_frames, sequence.shape[1]), dtype=np.float32)
    return np.concatenate([sequence, padding], axis=0)


# ─────────────────────────────────────────────────────────────────────────────
# Sample collection
# ─────────────────────────────────────────────────────────────────────────────

def collect_samples():
    """
    Walk npy/{train|test}/{signer}/{sign_id}/*.npy and return sample lists.

    Returns:
        train_samples : list of (npy_path, sign_id)
        test_samples  : list of (npy_path, sign_id)
    """
    train_samples = []
    test_samples  = []

    for split_name in ["train", "test"]:
        for signer in SIGNERS:
            signer_dir = os.path.join(NPY_DIR, split_name, signer)
            if not os.path.isdir(signer_dir):
                print(f"  Warning: Missing directory: {signer_dir}")
                continue

            for sign_folder in sorted(os.listdir(signer_dir)):
                sign_path = os.path.join(signer_dir, sign_folder)
                if not os.path.isdir(sign_path):
                    continue

                try:
                    sign_id = int(sign_folder)
                except ValueError:
                    continue

                if sign_id < 1 or sign_id > NUM_CLASSES:
                    continue

                for npy_file in sorted(os.listdir(sign_path)):
                    if not npy_file.endswith(".npy"):
                        continue
                    npy_path = os.path.join(sign_path, npy_file)
                    if split_name == "train":
                        train_samples.append((npy_path, sign_id))
                    else:
                        test_samples.append((npy_path, sign_id))

    return train_samples, test_samples


# ─────────────────────────────────────────────────────────────────────────────
# Array builder
# ─────────────────────────────────────────────────────────────────────────────

def build_arrays(samples, split_name):
    """
    Load .npy files, normalize, pad/truncate, and consolidate into arrays.

    Args:
        samples    : list of (npy_path, sign_id)
        split_name : "train" or "test" (used only for progress display)

    Returns:
        X : np.ndarray  (N, MAX_SEQUENCE_LENGTH, 126)
        y : np.ndarray  (N,)  0-based class indices
    """
    X_list  = []
    y_list  = []
    skipped = 0
    skip_by_class = Counter()

    for npy_path, sign_id in tqdm(samples, desc=f"Building {split_name}"):
        try:
            seq = np.load(npy_path)   # shape: (frames, 126)

            # Validate shape — V3 extraction produces 126 features (no pose)
            if seq.ndim != 2 or seq.shape[1] != NUM_FEATURES:
                tqdm.write(
                    f"  Shape mismatch: {npy_path} "
                    f"expected (*, {NUM_FEATURES}), got {seq.shape}"
                )
                skipped += 1
                skip_by_class[sign_id] += 1
                continue

            # Skip sequences where no hand was detected in any frame
            if np.all(seq == 0):
                skipped += 1
                skip_by_class[sign_id] += 1
                continue

            # Normalize (imported from preprocessing.py — same as Pi deployment)
            seq = normalize_sequence(seq)

            # Pad or truncate to fixed length
            seq = pad_or_truncate(seq, MAX_SEQUENCE_LENGTH)

            X_list.append(seq)
            y_list.append(sign_id - 1)   # 1-based → 0-based

        except Exception as e:
            tqdm.write(f"  Error loading {npy_path}: {e}")
            skipped += 1
            skip_by_class[sign_id] += 1

    if skipped > 0:
        print(f"\n  Skipped {skipped} sample(s) in {split_name}.")
        print(f"  Skip breakdown by class (top 10): "
              f"{skip_by_class.most_common(10)}")

    X = np.array(X_list, dtype=np.float32)
    y = np.array(y_list,  dtype=np.int32)
    return X, y


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("Sign-Bot: Dataset Builder V3")
    print("=" * 60)
    print(f"  NPY source      : {NPY_DIR}")
    print(f"  Output          : {DATASET_DIR}")
    print(f"  Classes         : 1–{NUM_CLASSES}")
    print(f"  Sequence length : {MAX_SEQUENCE_LENGTH} frames")
    print(f"  Features/frame  : {NUM_FEATURES}  (hands only)")
    print(f"  Validation split: {VALIDATION_SPLIT*100:.0f}% of training data")
    print(f"  Random seed     : {RANDOM_SEED}")
    print()

    # ── Load class names ──────────────────────────────────────────────────────
    class_names_path = os.path.join(LABELS_DIR, "class_names.json")
    label_map_path   = os.path.join(LABELS_DIR, "label_map.json")

    if not os.path.exists(class_names_path):
        print("Error: class_names.json not found. Run create_label_map.py first.")
        sys.exit(1)

    with open(class_names_path, "r", encoding="utf-8") as f:
        class_names = json.load(f)
    print(f"  Labels: {len(class_names)} Arabic classes")
    print(f"  Sample: {class_names[:3]}\n")

    # ── Collect samples ───────────────────────────────────────────────────────
    print("Scanning .npy files …")
    train_samples, test_samples = collect_samples()
    print(f"  Train samples : {len(train_samples)}")
    print(f"  Test samples  : {len(test_samples)}\n")

    if not train_samples:
        print("Error: No training samples found. Run extract_hands.py first.")
        sys.exit(1)

    # ── Class distribution ────────────────────────────────────────────────────
    train_dist = Counter(sid for _, sid in train_samples)
    test_dist  = Counter(sid for _, sid in test_samples)
    print("Class distribution:")
    print(f"  Train — {len(train_dist)} classes, "
          f"min={min(train_dist.values())}, max={max(train_dist.values())} samples/class")
    if test_dist:
        print(f"  Test  — {len(test_dist)} classes, "
              f"min={min(test_dist.values())}, max={max(test_dist.values())} samples/class")
    print()

    # ── Frame length statistics ───────────────────────────────────────────────
    print("Frame length statistics (all training samples) …")
    lengths = []
    for npy_path, _ in train_samples:
        try:
            lengths.append(np.load(npy_path).shape[0])
        except Exception:
            pass
    if lengths:
        lengths = np.array(lengths)
        print(f"  Mean   : {lengths.mean():.1f}")
        print(f"  Median : {np.median(lengths):.1f}")
        print(f"  Min    : {lengths.min()}")
        print(f"  Max    : {lengths.max()}")
        print(f"  P95    : {np.percentile(lengths, 95):.0f}")
        print(f"  Truncated (>{MAX_SEQUENCE_LENGTH} frames): "
              f"{(lengths > MAX_SEQUENCE_LENGTH).mean()*100:.1f}%")
        print(f"  Padded   (<{MAX_SEQUENCE_LENGTH} frames): "
              f"{(lengths < MAX_SEQUENCE_LENGTH).mean()*100:.1f}%")
    print()

    # ── Build arrays ──────────────────────────────────────────────────────────
    X_train_all, y_train_all = build_arrays(train_samples, "train")
    X_test,      y_test      = build_arrays(test_samples,  "test")

    print(f"\n  X_train_all : {X_train_all.shape}  y_train_all : {y_train_all.shape}")
    print(f"  X_test      : {X_test.shape}   y_test      : {y_test.shape}")

    # ── Validation split (CRITICAL CORRECTION) ────────────────────────────────
    # Split training data into train + validation.
    # Validation set is used by callbacks during training.
    # Test set is never seen during training.
    print(f"\nCreating stratified validation split ({VALIDATION_SPLIT*100:.0f}%) …")
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_all, y_train_all,
        test_size=VALIDATION_SPLIT,
        stratify=y_train_all,
        random_state=RANDOM_SEED,
    )
    print(f"  X_train : {X_train.shape}")
    print(f"  X_val   : {X_val.shape}")
    print(f"  X_test  : {X_test.shape}  (held out — never used during training)")

    # ── Normalization sanity check ────────────────────────────────────────────
    print(f"\nNormalization check:")
    print(f"  X_train range : [{X_train.min():.4f}, {X_train.max():.4f}]")
    print(f"  X_val   range : [{X_val.min():.4f},   {X_val.max():.4f}]")
    print(f"  X_test  range : [{X_test.min():.4f},  {X_test.max():.4f}]")

    nz = X_train[X_train != 0]
    if len(nz) > 0:
        print(f"  X_train non-zero mean: {nz.mean():.4f}  std: {nz.std():.4f}")

    assert X_train.shape[1:] == (MAX_SEQUENCE_LENGTH, NUM_FEATURES), "Train shape mismatch"
    assert X_val.shape[1:]   == (MAX_SEQUENCE_LENGTH, NUM_FEATURES), "Val shape mismatch"
    assert X_test.shape[1:]  == (MAX_SEQUENCE_LENGTH, NUM_FEATURES), "Test shape mismatch"

    # ── Save arrays ───────────────────────────────────────────────────────────
    os.makedirs(DATASET_DIR, exist_ok=True)
    files_to_zip = {}

    arrays = [
        ("X_train.npy", X_train),
        ("y_train.npy", y_train),
        ("X_val.npy",   X_val),
        ("y_val.npy",   y_val),
        ("X_test.npy",  X_test),
        ("y_test.npy",  y_test),
    ]
    for name, arr in arrays:
        path = os.path.join(DATASET_DIR, name)
        np.save(path, arr)
        files_to_zip[name] = path
        print(f"  Saved {name}: {os.path.getsize(path)/(1024**2):.1f} MB")

    cn_path = os.path.join(DATASET_DIR, "class_names.npy")
    np.save(cn_path, np.array(class_names))
    files_to_zip["class_names.npy"] = cn_path

    lm_dest = os.path.join(DATASET_DIR, "label_map.json")
    shutil.copy2(label_map_path, lm_dest)
    files_to_zip["label_map.json"] = lm_dest

    # ── Package ZIP for Colab ─────────────────────────────────────────────────
    zip_path = os.path.join(OUTPUT_ROOT, "signbot_colab_package.zip")
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for arcname, filepath in files_to_zip.items():
            zf.write(filepath, arcname)

    print(f"\n{'='*60}")
    print(f"Colab package ready: {zip_path}")
    print(f"  Size: {os.path.getsize(zip_path)/(1024**2):.1f} MB")
    print(f"  X_train {X_train.shape}  X_val {X_val.shape}  X_test {X_test.shape}")
    print(f"{'='*60}")
    print("\nUpload signbot_colab_package.zip to Google Drive for training.")


if __name__ == "__main__":
    main()
