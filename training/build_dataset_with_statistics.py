"""
Sign-Bot: Dataset Builder with Statistics
==========================================
This script produces EXACTLY the same dataset outputs as build_dataset.py,
then adds dataset analysis and visualization for thesis documentation.

How correctness is guaranteed
------------------------------
This script does not reimplement any dataset-building logic. It imports
the three functions that actually do the work directly from build_dataset.py:

    - collect_samples()   -> which files go into train/test
    - build_arrays()      -> loading, shape validation, zero-frame skipping,
                              normalization (via preprocessing.normalize_sequence),
                              padding/truncation
    - pad_or_truncate()   -> used indirectly through build_arrays()

The train/validation split below calls sklearn.train_test_split with the
same test_size, stratify array, and random_state as build_dataset.py, on
the same X_train_all/y_train_all produced by the same build_arrays() call.
No other randomness is introduced before this call, so the resulting
X_train/X_val/X_test/y_train/y_val/y_test arrays are bit-for-bit identical
to running build_dataset.py directly. File names, save locations, and the
final signbot_colab_package.zip contents are unchanged.

Original build_dataset.py is not modified in any way.

Usage
-----
    python build_dataset_with_statistics.py
"""

import json
import os
import shutil
import sys
import zipfile
from collections import Counter

import numpy as np
import matplotlib
matplotlib.use("Agg")  # headless rendering, no display needed
import matplotlib.pyplot as plt
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
from preprocessing import normalize_sequence  # noqa: F401 (kept for parity with original imports)

# Reuse the exact, already-working functions from build_dataset.py.
# Nothing about sample collection, normalization, or padding is
# reimplemented here.
from build_dataset import collect_samples, build_arrays, pad_or_truncate  # noqa: F401


# ==========================
# Dataset Statistics
# ==========================
# Everything in this section is new. It does not touch any variable
# used by the dataset-building steps in main() below.

STATS_DIR = os.path.join("output", "dataset", "statistics")


def plot_frame_length_distribution(lengths, max_seq_len, save_path):
    """
    Figure 1: Histogram of raw sequence lengths (all train + test videos),
    measured BEFORE padding/truncation is applied.
    """
    fig, ax = plt.subplots(figsize=(9, 6), dpi=150)
    ax.hist(lengths, bins=40, color="#3B6FA0", edgecolor="black", linewidth=0.5)
    ax.axvline(
        max_seq_len,
        color="#C0392B",
        linestyle="--",
        linewidth=1.5,
        label=f"Pad/Truncate length = {max_seq_len} frames",
    )
    ax.set_xlabel("Number of Frames", fontsize=12)
    ax.set_ylabel("Number of Videos", fontsize=12)
    ax.set_title(
        "Frame-Length Distribution (Before Padding/Truncation)",
        fontsize=13,
        fontweight="bold",
    )
    ax.legend(fontsize=10, frameon=True)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(save_path)
    plt.close(fig)


def plot_split_distribution(n_train, n_val, n_test, save_path):
    """
    Figure 2: Bar chart of sample counts in the train / validation / test splits.
    """
    labels = ["Training", "Validation", "Testing"]
    counts = [n_train, n_val, n_test]
    colors = ["#3B6FA0", "#77A34A", "#C0392B"]

    fig, ax = plt.subplots(figsize=(7, 6), dpi=150)
    bars = ax.bar(labels, counts, color=colors, edgecolor="black", linewidth=0.7)
    for bar, count in zip(bars, counts):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            f"{count}",
            ha="center",
            va="bottom",
            fontsize=11,
            fontweight="bold",
        )
    ax.set_ylabel("Number of Samples", fontsize=12)
    ax.set_title("Dataset Split Distribution", fontsize=13, fontweight="bold")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(save_path)
    plt.close(fig)


def plot_class_distribution(y_all, num_classes, save_path):
    """
    Figure 3: Bar chart of sample counts per class, across all splits combined.
    y_all contains 0-based class indices (as produced by build_arrays()).
    """
    counts = Counter(y_all.tolist())
    x = np.arange(num_classes)
    heights = [counts.get(i, 0) for i in range(num_classes)]

    fig, ax = plt.subplots(figsize=(16, 6), dpi=150)
    ax.bar(x, heights, color="#3B6FA0", edgecolor="black", linewidth=0.3)
    ax.set_xlabel("Class ID", fontsize=12)
    ax.set_ylabel("Number of Samples", fontsize=12)
    ax.set_title("Class Distribution (All Samples)", fontsize=13, fontweight="bold")

    tick_step = max(1, num_classes // 25)
    ax.set_xticks(x[::tick_step])
    ax.set_xticklabels([str(i + 1) for i in x[::tick_step]], rotation=90, fontsize=8)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(save_path)
    plt.close(fig)


def write_summary_report(stats, save_path):
    """
    Figure 4: Dataset summary. Printed to console and saved as a text file.
    """
    lines = [
        "=" * 60,
        "Sign-Bot Dataset Statistics Report",
        "=" * 60,
        f"Total videos (raw, train + test)        : {stats['total_videos']}",
        f"Number of classes                       : {stats['num_classes']}",
        f"Train samples                           : {stats['n_train']}",
        f"Validation samples                      : {stats['n_val']}",
        f"Test samples                             : {stats['n_test']}",
        "",
        "Frame-length statistics (raw, before padding/truncation):",
        f"  Minimum sequence length                : {stats['len_min']}",
        f"  Maximum sequence length                 : {stats['len_max']}",
        f"  Mean sequence length                    : {stats['len_mean']:.2f}",
        f"  Median sequence length                  : {stats['len_median']:.2f}",
        f"  Std. deviation of sequence length       : {stats['len_std']:.2f}",
        "=" * 60,
    ]
    report = "\n".join(lines)
    print("\n" + report)
    with open(save_path, "w", encoding="utf-8") as f:
        f.write(report + "\n")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# The dataset-building steps below are identical in behavior to
# build_dataset.py's main() — same functions, same call order, same
# parameters. Only the statistics block at the end (clearly marked) is new.
# ─────────────────────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("Sign-Bot: Dataset Builder V3 + Statistics")
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

    # ── Collect samples (identical to build_dataset.py) ─────────────────────
    print("Scanning .npy files …")
    train_samples, test_samples = collect_samples()
    print(f"  Train samples : {len(train_samples)}")
    print(f"  Test samples  : {len(test_samples)}\n")

    if not train_samples:
        print("Error: No training samples found. Run extract_hands.py first.")
        sys.exit(1)

    # ── Class distribution (console summary, identical to build_dataset.py) ─
    train_dist = Counter(sid for _, sid in train_samples)
    test_dist  = Counter(sid for _, sid in test_samples)
    print("Class distribution:")
    print(f"  Train — {len(train_dist)} classes, "
          f"min={min(train_dist.values())}, max={max(train_dist.values())} samples/class")
    if test_dist:
        print(f"  Test  — {len(test_dist)} classes, "
              f"min={min(test_dist.values())}, max={max(test_dist.values())} samples/class")
    print()

    # ── Frame length statistics (identical to build_dataset.py) ─────────────
    print("Frame length statistics (all training samples) …")
    lengths = []
    for npy_path, _ in train_samples:
        try:
            lengths.append(np.load(npy_path).shape[0])
        except Exception:
            pass
    if lengths:
        lengths_arr = np.array(lengths)
        print(f"  Mean   : {lengths_arr.mean():.1f}")
        print(f"  Median : {np.median(lengths_arr):.1f}")
        print(f"  Min    : {lengths_arr.min()}")
        print(f"  Max    : {lengths_arr.max()}")
        print(f"  P95    : {np.percentile(lengths_arr, 95):.0f}")
        print(f"  Truncated (>{MAX_SEQUENCE_LENGTH} frames): "
              f"{(lengths_arr > MAX_SEQUENCE_LENGTH).mean()*100:.1f}%")
        print(f"  Padded   (<{MAX_SEQUENCE_LENGTH} frames): "
              f"{(lengths_arr < MAX_SEQUENCE_LENGTH).mean()*100:.1f}%")
    print()

    # ── Build arrays (identical to build_dataset.py — same function calls) ──
    X_train_all, y_train_all = build_arrays(train_samples, "train")
    X_test,      y_test      = build_arrays(test_samples,  "test")

    print(f"\n  X_train_all : {X_train_all.shape}  y_train_all : {y_train_all.shape}")
    print(f"  X_test      : {X_test.shape}   y_test      : {y_test.shape}")

    # ── Validation split (identical parameters to build_dataset.py) ─────────
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

    # ── Normalization sanity check (identical to build_dataset.py) ──────────
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

    # ── Save arrays (identical file names/locations to build_dataset.py) ────
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

    # ── Package ZIP for Colab (identical to build_dataset.py) ───────────────
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

    # ==========================
    # Dataset Statistics
    # ==========================
    # Everything from this point on is additive: it only reads variables
    # already produced above (lengths, samples, X_*, y_*, class_names).
    # It does not alter the dataset, the saved .npy files, or the zip
    # package generated above.
    print(f"\n{'='*60}")
    print("Generating thesis statistics and figures …")
    print(f"{'='*60}")

    os.makedirs(STATS_DIR, exist_ok=True)

    # Figure 1: raw frame-length distribution across ALL videos (train + test),
    # measured before padding/truncation.
    all_lengths = list(lengths)  # train lengths, already computed above
    for npy_path, _ in test_samples:
        try:
            all_lengths.append(np.load(npy_path).shape[0])
        except Exception:
            pass
    all_lengths_arr = np.array(all_lengths)

    fig1_path = os.path.join(STATS_DIR, "frame_length_distribution.png")
    plot_frame_length_distribution(all_lengths_arr, MAX_SEQUENCE_LENGTH, fig1_path)
    print(f"  Saved Figure 1: {fig1_path}")

    # Figure 2: train/val/test split sizes.
    fig2_path = os.path.join(STATS_DIR, "dataset_split_distribution.png")
    plot_split_distribution(X_train.shape[0], X_val.shape[0], X_test.shape[0], fig2_path)
    print(f"  Saved Figure 2: {fig2_path}")

    # Figure 3: per-class sample counts, across all splits combined.
    y_all_combined = np.concatenate([y_train_all, y_test])
    fig3_path = os.path.join(STATS_DIR, "class_distribution.png")
    plot_class_distribution(y_all_combined, len(class_names), fig3_path)
    print(f"  Saved Figure 3: {fig3_path}")

    # Figure 4: dataset summary report (console + text file).
    summary_stats = {
        "total_videos": len(train_samples) + len(test_samples),
        "num_classes": len(class_names),
        "n_train": int(X_train.shape[0]),
        "n_val": int(X_val.shape[0]),
        "n_test": int(X_test.shape[0]),
        "len_min": int(all_lengths_arr.min()),
        "len_max": int(all_lengths_arr.max()),
        "len_mean": float(all_lengths_arr.mean()),
        "len_median": float(np.median(all_lengths_arr)),
        "len_std": float(all_lengths_arr.std()),
    }
    fig4_path = os.path.join(STATS_DIR, "dataset_statistics.txt")
    write_summary_report(summary_stats, fig4_path)
    print(f"  Saved Figure 4 report: {fig4_path}")

    print(f"\nAll statistics and figures saved to: {STATS_DIR}")


if __name__ == "__main__":
    main()
