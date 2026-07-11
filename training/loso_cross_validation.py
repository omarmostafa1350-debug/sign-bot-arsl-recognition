# -*- coding: utf-8 -*-
"""
Sign-Bot: Leave-One-Signer-Out (LOSO) Cross-Validation
=======================================================
Run on Google Colab (T4 GPU).

Purpose
-------
LOSO proves your BiLSTM architecture generalizes to unseen signers.
It does NOT produce the model you deploy. It produces accuracy metrics
you report in Chapter 3 of your thesis.

The three folds:
    Fold 1: Train on Signers 02+03, Test on Signer 01
    Fold 2: Train on Signers 01+03, Test on Signer 02
    Fold 3: Train on Signers 01+02, Test on Signer 03

For each fold:
  1. Combine train splits of the two training signers → unified train pool
  2. Extract 15% stratified validation from that pool
  3. Train the same BiLSTM architecture with the same hyperparameters
  4. Evaluate on the held-out signer's test split
  5. Save per-class metrics and confusion matrix

Final report: mean ± std of test accuracy across 3 folds.

Upload signbot_colab_package_loso.zip to Google Drive before running.
That ZIP must contain per-signer arrays:
    X_train_s01.npy, y_train_s01.npy, X_test_s01.npy, y_test_s01.npy
    X_train_s02.npy, y_train_s02.npy, X_test_s02.npy, y_test_s02.npy
    X_train_s03.npy, y_train_s03.npy, X_test_s03.npy, y_test_s03.npy
    class_names.npy

Build these per-signer files with build_dataset_loso.py (or manually
by running build_dataset.py once per signer and renaming the outputs).
"""

import os
import json
import random
import zipfile

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from google.colab import drive
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import (
    EarlyStopping,
    ModelCheckpoint,
    ReduceLROnPlateau,
)
from tensorflow.keras.layers import (
    Bidirectional,
    Dense,
    Dropout,
    Input,
    LSTM,
    Masking,
)
from tensorflow.keras.models import Sequential

# ─────────────────────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────────────────────
SEED         = 42
NUM_CLASSES  = 100
SEQUENCE_LEN = 20
NUM_FEATURES = 126
EPOCHS       = 100
BATCH_SIZE   = 32
VAL_SPLIT    = 0.15

random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

drive.mount("/content/drive")

ZIP_PATH    = "/content/drive/MyDrive/signbot/dataset/signbot_colab_package_loso.zip"
EXTRACT_DIR = "/content/dataset_loso"
RESULTS_DIR = "/content/drive/MyDrive/signbot/results/loso"
os.makedirs(RESULTS_DIR, exist_ok=True)

if not os.path.exists(EXTRACT_DIR):
    with zipfile.ZipFile(ZIP_PATH, "r") as zf:
        zf.extractall(EXTRACT_DIR)
    print("LOSO dataset extracted.")


# ─────────────────────────────────────────────────────────────────────────────
# Model factory — identical architecture for every fold
# ─────────────────────────────────────────────────────────────────────────────
def build_model(num_classes=NUM_CLASSES):
    """
    Build a fresh BiLSTM model.
    Called once per fold so weights are never shared between folds.
    """
    m = Sequential([
        Input(shape=(SEQUENCE_LEN, NUM_FEATURES)),
        Masking(mask_value=0.0),
        Bidirectional(LSTM(128, return_sequences=True, use_cudnn=False)),
        Dropout(0.3),
        Bidirectional(LSTM(64, return_sequences=False, use_cudnn=False)),
        Dropout(0.3),
        Dense(64, activation="relu"),
        Dense(num_classes, activation="softmax"),
    ])
    m.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return m


# ─────────────────────────────────────────────────────────────────────────────
# Load per-signer arrays
# ─────────────────────────────────────────────────────────────────────────────
def load_signer(signer_id, split):
    """Load X and y for one signer and split ('train' or 'test')."""
    tag  = f"s{signer_id:02d}"
    X    = np.load(os.path.join(EXTRACT_DIR, f"X_{split}_{tag}.npy"))
    y    = np.load(os.path.join(EXTRACT_DIR, f"y_{split}_{tag}.npy"))
    return X, y

class_names = np.load(
    os.path.join(EXTRACT_DIR, "class_names.npy"), allow_pickle=True
)


# ─────────────────────────────────────────────────────────────────────────────
# LOSO fold definitions
# ─────────────────────────────────────────────────────────────────────────────
FOLDS = [
    {"test_signer": 1, "train_signers": [2, 3]},
    {"test_signer": 2, "train_signers": [1, 3]},
    {"test_signer": 3, "train_signers": [1, 2]},
]

fold_results = []

for fold_num, fold in enumerate(FOLDS, start=1):
    test_signer    = fold["test_signer"]
    train_signers  = fold["train_signers"]

    print("\n" + "="*60)
    print(f"FOLD {fold_num}/3 — Test signer: {test_signer:02d} | "
          f"Train signers: {train_signers}")
    print("="*60)

    # ── Build train pool from the two training signers ────────────────────────
    X_parts, y_parts = [], []
    for sid in train_signers:
        Xs, ys = load_signer(sid, "train")
        X_parts.append(Xs)
        y_parts.append(ys)

    X_pool = np.concatenate(X_parts, axis=0)
    y_pool = np.concatenate(y_parts, axis=0)

    # ── Stratified validation split from train pool ───────────────────────────
    X_tr, X_val, y_tr, y_val = train_test_split(
        X_pool, y_pool,
        test_size=VAL_SPLIT,
        stratify=y_pool,
        random_state=SEED,
    )

    # ── Test set: held-out signer's test split ────────────────────────────────
    X_test, y_test = load_signer(test_signer, "test")

    print(f"  Train : {X_tr.shape}")
    print(f"  Val   : {X_val.shape}")
    print(f"  Test  : {X_test.shape}  (signer {test_signer:02d} — unseen during training)")

    # ── Build fresh model ─────────────────────────────────────────────────────
    tf.random.set_seed(SEED)  # reset seed per fold for reproducibility
    model = build_model()

    fold_model_path = os.path.join(
        RESULTS_DIR, f"fold{fold_num}_best.h5"
    )

    callbacks = [
        ModelCheckpoint(
            fold_model_path,
            monitor="val_loss",
            save_best_only=True,
            mode="min",
            verbose=0,
        ),
        ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=5,
            min_lr=1e-6,
            verbose=0,
        ),
        EarlyStopping(
            monitor="val_loss",
            patience=10,
            restore_best_weights=True,
            verbose=1,
        ),
    ]

    # ── Train ─────────────────────────────────────────────────────────────────
    history = model.fit(
        X_tr, y_tr,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        validation_data=(X_val, y_val),
        callbacks=callbacks,
        verbose=2,
    )

    # Persist history immediately — survives runtime disconnect/type change.
    fold_history_path = os.path.join(RESULTS_DIR, f"fold{fold_num}_history.json")
    with open(fold_history_path, "w") as f:
        json.dump(history.history, f)
    print(f"  History saved: {fold_history_path}")

    # ── Evaluate on unseen signer ─────────────────────────────────────────────
    # CPU device lock required: use_cudnn=False at build time is not reliably
    # restored by load_model(). Running on GPU re-selects the CuDNN kernel and
    # Bidirectional+Masking throws an Assert/Assert InvalidArgumentError.
    with tf.device('/CPU:0'):
        best_model = tf.keras.models.load_model(fold_model_path)

        zero_mask = np.all(X_test.reshape(X_test.shape[0], -1) == 0, axis=1)
        if zero_mask.sum() > 0:
            print(f"  Removing {zero_mask.sum()} fully-zero test sequences "
                  f"(signer {test_signer:02d})")
            X_test = X_test[~zero_mask]
            y_test = y_test[~zero_mask]

        test_loss, test_acc = best_model.evaluate(X_test, y_test, verbose=0)
    print(f"\n  Fold {fold_num} Test Accuracy  : {test_acc:.4f}")
    print(f"  Fold {fold_num} Test Loss      : {test_loss:.4f}")

    fold_results.append({
        "fold"      : fold_num,
        "test_signer": test_signer,
        "accuracy"  : test_acc,
        "loss"      : test_loss,
    })

    # ── Per-class report ──────────────────────────────────────────────────────
    with tf.device('/CPU:0'):
        y_pred = np.argmax(best_model.predict(X_test, verbose=0), axis=1)

    report = classification_report(
        y_test, y_pred, target_names=class_names, digits=4
    )
    report_path = os.path.join(RESULTS_DIR, f"fold{fold_num}_report.txt")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(f"Fold {fold_num} — Test Signer: {test_signer:02d}\n")
        f.write(f"Test Accuracy : {test_acc:.4f}\n")
        f.write(f"Test Loss     : {test_loss:.4f}\n\n")
        f.write(report)
    print(f"  Report saved: {report_path}")

    # ── Confusion matrix ──────────────────────────────────────────────────────
    cm      = confusion_matrix(y_test, y_pred)
    cm_norm = cm.astype("float") / cm.sum(axis=1, keepdims=True)

    plt.figure(figsize=(14, 12))
    sns.heatmap(cm_norm, annot=False, cmap="Blues",
                xticklabels=False, yticklabels=False)
    plt.title(f"Fold {fold_num} — Test Signer {test_signer:02d} "
              f"(Acc: {test_acc:.4f})", fontsize=12)
    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")
    plt.tight_layout()
    cm_path = os.path.join(RESULTS_DIR, f"fold{fold_num}_confusion_matrix.png")
    plt.savefig(cm_path, dpi=300)
    plt.close()
    print(f"  Confusion matrix saved: {cm_path}")

    # ── Learning curves ───────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(history.history["accuracy"],     label="Train")
    axes[0].plot(history.history["val_accuracy"], label="Val")
    axes[0].set_title(f"Fold {fold_num} Accuracy")
    axes[0].set_xlabel("Epoch")
    axes[0].legend()

    axes[1].plot(history.history["loss"],     label="Train")
    axes[1].plot(history.history["val_loss"], label="Val")
    axes[1].set_title(f"Fold {fold_num} Loss")
    axes[1].set_xlabel("Epoch")
    axes[1].legend()

    plt.tight_layout()
    curves_path = os.path.join(RESULTS_DIR, f"fold{fold_num}_learning_curves.png")
    plt.savefig(curves_path, dpi=300)
    plt.close()

# ─────────────────────────────────────────────────────────────────────────────
# Final LOSO summary
# ─────────────────────────────────────────────────────────────────────────────
accuracies = [r["accuracy"] for r in fold_results]
mean_acc   = np.mean(accuracies)
std_acc    = np.std(accuracies)

print("\n" + "="*60)
print("LOSO CROSS-VALIDATION SUMMARY")
print("="*60)
for r in fold_results:
    print(f"  Fold {r['fold']} (Test Signer {r['test_signer']:02d}): "
          f"Acc={r['accuracy']:.4f}  Loss={r['loss']:.4f}")
print(f"\n  Mean Accuracy : {mean_acc:.4f}")
print(f"  Std Deviation : {std_acc:.4f}")
print(f"\n  Report for thesis: {mean_acc*100:.2f}% ± {std_acc*100:.2f}%")
print("="*60)

summary_path = os.path.join(RESULTS_DIR, "loso_summary.txt")
with open(summary_path, "w") as f:
    f.write("LOSO Cross-Validation Summary\n\n")
    for r in fold_results:
        f.write(f"Fold {r['fold']} (Test Signer {r['test_signer']:02d}): "
                f"Acc={r['accuracy']:.4f}  Loss={r['loss']:.4f}\n")
    f.write(f"\nMean Accuracy : {mean_acc:.4f}\n")
    f.write(f"Std Deviation : {std_acc:.4f}\n")
    f.write(f"Report value  : {mean_acc*100:.2f}% +/- {std_acc*100:.2f}%\n")
print(f"Summary saved: {summary_path}")
