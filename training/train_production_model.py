# -*- coding: utf-8 -*-
"""
Sign-Bot: Production Model Training
=====================================
Run on Google Colab (T4 GPU) AFTER completing LOSO cross-validation.

Purpose
-------
Train the final model on ALL three signers combined.
This is the model you deploy on the Raspberry Pi 4.

LOSO folds proved your architecture generalizes across signers.
This script maximizes training data to get the best possible weights
for real-world deployment.

Pipeline
--------
1. Load per-signer train arrays and combine into one pool
2. Create stratified 15% validation split from the combined pool
3. Train BiLSTM with identical architecture and hyperparameters as LOSO
4. Save best checkpoint (monitored by val_loss)
5. Evaluate on combined test set (all three signers)
6. Convert best checkpoint to TFLite (see convert_tflite.py)

Requires: signbot_colab_package_loso.zip (same as LOSO script)
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
# 1. Seeds
# ─────────────────────────────────────────────────────────────────────────────
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)
print(f"Random seeds set to {SEED}")

# ─────────────────────────────────────────────────────────────────────────────
# 2. Paths
# ─────────────────────────────────────────────────────────────────────────────
drive.mount("/content/drive")

ZIP_PATH         = "/content/drive/MyDrive/signbot/dataset/signbot_colab_package_loso.zip"
EXTRACT_DIR      = "/content/dataset_loso"
MODEL_SAVE_PATH  = "/content/drive/MyDrive/signbot/model/signbot_production_best.h5"
RESULTS_DIR      = "/content/drive/MyDrive/signbot/results/production"

os.makedirs(RESULTS_DIR, exist_ok=True)

if not os.path.exists(EXTRACT_DIR):
    with zipfile.ZipFile(ZIP_PATH, "r") as zf:
        zf.extractall(EXTRACT_DIR)
    print("Dataset extracted.")

# ─────────────────────────────────────────────────────────────────────────────
# 3. Load and combine all three signers
# ─────────────────────────────────────────────────────────────────────────────
def load_signer(signer_id, split):
    tag = f"s{signer_id:02d}"
    X   = np.load(os.path.join(EXTRACT_DIR, f"X_{split}_{tag}.npy"))
    y   = np.load(os.path.join(EXTRACT_DIR, f"y_{split}_{tag}.npy"))
    return X, y

class_names = np.load(
    os.path.join(EXTRACT_DIR, "class_names.npy"), allow_pickle=True
)
NUM_CLASSES = len(class_names)

print("Loading per-signer arrays …")

# Combine train splits from all three signers
X_train_parts, y_train_parts = [], []
X_test_parts,  y_test_parts  = [], []

for sid in [1, 2, 3]:
    Xtr, ytr = load_signer(sid, "train")
    Xte, yte = load_signer(sid, "test")
    X_train_parts.append(Xtr)
    y_train_parts.append(ytr)
    X_test_parts.append(Xte)
    y_test_parts.append(yte)
    print(f"  Signer {sid:02d} — train: {Xtr.shape}  test: {Xte.shape}")

X_all_train = np.concatenate(X_train_parts, axis=0)
y_all_train = np.concatenate(y_train_parts, axis=0)
X_test_all  = np.concatenate(X_test_parts,  axis=0)
y_test_all  = np.concatenate(y_test_parts,  axis=0)

print(f"\nCombined train pool : {X_all_train.shape}")
print(f"Combined test set   : {X_test_all.shape}")

SEQUENCE_LEN = X_all_train.shape[1]   # 20
NUM_FEATURES = X_all_train.shape[2]   # 126

# ─────────────────────────────────────────────────────────────────────────────
# 4. Stratified validation split
# ─────────────────────────────────────────────────────────────────────────────
X_train, X_val, y_train, y_val = train_test_split(
    X_all_train, y_all_train,
    test_size=0.15,
    stratify=y_all_train,
    random_state=SEED,
)

print(f"\nTrain : {X_train.shape}")
print(f"Val   : {X_val.shape}")
print(f"Test  : {X_test_all.shape}  (held out — never used during training)")

# ─────────────────────────────────────────────────────────────────────────────
# 5. Build production model
# Same architecture as LOSO — do not change hyperparameters between experiments
# ─────────────────────────────────────────────────────────────────────────────
model = Sequential([
    Input(shape=(SEQUENCE_LEN, NUM_FEATURES)),
    Masking(mask_value=0.0),
    Bidirectional(LSTM(128, return_sequences=True, use_cudnn=False)),
    Dropout(0.3),
    Bidirectional(LSTM(64, return_sequences=False, use_cudnn=False)),
    Dropout(0.3),
    Dense(64, activation="relu"),
    Dense(NUM_CLASSES, activation="softmax"),
])

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
)

model.summary()

# ─────────────────────────────────────────────────────────────────────────────
# 6. Callbacks
# ─────────────────────────────────────────────────────────────────────────────
callbacks = [
    ModelCheckpoint(
        MODEL_SAVE_PATH,
        monitor="val_loss",
        save_best_only=True,
        mode="min",
        verbose=1,
    ),
    ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.5,
        patience=5,
        min_lr=1e-6,
        verbose=1,
    ),
    EarlyStopping(
        monitor="val_loss",
        patience=10,
        restore_best_weights=True,
        verbose=1,
    ),
]

# ─────────────────────────────────────────────────────────────────────────────
# 7. Train
# ─────────────────────────────────────────────────────────────────────────────
print("\nTraining production model …")
history = model.fit(
    X_train, y_train,
    epochs=100,
    batch_size=32,
    validation_data=(X_val, y_val),
    callbacks=callbacks,
)

# Persist history immediately — survives runtime disconnect/type change.
history_path = os.path.join(RESULTS_DIR, "production_history.json")
with open(history_path, "w") as f:
    json.dump(history.history, f)
print(f"Training history saved: {history_path}")

# ─────────────────────────────────────────────────────────────────────────────
# 8. Final test evaluation — evaluated ONCE on combined test set
# CPU device lock required: use_cudnn=False at build time is not reliably
# restored by load_model(). Running on GPU re-selects the CuDNN kernel and
# Bidirectional+Masking throws an Assert/Assert InvalidArgumentError.
# ─────────────────────────────────────────────────────────────────────────────
print("\nLoading best checkpoint …")
with tf.device('/CPU:0'):
    best_model = tf.keras.models.load_model(MODEL_SAVE_PATH)

    zero_mask = np.all(X_test_all.reshape(X_test_all.shape[0], -1) == 0, axis=1)
    n_zero = zero_mask.sum()
    print(f"Fully-zero test sequences found: {n_zero} / {len(X_test_all)}")
    if n_zero > 0:
        X_test_all = X_test_all[~zero_mask]
        y_test_all = y_test_all[~zero_mask]
        print(f"Removed {n_zero} empty sequences. Evaluating on {len(X_test_all)} samples.")

    test_loss, test_acc = best_model.evaluate(X_test_all, y_test_all, verbose=1)
print(f"\nProduction Model Test Accuracy : {test_acc:.4f}")
print(f"Production Model Test Loss     : {test_loss:.4f}")

# ─────────────────────────────────────────────────────────────────────────────
# 9. Per-class metrics
# ─────────────────────────────────────────────────────────────────────────────
with tf.device('/CPU:0'):
    y_pred = np.argmax(best_model.predict(X_test_all, verbose=0), axis=1)

report = classification_report(
    y_test_all, y_pred, target_names=class_names, digits=4
)
print(report)

report_path = os.path.join(RESULTS_DIR, "production_classification_report.txt")
with open(report_path, "w", encoding="utf-8") as f:
    f.write(f"Production Model\n")
    f.write(f"Test Accuracy : {test_acc:.4f}\n")
    f.write(f"Test Loss     : {test_loss:.4f}\n\n")
    f.write(report)
print(f"Report saved: {report_path}")

# ─────────────────────────────────────────────────────────────────────────────
# 10. Confusion matrix
# ─────────────────────────────────────────────────────────────────────────────
cm      = confusion_matrix(y_test_all, y_pred)
cm_norm = cm.astype("float") / cm.sum(axis=1, keepdims=True)

plt.figure(figsize=(22, 20))
sns.heatmap(cm_norm, annot=False, cmap="Blues",
            xticklabels=False, yticklabels=False)
plt.title(f"Production Model — Normalized Confusion Matrix  "
          f"(Test Acc: {test_acc:.4f})", fontsize=14)
plt.ylabel("True Label")
plt.xlabel("Predicted Label")
plt.tight_layout()
cm_path = os.path.join(RESULTS_DIR, "production_confusion_matrix.png")
plt.savefig(cm_path, dpi=300)
plt.show()
print(f"Confusion matrix saved: {cm_path}")

# ─────────────────────────────────────────────────────────────────────────────
# 11. Learning curves
# Load from saved JSON — works even in a fresh runtime after disconnect.
# ─────────────────────────────────────────────────────────────────────────────
history_path = os.path.join(RESULTS_DIR, "production_history.json")
with open(history_path, "r") as f:
    history_dict = json.load(f)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

axes[0].plot(history_dict["accuracy"],     label="Train")
axes[0].plot(history_dict["val_accuracy"], label="Val")
axes[0].set_title("Production Model — Accuracy")
axes[0].set_xlabel("Epoch")
axes[0].set_ylabel("Accuracy")
axes[0].legend()

axes[1].plot(history_dict["loss"],     label="Train")
axes[1].plot(history_dict["val_loss"], label="Val")
axes[1].set_title("Production Model — Loss")
axes[1].set_xlabel("Epoch")
axes[1].set_ylabel("Loss")
axes[1].legend()

plt.tight_layout()
curves_path = os.path.join(RESULTS_DIR, "production_learning_curves.png")
plt.savefig(curves_path, dpi=300)
plt.show()
print(f"Learning curves saved: {curves_path}")

print("\n" + "="*60)
print("Production model training complete.")
print(f"  Best checkpoint : {MODEL_SAVE_PATH}")
print(f"  Test Accuracy   : {test_acc:.4f}")
print(f"  Next step       : run convert_tflite.py to produce .tflite for Pi")
print("="*60)
