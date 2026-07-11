"""
Sign-Bot: TFLite Conversion  (Simplified — from_keras_model direct)
======================================================================
Run locally on your Mac.

Why this version is different
-------------------------------
Every previous attempt (manual @tf.function tracing, tf.saved_model.save
with a custom signature, with/without _experimental_lower_tensor_list_ops,
with/without implementation=2) produced a broken .tflite file — LSTM
weights not embedded, ~0.05 MB output, either a hard READ_VARIABLE crash
or silent near-uniform predictions on real data.

Crucially: the model currently deployed on the Pi has the SAME defect
(confirmed via real-data testing — 0% accuracy, ~0.01 confidence on every
sample). This bug has been present since the very first conversion attempt
in this project, across multiple TF versions and multiple conversion entry
points. That rules out version drift as the cause.

What every failed attempt has in common: a manually constructed
@tf.function / signature that WE trace ourselves, then hand to the
converter. This version skips that entirely. We already proved (Phase 2's
max_diff: 0.000000) that the model loaded directly from the .h5 file
produces byte-identical predictions in Python — the weights are correct
in memory. The fix: hand that already-loaded model straight to
TFLiteConverter.from_keras_model(), which has its own mature internal
tracing/freezing logic, and never touches our manual signature code at all.

Usage
-----
    conda activate action_detection
    python convert_tflite_simple.py
"""

import os
import json
import shutil
import signal
import time

import h5py
import numpy as np
import tensorflow as tf

print(f"TensorFlow version : {tf.__version__}")
print(f"GPU devices        : {tf.config.list_physical_devices('GPU')}\n")

# ─────────────────────────────────────────────────────────────────────────────
# Config — update these paths to match your local Mac filesystem.
# ─────────────────────────────────────────────────────────────────────────────
MODEL_H5_PATH = os.path.expanduser(
    "~/Desktop/signbot_pipeline/model/signbot_production_best.h5"
)
OUTPUT_DIR = os.path.expanduser("~/Desktop/signbot_pipeline/model")
LOCAL_OUT_PATH = os.path.join(OUTPUT_DIR, "signbot_production.tflite")
DATA_DIR = os.path.expanduser("~/Desktop/signbot_pipeline/output/dataset")

CONVERSION_TIMEOUT = 420
SEQUENCE_LEN = 20
NUM_FEATURES = 126

os.makedirs(OUTPUT_DIR, exist_ok=True)

if not os.path.exists(MODEL_H5_PATH):
    raise FileNotFoundError(f"Model checkpoint not found at: {MODEL_H5_PATH}")
print(f"Model checkpoint: {MODEL_H5_PATH}")
print(f"  Size: {os.path.getsize(MODEL_H5_PATH) / (1024**2):.1f} MB\n")

# ─────────────────────────────────────────────────────────────────────────────
# STEP 1 — Load the model directly, no rebuild
# Patch the H5 config to strip 'quantization_config' (Keras version schema
# mismatch, unrelated to the weight-embedding bug — same fix as before).
# ─────────────────────────────────────────────────────────────────────────────
print("[Step 1] Loading model directly (no CPU rebuild) …")

def _strip_quantization_config(obj):
    if isinstance(obj, dict):
        obj.pop("quantization_config", None)
        for v in obj.values():
            _strip_quantization_config(v)
    elif isinstance(obj, list):
        for item in obj:
            _strip_quantization_config(item)
    return obj

MODEL_H5_COMPAT_PATH = MODEL_H5_PATH.replace(".h5", "_compat.h5")
shutil.copy2(MODEL_H5_PATH, MODEL_H5_COMPAT_PATH)

with h5py.File(MODEL_H5_COMPAT_PATH, "r+") as f:
    raw_config = f.attrs["model_config"]
    if isinstance(raw_config, bytes):
        raw_config = raw_config.decode("utf-8")
    config_dict = json.loads(raw_config)
    _strip_quantization_config(config_dict)
    f.attrs["model_config"] = json.dumps(config_dict)

model = tf.keras.models.load_model(MODEL_H5_COMPAT_PATH, compile=False)
print("  Model loaded  ✓")
model.summary()

# Sanity check: confirm it actually predicts something non-trivial on a
# real sample before we even attempt conversion.
print("\n  Sanity check with a real sample …")
try:
    X_sample = np.load(os.path.join(DATA_DIR, "X_test_s02.npy"))[0:1]
    y_sample = np.load(os.path.join(DATA_DIR, "y_test_s02.npy"))[0]
    pred = model.predict(X_sample, verbose=0)[0]
    pred_class = int(np.argmax(pred))
    confidence = float(pred[pred_class])
    print(f"  True class: {y_sample}  Predicted: {pred_class}  Confidence: {confidence:.3f}")
    if confidence < 0.1:
        print("  WARNING: low confidence even in the Python model itself.")
        print("  This would indicate a training/weight problem, not a")
        print("  conversion problem. Stop here and investigate the .h5 file")
        print("  directly before attempting conversion.")
    else:
        print("  Model produces confident predictions in Python — good.")
        print("  Any conversion failure from here is a TFLite export issue,")
        print("  not a problem with the trained weights themselves.")
except FileNotFoundError:
    print("  Could not find local test data for sanity check — skipping.")
    print("  Proceeding with conversion anyway.")

# ─────────────────────────────────────────────────────────────────────────────
# STEP 2 — Convert directly with from_keras_model
# No manual tracing, no custom signature, no CPU rebuild. Let the converter's
# own internal handling do the freezing.
# ─────────────────────────────────────────────────────────────────────────────
print("\n[Step 2] Converting with TFLiteConverter.from_keras_model() …")
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.target_spec.supported_ops = [
    tf.lite.OpsSet.TFLITE_BUILTINS,
    tf.lite.OpsSet.SELECT_TF_OPS,
]
print("  Converter configured  ✓")

class ConversionTimeout(Exception):
    pass

def _handler(signum, frame):
    raise ConversionTimeout("Conversion timed out.")

signal.signal(signal.SIGALRM, _handler)
signal.alarm(CONVERSION_TIMEOUT)
t0 = time.time()

try:
    tflite_model = converter.convert()
    signal.alarm(0)
    size_mb = len(tflite_model) / 1_048_576
    print(f"  Converted in {time.time()-t0:.1f}s  |  size: {size_mb:.2f} MB")
except ConversionTimeout as e:
    raise RuntimeError(f"Conversion timed out: {e}") from e
except Exception as e:
    signal.alarm(0)
    raise RuntimeError(f"Conversion failed: {e}") from e

with open(LOCAL_OUT_PATH, "wb") as f:
    f.write(tflite_model)
print(f"  Written: {LOCAL_OUT_PATH}")

# ─────────────────────────────────────────────────────────────────────────────
# STEP 3 — Inspect tensors and test with real data immediately
# ─────────────────────────────────────────────────────────────────────────────
print("\n[Step 3] Inspecting converted model …")
interp = tf.lite.Interpreter(model_path=LOCAL_OUT_PATH)
interp.allocate_tensors()
tensor_details = interp.get_tensor_details()
print(f"  Total tensors: {len(tensor_details)}")

inp_det = interp.get_input_details()[0]
out_det = interp.get_output_details()[0]

print("\n[Step 4] Testing with real labeled data …")
try:
    X = np.load(os.path.join(DATA_DIR, "X_test_s02.npy"))
    y = np.load(os.path.join(DATA_DIR, "y_test_s02.npy"))
except FileNotFoundError:
    X = np.load(os.path.join(DATA_DIR, "X_test.npy"))
    y = np.load(os.path.join(DATA_DIR, "y_test.npy"))

n_test = min(20, len(X))
n_correct = 0
n_confident = 0

print(f"{'True':>4}  {'Pred':>4}  {'Conf':>6}  {'Correct'}")
print("-" * 40)
for i in range(n_test):
    sample = X[i:i+1].astype(np.float32)
    interp.set_tensor(inp_det["index"], sample)
    interp.invoke()
    output = interp.get_tensor(out_det["index"])[0]
    pred_class = int(np.argmax(output))
    confidence = float(output[pred_class])
    true_class = int(y[i])
    is_correct = pred_class == true_class
    if is_correct:
        n_correct += 1
    if confidence > 0.5:
        n_confident += 1
    print(f"{true_class:>4}  {pred_class:>4}  {confidence:>6.3f}  {'✓' if is_correct else ''}")

print("-" * 40)
print(f"\nAccuracy   : {n_correct}/{n_test} ({n_correct/n_test*100:.1f}%)")
print(f"Confident  : {n_confident}/{n_test}")

if n_confident == 0:
    print("\nSTILL BROKEN. Same defect as every previous attempt.")
    print("The from_keras_model direct path did not fix it either.")
else:
    print("\nWORKING. This conversion produced a functional model.")
    print(f"Deploy this file: {LOCAL_OUT_PATH}")
