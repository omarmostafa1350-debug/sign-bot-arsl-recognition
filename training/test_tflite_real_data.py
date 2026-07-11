"""
Test a .tflite model against REAL labeled test data.
=======================================================
Random noise input can't distinguish "model is appropriately uncertain
on garbage" from "model weights are effectively missing/zero". This
script feeds real, labeled hand-landmark sequences through the model
and checks whether predictions are confident and roughly correct.

Usage:
    python test_tflite_real_data.py /path/to/model.tflite

Requires: your local per-signer test arrays in output/dataset/
(X_test_s01.npy, y_test_s01.npy, etc. — the same ones used earlier
for accuracy measurement).
"""

import os
import sys
import numpy as np
import tensorflow as tf

if len(sys.argv) != 2:
    print("Usage: python test_tflite_real_data.py /path/to/model.tflite")
    sys.exit(1)

MODEL_PATH = sys.argv[1]

# Adjust this if your dataset folder is somewhere else
DATA_DIR = os.path.expanduser("~/Desktop/signbot_pipeline/output/dataset")

print(f"Model: {MODEL_PATH}")
print(f"Data : {DATA_DIR}\n")

# ── Load a handful of real, labeled samples ──────────────────────────────────
try:
    X = np.load(os.path.join(DATA_DIR, "X_test_s02.npy"))
    y = np.load(os.path.join(DATA_DIR, "y_test_s02.npy"))
    print(f"Loaded X_test_s02.npy: {X.shape}")
except FileNotFoundError:
    print("X_test_s02.npy not found — trying combined X_test.npy instead")
    X = np.load(os.path.join(DATA_DIR, "X_test.npy"))
    y = np.load(os.path.join(DATA_DIR, "y_test.npy"))
    print(f"Loaded X_test.npy: {X.shape}")

# Load class names if available for readable output
class_names = None
try:
    class_names = np.load(os.path.join(DATA_DIR, "class_names.npy"), allow_pickle=True)
except FileNotFoundError:
    pass

# ── Load interpreter ──────────────────────────────────────────────────────────
interp = tf.lite.Interpreter(model_path=MODEL_PATH)
interp.allocate_tensors()
inp_det = interp.get_input_details()[0]
out_det = interp.get_output_details()[0]

print(f"\nModel expects input shape: {inp_det['shape']}")
print(f"Model expects input dtype: {inp_det['dtype']}\n")

# ── Test on 20 real samples, skipping any fully-zero ones ────────────────────
n_test = min(20, len(X))
n_correct = 0
n_confident = 0  # top prediction > 0.5

print(f"{'True':>4}  {'Pred':>4}  {'Conf':>6}  {'Correct'}")
print("-" * 40)

for i in range(n_test):
    sample = X[i:i+1].astype(inp_det['dtype'])
    if sample.shape != tuple(inp_det['shape']):
        sample = np.reshape(sample, inp_det['shape'])

    interp.set_tensor(inp_det['index'], sample)
    interp.invoke()
    output = interp.get_tensor(out_det['index'])[0]

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
print(f"\nAccuracy on {n_test} real samples : {n_correct}/{n_test} ({n_correct/n_test*100:.1f}%)")
print(f"Predictions with >50% confidence  : {n_confident}/{n_test}")

if n_confident == 0:
    print("\nWARNING: No prediction exceeded 50% confidence on real labeled data.")
    print("This is consistent with missing/zeroed weights producing")
    print("near-uniform output regardless of input.")
elif n_correct / n_test > 0.5:
    print("\nModel appears to be functioning — confident and mostly correct")
    print("on real data. The earlier random-noise test was a false alarm;")
    print("uniform output on nonsense input is expected model behavior.")
else:
    print("\nModel is producing confident predictions but they're frequently")
    print("wrong. This needs closer inspection — could be a genuine accuracy")
    print("issue distinct from the weight-embedding question.")
