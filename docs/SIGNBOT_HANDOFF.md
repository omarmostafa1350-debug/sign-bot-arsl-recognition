# SIGNBOT_HANDOFF.md

**Complete Technical Handoff Document — Full Project Scope**
*Generated to allow any AI assistant to continue this work with zero prior context.*

---

## 1. Document Scope

This document covers the complete Machine Learning and deployment-validation work for Sign-Bot, an Arabic Sign Language recognition system built as an undergraduate Mechatronics Engineering graduation project (AAST, Cairo). It spans:

- Dataset extraction and preprocessing (MediaPipe landmark pipeline)
- Model training, correction of methodological bugs, and Leave-One-Signer-Out (LOSO) cross-validation
- Production model training
- TensorFlow Lite conversion, including an extended debugging investigation that uncovered a critical, previously undetected deployment failure
- Real-time validation on local hardware (MacBook Air M1 webcam), including Arabic text rendering
- Academic thesis documentation (a Word document chapter section built to match the student's existing thesis formatting)

It does **not** cover: Raspberry Pi hardware wiring, the PyQt5 dashboard GUI implementation details beyond what was directly reviewed, Text-to-Speech/Speech-to-Text subsystems, or non-ML business/coaching/finance topics the student has worked on in unrelated conversations.

---

## 2. Project Overview

**Objective:** Sign-Bot recognizes Arabic Sign Language signs in real time via a camera feed and outputs the recognized sign, deployed on a Raspberry Pi 4 as part of an educational humanoid robot.

**ML pipeline, current corrected state:**
```
KArSL-502 dataset videos (100 classes, 3 signers, 15,099 videos)
    ↓
MediaPipe Hands landmark extraction (126 features/frame: 21 landmarks × 3 coords × 2 hands)
    ↓
Wrist-relative, scale-invariant normalization (shared preprocessing.py module)
    ↓
Fixed 20-frame sequences (pad/truncate)
    ↓
Bidirectional LSTM classifier (Masking → BiLSTM(128) → BiLSTM(64) → Dense(64) → Dense(100, softmax))
    ↓
Training: single-pool baseline, LOSO cross-validation (3 folds), production model (all 3 signers)
    ↓
TFLite conversion (TFLiteConverter.from_keras_model, dynamic range quantization, Flex delegate)
    ↓
Deployment: Raspberry Pi 4 (real-time), validated first on Mac webcam
```

**Model input:** `(20, 126)` float32 tensor per sample.
**Model output:** 100-class softmax probability vector.

---

## 3. Current Development Status

| Module | Status | Notes |
|---|---|---|
| Dataset extraction (`extract_hands.py`) | Complete | MediaPipe Hands, 126 features, detector reset per video |
| Dataset building (`build_dataset.py`) | Complete | Validation split added, normalization shared via `preprocessing.py` |
| Baseline single-pool training | Complete | 96.29% test accuracy, 0.1636 test loss |
| LOSO cross-validation (3 folds) | Complete | 36.11% ± 2.70% mean accuracy |
| Production model training | Complete, but final accuracy/loss numbers were never explicitly reported by the user despite being requested multiple times | 440,228 params, 5.1 MB `.h5` checkpoint |
| TFLite conversion | **Complete and confirmed working**, after an extensive debugging investigation | Final working file: 0.56 MB (584 KB) |
| Real-time Mac validation | Complete | Live webcam test with Arabic rendering, confirmed functional by user ("the model is very good really") |
| Raspberry Pi physical deployment | Not confirmed in this conversation | User was advised to copy the corrected `.tflite` file and test physically; no result was reported back |
| Thesis Section 6.5 documentation | Complete, in `.docx` format matching existing thesis styling | Three figure placeholders remain unfilled (see Section 15) |

---

## 4. Dataset Information

**Dataset:** KArSL-502 (Arabic Sign Language), first 100 of 502 classes used, due to Raspberry Pi 4 hardware constraints.

**Signers:** 3 signers, each performing all 100 signs. Total 15,099 source videos.

**Label structure:** Sign IDs 1–100 map to Arabic labels via a user-built Excel-derived mapping. IDs 1–31 are digits, tens, hundreds, and large numbers (stored as literal numeric strings, e.g., `"0"`, `"1000000"`). IDs 71+ are compound anatomical/medical Arabic terms (e.g., "هيكل عظمي" = skeleton).

**Data quality observations (from actual per-signer build runs):**
- Signer 01: mean sequence length 29.7 frames, 76.7% truncated at the 20-frame cap, skip rate 2.3% (heavily concentrated in class 31, "10,000,000")
- Signer 02: mean length 24.8 frames, 42.3% truncated, skip rate 0.2% (cleanest signer)
- Signer 03: mean length 37.0 frames, 95.5% truncated, skip rate 7.9% (noisiest signer, broad-based skip pattern, not concentrated in one class)

These per-signer differences are relevant context for interpreting LOSO fold-to-fold variation (Section 9).

---

## 5. Complete Pipeline Detail

### 5.1 Extraction — `extract_hands.py`

Originally used MediaPipe **Holistic** (159 features: 33 pose + 126 hands). **Corrected** to MediaPipe **Hands** (126 features, no pose), matching the deployment inference engine and eliminating a training/inference domain mismatch.

```python
mp.solutions.hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    model_complexity=0,
    min_detection_confidence=0.6,
    min_tracking_confidence=0.5,
)
```

A fresh detector instance is created **per video** (via context manager) rather than one shared instance across all 15,099 videos, preventing MediaPipe's internal tracker state from bleeding between unrelated clips.

Output: one `.npy` file per video, shape `(num_frames, 126)`, saved to `output/npy/{split}/{signer}/{sign_id}/`.

### 5.2 Normalization — `preprocessing.py`

Shared module, imported by both the dataset builder and (intended for) the Pi's inference engine, guaranteeing training/inference normalization can never diverge.

```python
def normalize_hand_landmarks(hand_row, num_points=21):
    if np.all(hand_row == 0):
        return hand_row
    coords = hand_row.reshape(num_points, 3)
    wrist = coords[0].copy()
    coords = coords - wrist                      # translation invariant
    distances = np.linalg.norm(coords, axis=1)
    max_dist = distances.max()
    if max_dist > 1e-6:
        coords = coords / max_dist                # scale invariant
    return coords.flatten().astype(np.float32)
```

Applied per hand, per frame. No global dataset statistics used anywhere — zero leakage risk.

### 5.3 Dataset Building — `build_dataset.py`

- Pose-stripping step removed (extraction no longer produces pose).
- Stratified 15% validation split added, carved from training data only:
```python
X_train, X_val, y_train, y_val = train_test_split(
    X_train_all, y_train_all, test_size=0.15, stratify=y_train_all, random_state=42,
)
```
- Sequences padded/truncated to 20 frames (front-truncation for long sequences — flagged as a potential limitation, never re-validated against actual frame-length distributions).
- Fully-zero sequences (no hand detected in any frame) are filtered out and skip counts are reported per class.

For LOSO, this script is run **three times**, once per signer, with `SIGNERS = ["01"]`, `["02"]`, `["03"]` in `config_training.py`, and outputs renamed to `X_train_s01.npy` etc.

### 5.4 Model Architecture

```python
model = Sequential([
    Input(shape=(20, 126)),
    Masking(mask_value=0.0),
    Bidirectional(LSTM(128, return_sequences=True, use_cudnn=False)),
    Dropout(0.3),
    Bidirectional(LSTM(64, return_sequences=False, use_cudnn=False)),
    Dropout(0.3),
    Dense(64, activation="relu"),
    Dense(100, activation="softmax"),
])
```

**Critical, non-optional detail:** `use_cudnn=False` on both LSTM layers. Combining a `Masking` layer with a CuDNN-accelerated Bidirectional LSTM throws `InvalidArgumentError: ... RNN mask that does not correspond to right-padded sequences, while using cuDNN, which is not supported` under the TensorFlow/Keras versions used throughout this project (both Colab and local Mac). This was discovered via a training-time crash and confirmed as the fix; do not remove it.

Total parameters: 440,228 (1.68 MB uncompressed float32).

Optimizer: Adam, lr=0.001. Loss: sparse categorical crossentropy. Seeds: 42 (`random`, `numpy`, `tensorflow`, set before any model construction).

### 5.5 Training

`ModelCheckpoint(monitor="val_loss", mode="min")`, `ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=5)`, `EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True)`. All monitor the **validation split**, never the test set (this was a fixed bug — see Section 13).

**Critical operational lesson:** `history.history` (the Keras training-history object) exists only in the live Python kernel's memory. If a Colab runtime disconnects or is switched (e.g., T4 GPU → CPU), this object is permanently lost with no way to regenerate it short of retraining. The fix applied throughout: immediately after `model.fit()` completes, `json.dump(history.history, f)` to a JSON file, and all learning-curve plotting cells load from that JSON rather than the live object.

---

## 6. Evaluation Results (all numbers confirmed during this conversation, none invented)

### 6.1 Single-pool baseline

| Metric | Value |
|---|---|
| Test Accuracy | 96.29% |
| Test Loss | 0.1636 |
| Best epoch | 34 (early stopping fired at epoch 44) |

This is trained/tested on a pool combining all 3 signers — it does **not** demonstrate cross-signer generalization.

### 6.2 LOSO Cross-Validation

| Fold | Train signers | Test signer (unseen) | Accuracy | Loss |
|---|---|---|---|---|
| 1 | 02, 03 | 01 | 35.96% | 5.7041 |
| 2 | 01, 03 | 02 | 32.87% | 7.6121 |
| 3 | 01, 02 | 03 | 39.49% | 5.0993 |
| **Mean ± SD** | | | **36.11% ± 2.70%** | |

Random-classifier baseline loss for 100 classes: `ln(100) ≈ 4.605`. All three folds exceed this, meaning the model is not just uncertain on unseen signers but confidently wrong in a systematic direction — consistent with the architecture partially learning signer-specific motion characteristics rather than fully signer-invariant representations. The tight standard deviation (±2.70%) indicates this is a general limitation, not one problematic signer.

**This is the single most important limitation to state explicitly in any write-up of this project**: same-pool accuracy (96.29%) does not predict performance on a genuinely new user. Recommended future work: LOSO with 5–6 signers instead of 3.

### 6.3 Production Model

| Metric | Value |
|---|---|
| Parameters | 440,228 |
| Checkpoint size | 5.1 MB (`.h5`) |
| Final test accuracy | **Never explicitly obtained in this conversation — must be retrieved from the training log or Drive results before use in any report** |
| Final test loss | Same — not recorded |

### 6.4 TFLite Conversion — Final Working Result

| Metric | Value |
|---|---|
| Output file size | 0.56 MB (584 KB) |
| Quantization | Dynamic range (`Optimize.DEFAULT`) |
| Required ops | `TFLITE_BUILTINS` + `SELECT_TF_OPS` (Flex delegate) |
| Real-data spot-check (20 samples) | 16/20 correct (80%), 19/20 above 50% confidence |

This was a smoke test, not a full-test-set accuracy measurement. A comprehensive TFLite-vs-Keras accuracy comparison across the entire test set was never performed.

---

## 7. TFLite Conversion — Full Debugging Narrative (critical, do not skip)

This is one of the most important parts of the project history and directly affects what is safe to deploy.

**Initial symptom:** every conversion attempt produced a `.tflite` file of only 0.05–0.06 MB (versus an expected ~0.4–0.5 MB for this architecture under dynamic-range quantization), and either crashed at inference with `RuntimeError: tensorflow/lite/kernels/read_variable.cc:67 variable != nullptr was not true` or, in one variant, ran without crashing but produced near-uniform ~1% confidence on every class regardless of real input.

**Root cause, confirmed via tensor inspection:** LSTM kernel and recurrent-kernel weight matrices were absent from the exported graph entirely — present only as dangling resource-variable references, not embedded constants.

**Attempts that did NOT fix it** (in order tried):
1. Manual `@tf.function` tracing + `TFLiteConverter.from_concrete_functions()` — original approach, broken.
2. Removing an inherited `_experimental_lower_tensor_list_ops = False` flag — no change.
3. Adding `implementation=2` to both LSTM layers (matching a difference spotted in an older, possibly-also-broken conversion script) — no change.
4. Exporting via `tf.saved_model.save()` with a manually specified serving signature, then `from_saved_model()` — no change.

**What finally worked:** abandoning all manual tracing entirely and converting directly from the loaded Keras model:
```python
converter = tf.lite.TFLiteConverter.from_keras_model(model)
```
This is TFLite's simplest, most standard entry point. It correctly resolved all 16 resource-variable captures present in the model's graph (visible in the Colab log as 16 `TensorSpec(shape=(), dtype=tf.resource, ...)` captures during export). The resulting file was ~10x larger than every previous attempt and, critically, passed both a tensor-inspection check (real 2D weight-shaped tensors of expected size present) and a real-data test (80% accuracy, confident predictions, not the flat ~1% pattern).

**Separate but related environment issue encountered along the way:** Colab's TF build drifted to 2.20.0 mid-session, which broke Flex delegate registration for `tf.lite.Interpreter` (`RuntimeError: Select TensorFlow op(s) ... not supported by this interpreter`), even for a correctly-converted model. Attempts to fix this in Colab (switching to `ai_edge_litert.Interpreter`, installing `tensorflow-text`, manually loading the Flex delegate `.so`, pinning `tensorflow==2.17.0`) either failed or were derailed by a network timeout downloading the pinned version. **The eventual solution was to abandon Colab for the conversion step entirely and run it locally** on the user's MacBook Air M1 (conda env `action_detection`, TensorFlow 2.19.1, confirmed no Flex-registration issue on this platform/version combination).

**A second, unrelated bug hit during local conversion:** loading the `.h5` checkpoint via `tf.keras.models.load_model()` failed with `TypeError: Unrecognized keyword arguments passed to Dense: {'quantization_config': None}`. This is a Keras version schema mismatch — the checkpoint was saved by a newer Keras that includes a `quantization_config` field in its layer serialization format, which the local, older Keras bundled with TF 2.19.1 doesn't recognize. **Fix:** a small script patches a *copy* of the `.h5` file's stored JSON config (`f.attrs["model_config"]`), recursively stripping the `quantization_config` key, before calling `load_model()` on the patched copy. The original checkpoint file is never modified. `load_weights(by_name=True)` was unaffected by this issue since it doesn't touch the architecture config at all, only matches weight arrays by layer name.

### 7.1 Critical discovery: the model already deployed on the Raspberry Pi was never functional

During this debugging process, the **original** `signbot_arabic_v2.tflite` file (from an earlier, pre-this-conversation session, already sitting on the Pi) was tested against real labeled data for the first time. It showed the **identical defect**: 0% accuracy, near-uniform ~1% confidence on every one of 20 real test samples, with predictions clustering on the same 1-2 classes regardless of the true label (classic signature of a model whose recurrent layers contribute nothing, output driven only by fixed Dense-layer biases). **The Raspberry Pi has, at least as of this conversation, never performed real sign recognition, despite having a deployed model that passed its original conversion smoke test.** The original conversion script's smoke-test error handling was non-fatal (`except: print("WARNING — smoke-test failed, file still saved")`), which is why this went undetected.

**Action required, not yet confirmed done:** copy the new, verified-working `signbot_production.tflite` (Section 6.4 of this document) to the Pi, replacing the broken file, and physically test it on the actual hardware. This was advised but no confirmation of physical Pi testing was received in this conversation.

---

## 8. Real-Time Validation (Mac webcam)

Script: `test_webcam_live.py`. Mirrors the Pi's intended real-time inference loop as closely as possible without requiring the physical device:

- MediaPipe Hands extraction at the same settings as training (`model_complexity=0`, confidence thresholds 0.6/0.5)
- Sliding window buffer, `maxlen=20`, cleared when hands transition from absent to visible (gesture-boundary heuristic)
- Inference every 3 frames once buffer is full
- Two-tier confidence gating: ≥0.80 shows the sign name, 0.60–0.80 shows it prefixed with `~`, below 0.60 shows `...`
- On-screen debug overlay shows which hand MediaPipe currently labels "Left"/"Right", used to verify no camera-mirroring mismatch exists between training and inference (this had been flagged much earlier in the project as unverified; the live test confirmed it directly)

**Arabic text rendering:** `cv2.putText` cannot render Arabic script at all (no right-to-left support, no contextual letter shaping). Text is rendered through `arabic_reshaper.reshape()` (fixes letter connection forms) → `bidi.algorithm.get_display()` (fixes RTL ordering) → PIL `ImageDraw` (actually draws the Unicode glyphs, using macOS's built-in Geeza Pro font at `/System/Library/Fonts/GeezaPro.ttc`) → converted back to an OpenCV BGR frame.

**A number-rendering issue was investigated but never fully root-caused.** Symptom: performing a number sign showed nothing on screen at all, not even the confidence percentage (which rules out a pure Arabic-rendering bug, since the confidence value is plain digits drawn through the identical code path). A defensive fix was applied: `draw_text()` now wraps its rendering logic in try/except, printing the real exception to the terminal and falling back to plain ASCII `cv2.putText` rather than silently failing; both prediction-text branches now explicitly cast to `str()`; a startup diagnostic checks `class_names` for mixed Python types (in case digit-only labels like `"0"`, `"1000000"` were exported as actual numbers rather than strings somewhere upstream). **The user's final report was "the model is very good really" without confirming which hypothesis was correct or whether the issue recurred.** Treat this as resolved-in-practice but not definitively diagnosed.

---

## 9. Source Files

| File | Runs on | Purpose | Status |
|---|---|---|---|
| `config_training.py` | Mac | All constants/paths for training-side scripts | Complete |
| `preprocessing.py` | Mac + intended for Pi | Shared normalization, single source of truth | Complete |
| `extract_hands.py` | Mac | MediaPipe Hands extraction, per-video detector reset | Complete, executed |
| `build_dataset.py` | Mac | Normalizes, pads/truncates, creates val split, packages ZIP | Complete, executed (both single-pool and per-signer LOSO modes) |
| `train_model.py` / `.ipynb` | Colab T4 | Baseline single-pool training | Complete, executed — 96.29% |
| `loso_cross_validation.py` / `.ipynb` | Colab T4 | 3-fold LOSO | Complete, executed — 36.11% ± 2.70% |
| `train_production_model.py` / `.ipynb` | Colab T4 | Final deployment model, all 3 signers | Complete, executed — exact accuracy not recorded |
| `convert_tflite.py` / `.ipynb` | Colab | Original conversion attempt (manual tracing) | Superseded — kept only for reference, do not use |
| `convert_tflite_local.py` | Mac | First local conversion attempt (SavedModel approach) | Superseded — did not resolve the missing-weights bug |
| `convert_tflite_simple.py` | Mac | **Working conversion script** — `from_keras_model()` direct, includes H5 config patch, weight verification, tensor inspection, real-data test | **This is the correct script to use for any future conversion** |
| `check_tflite_weights.py` | Mac | Standalone diagnostic: inspects any `.tflite` file's tensors for missing LSTM weights, tests real inference | Used to confirm both the broken and working conversions |
| `test_tflite_real_data.py` | Mac | Tests a `.tflite` file against real labeled data (not random noise), reports accuracy/confidence per sample | Used to discover the Pi's deployed model was non-functional |
| `test_webcam_live.py` | Mac | Live webcam validation with Arabic rendering | Complete, confirmed working by user |

---

## 10. Configuration Reference

```python
# config_training.py
SSD_ROOT = "<path to raw KArSL videos>"
NPY_DIR = "output/npy"
DATASET_DIR = "output/dataset"
NUM_CLASSES = 100
SIGNERS = ["01", "02", "03"]          # set to single signer for LOSO array building
HANDS_MODEL_COMPLEXITY = 0
MIN_DETECTION_CONFIDENCE = 0.6
MIN_TRACKING_CONFIDENCE = 0.5
FRAME_WIDTH = 320
FRAME_HEIGHT = 240
MAX_SEQUENCE_LENGTH = 20
RANDOM_SEED = 42
VALIDATION_SPLIT = 0.15
```

**Local conversion paths (Mac):**
```python
MODEL_H5_PATH = "~/Desktop/signbot_pipeline/model/signbot_production_best.h5"
OUTPUT_DIR = "~/Desktop/signbot_pipeline/model"
DATA_DIR = "~/Desktop/signbot_pipeline/output/dataset"
```

**Environment:** conda env `action_detection`, TensorFlow 2.19.1, macOS arm64 (M1). Training happens on Google Colab T4 GPU; the Mac is deliberately kept off training duty to avoid thermal load, used only for extraction, dataset building, and the final conversion/validation steps.

---

## 11. Design Decisions and Rationale

| Decision | Why |
|---|---|
| MediaPipe Hands over Holistic | Matches deployment detector; eliminates train/inference domain mismatch |
| Wrist-relative + scale-invariant normalization, computed per-frame with no global stats | Zero leakage risk; translation and scale invariance without needing dataset-wide statistics |
| `use_cudnn=False` on all LSTM layers | Required to avoid a hard crash when combined with `Masking`; confirmed necessary across every environment tested (Colab and local Mac) |
| Validation split carved from training data only, test set touched once | Fixes a confirmed test-set-leakage bug in the original pipeline |
| LOSO with 3 folds, one per signer | Only 3 signers available; this is a stated limitation, not an ideal design |
| Production model trained on all 3 signers combined, separate from LOSO fold models | LOSO folds exist purely to report a generalization metric; none are deployment candidates since each was deliberately withheld from one signer |
| `TFLiteConverter.from_keras_model()` over any manual tracing approach | The only approach, among five tested, that correctly embeds LSTM weights rather than leaving dangling variable references |
| TFLite conversion moved from Colab to local Mac | Colab's TF 2.20 environment had an unresolved Flex delegate registration issue; local Mac (TF 2.19.1) did not exhibit it |
| Arabic text rendered via arabic_reshaper + python-bidi + PIL | `cv2.putText` has no Arabic script support at all |

---

## 12. Remaining Issues / Open Items

1. **Production model's exact test accuracy and loss were never obtained.** Needed before this can be cited in any report or thesis document.
2. **Physical Raspberry Pi deployment was never confirmed in this conversation.** The corrected `.tflite` file needs to be copied over and tested on real hardware; the previously deployed file is confirmed non-functional.
3. **Full-test-set TFLite-vs-Keras accuracy comparison was never run** — only a 20-sample spot-check exists. A proper quantization-drop measurement across the entire test set is still needed.
4. **The webcam number-rendering issue was patched defensively but never definitively root-caused.** If it recurs, check the terminal for the actual printed exception (the defensive wrapper now surfaces this) and check the startup `class_names` type-consistency diagnostic output.
5. **Raspberry Pi inference latency has never been measured**, only Colab-CPU and Mac-CPU latency during testing.
6. **The LOSO generalization gap (96.29% same-pool vs. 36.11% unseen-signer) is a fundamental, stated limitation**, not a bug to fix — expanding to 5–6 signers is the recommended mitigation, not yet attempted.
7. **Frame-length statistics and the 20-frame sequence length choice were never re-validated** against the actual per-signer distributions gathered during dataset building, despite that data being available.

---

## 13. Bug-Fix History (chronological, condensed)

1. **Test-set-as-validation leakage** — `model.fit(validation_data=(X_test, y_test))` let callbacks respond to test performance. Fixed by carving a proper validation split and touching the test set only once, post-training.
2. **CuDNN + Masking incompatibility** — hard crash on `model.fit()`. Fixed with `use_cudnn=False` on both LSTM layers, confirmed necessary in every environment.
3. **Same crash recurring at evaluation time** — `load_model()` doesn't reliably preserve `use_cudnn=False`. Fixed first with a `tf.device('/CPU:0')` wrapper (later found to be insufficient — intermittent failure observed), then with a more robust explicit-architecture-rebuild-plus-`load_weights()` pattern.
4. **`history.history` lost on Colab runtime disconnect/type-switch** — no disk backing. Fixed by `json.dump`-ing history immediately after every `fit()` call in all three training scripts.
5. **MediaPipe Holistic vs Hands domain mismatch** — training and (out-of-scope) deployment used different detectors. Fixed by switching extraction to Hands.
6. **TFLite conversion producing files with missing LSTM weights** — see full narrative in Section 7. Fixed by using `from_keras_model()` directly instead of any manual tracing approach.
7. **Keras `quantization_config` schema mismatch** on local Mac `load_model()` calls — fixed via an H5 config-patching script that strips the unrecognized key from a copy of the file.
8. **Colab Flex delegate registration broken** under TF 2.20 — worked around by moving TFLite conversion to a local Mac environment (TF 2.19.1) instead of continuing to fight the Colab environment.
9. **The originally-deployed Pi model was silently non-functional** — discovered via real-data testing, not caught by the original (non-fatal) smoke test. Not a code bug in this conversation's scripts, but a critical finding about prior deployment state.

---

## 14. Thesis Documentation Work

A separate deliverable was produced alongside the ML work: **Section 6.5** of the student's undergraduate thesis, documenting this entire pipeline-correction and validation effort, as a `.docx` file matching the exact formatting of the student's existing Chapter 6 (Times New Roman, US Letter page size, 1-inch margins, matching heading numbering convention 6.5, 6.5.1, 6.5.2, etc.).

**Build method:** generated programmatically via the `docx` npm package (not hand-edited in Word), reading the actual XML of the student's existing chapter to match font/page/margin settings exactly, then rendered to PDF and visually verified page-by-page before delivery.

**Structure (as of the final version produced in this conversation):**
- 6.5.1 Motivation for Pipeline Revision
- 6.5.2 Identified Issues in the Original Pipeline — **written as narrative prose** (not a table), six problems each explained as its own paragraph
- 6.5.3 Corrected Feature Extraction Pipeline
- 6.5.4 Corrected Dataset Construction (Figure 6.5.4.1 — dataset statistics — **placeholder still open, no image provided**)
- 6.5.5 Corrected Training Methodology, with baseline results table and Figures 6.5.5.1–6.5.5.3 (learning curves, confusion matrix, per-class accuracy — **images embedded**)
- 6.5.6 Leave-One-Signer-Out Cross-Validation, with per-fold results table and Figures 6.5.6.1–6.5.6.7 (individual per-fold confusion matrices and learning curves, plus the accuracy bar chart — **all images embedded**, restructured from an original single combined placeholder into 7 individual figures once the user supplied separate images for each)
- 6.5.7 Production Model Training, with metrics table (accuracy/loss cells still marked `[insert from training log]`) and Figures 6.5.7.1–6.5.7.2 (**images embedded**)
- 6.5.8 TFLite Conversion: Debugging and Resolution — **written as narrative prose** (not a table), describing the systematic elimination of candidate causes; small results table for final metrics kept. Figure 6.5.8.1 (terminal diagnostic output) — **placeholder still open, no image provided**
- 6.5.9 Real-Time Validation and Arabic Text Rendering. Figure 6.5.9.1 (webcam screenshot) — **placeholder still open, no image provided**
- 6.5.10 Summary of Results — **written as a synthesizing closing paragraph** (originally a table, converted to prose per user feedback that the document felt too table-heavy for thesis-quality writing)
- 6.5.11 Limitations and Future Work

**Revision history of this document within the conversation:**
1. First version: all placeholders open (no images existed yet), several sections table-heavy.
2. User supplied 12 PNG images (exported from the actual Colab training runs); all corresponding placeholders were replaced with real embedded, centered images with proper figure captions. This required restructuring Section 6.5.6 from one combined figure into 7 individual ones, since the user provided separate confusion-matrix and learning-curve images per fold rather than one combined figure as originally drafted.
3. User feedback: the document "doesn't feel professional" due to excessive table use for qualitative content. Two large tables (issues list, debugging attempts) were rewritten as flowing narrative prose; the closing summary table was converted to a synthesizing paragraph; only four small, genuinely numeric results tables were kept (baseline metrics, LOSO per-fold results, production metrics, TFLite conversion results). Two sections also received new interpretive paragraphs not present before (a discussion of what 96.29% baseline accuracy does and doesn't prove, and a discussion of what the per-fold confusion matrix patterns reveal about signer-specific vs. sign-invariant learning).

**Still outstanding on this document:** the three unfilled figure placeholders (6.5.4.1, 6.5.8.1, 6.5.9.1) and the production model's exact accuracy/loss numbers in Section 6.5.7's table and referenced in 6.5.10.

---

## 15. Quick Resume Prompt

I'm continuing work on Sign-Bot, an Arabic Sign Language recognition system (100 classes, KArSL-502 dataset, 3 signers, MediaPipe Hands → BiLSTM → TFLite, deployed on Raspberry Pi 4). Below is the complete state.

**Model:** `Masking → BiLSTM(128, use_cudnn=False) → Dropout(0.3) → BiLSTM(64, use_cudnn=False) → Dropout(0.3) → Dense(64) → Dense(100, softmax)`. `use_cudnn=False` is mandatory — omitting it crashes with a masking-related `InvalidArgumentError` in every environment tested.

**Confirmed results:** single-pool baseline 96.29% test accuracy (0.1636 loss). LOSO 3-fold cross-validation: 36.11% ± 2.70% (Fold 1/signer 01: 35.96%, Fold 2/signer 02: 32.87%, Fold 3/signer 03: 39.49%) — this large gap versus the single-pool number is the project's central limitation, only 3 signers total means limited pressure to learn signer-invariant features. Production model (all 3 signers, deployment target): trained successfully, 440,228 params, 5.1 MB checkpoint, but **exact final test accuracy/loss was never recorded and must be retrieved from Drive logs before citing anywhere**.

**TFLite conversion — critical history:** every manual-tracing-based conversion approach (concrete function tracing, SavedModel export with custom signature, various converter flags) produced a broken file with LSTM weights missing entirely (dangling variable references, not embedded constants) — symptom was a ~10x-undersized output file and either a `READ_VARIABLE` crash or near-uniform ~1% confidence on all real inputs. **The fix: convert directly via `TFLiteConverter.from_keras_model(model)` with zero manual tracing.** This produced a working 0.56 MB file, confirmed via tensor inspection and a real-data test (80% accuracy on a 20-sample spot-check). **Separately and importantly: the model already deployed on the physical Raspberry Pi was tested for the first time during this debugging process and found to have the identical defect — it has never performed real sign recognition.** The corrected file needs to replace it; this replacement was advised but not confirmed done.

**Files:** `extract_hands.py`, `build_dataset.py`, `preprocessing.py` (shared normalization, import don't duplicate), `train_model.py`, `loso_cross_validation.py`, `train_production_model.py` all complete and executed successfully on Colab T4. `convert_tflite_simple.py` is the **correct, working** local-Mac conversion script (uses `from_keras_model`, includes an H5 config-patch step for a Keras version schema mismatch around `quantization_config`, weight verification, and a real-data test) — do not use `convert_tflite.py` or `convert_tflite_local.py`, both superseded and broken. `test_webcam_live.py` provides live validation on a Mac webcam with Arabic text rendering (`arabic_reshaper` + `python-bidi` + PIL, since `cv2.putText` can't render Arabic at all) and a handedness-mirroring debug overlay.

**Open items:** (1) get the production model's real accuracy/loss numbers, (2) confirm the corrected `.tflite` actually works when physically deployed on the Pi, (3) run a full-test-set TFLite-vs-Keras accuracy comparison (only a 20-sample spot check exists), (4) a webcam number-display bug was patched defensively (try/except around text rendering, explicit `str()` casts, a `class_names` type-consistency startup check) but never fully root-caused — watch for a recurrence, (5) a thesis Section 6.5 Word document exists matching the student's chapter formatting, narrative-prose style per explicit feedback rather than table-heavy, with three figure placeholders still unfilled (dataset statistics, TFLite debugging terminal output, webcam screenshot) and the production accuracy gap noted in item 1 still open there too.

**Environment:** Google Colab T4 GPU for all training. Local Mac (M1, conda env `action_detection`, TensorFlow 2.19.1) for extraction, dataset building, and TFLite conversion — deliberately not used for training to avoid thermal load. Random seed 42 throughout.
