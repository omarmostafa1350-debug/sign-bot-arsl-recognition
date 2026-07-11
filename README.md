# Sign-Bot: Arabic Sign Language Recognition

Real-time Arabic Sign Language (ArSL) recognition system built on MediaPipe hand-landmark tracking, a Bi-LSTM classifier, and TensorFlow Lite, deployed on a Raspberry Pi 4. Graduation project, Mechatronics Engineering, AAST Cairo (2026).

## How It Works

1. MediaPipe Hands extracts 21 landmarks per hand (126 values per frame across both hands) from the camera feed
2. Landmarks are normalized (wrist-relative, scale-invariant) and buffered into fixed-length sequences
3. A Bi-LSTM classifier predicts the sign class from the sequence
4. The trained model is converted to TensorFlow Lite (0.56 MB) for on-device inference
5. Runs in real time on a Raspberry Pi 4, with a PyQt5 GUI, motorized head/base control, and text-to-speech output

## Model

| | |
|---|---|
| Architecture | Masking → BiLSTM(128) → Dropout(0.3) → BiLSTM(64) → Dropout(0.3) → Dense(64) → Dense(softmax) |
| Classes | 100 signs (KArSL-502 subset: numbers, Arabic alphabet, medical terms) |
| Training signers | 3 |
| Model size | 0.56 MB (TFLite) |

## Results

| Evaluation | Accuracy |
|---|---|
| Single-pool test (held-out samples, same 3 signers as training) | 96.29% |
| Production model (deployed, trained on full 3-signer pool) | 96.47% |
| Leave-One-Signer-Out cross-validation (mean across 3 folds) | 36.11% ± 2.70% |

Full evaluation details, confusion matrices, and per-fold reports are in [`docs/results/`](docs/results/).

## Known Limitation

The gap between same-pool accuracy (96.29%) and cross-signer generalization (36.11%) is the main limitation of the current system. With only three signers in the training set, each LOSO fold trains on just two identities, which gives the model limited pressure to learn signer-invariant representations rather than signer-correlated ones. Reported accuracy applies to the three signers used in this study. Generalization to a new signer has not been established. Expanding the training population to 5-6 signers is the primary direction for future work.

## Repository Structure
## Setup

```bash
git clone https://github.com/omarmostafa1350-debug/sign-bot-arsl-recognition.git
cd sign-bot-arsl-recognition
pip install -r requirements.txt
python src/main.py
```

Requires a camera-equipped Raspberry Pi 4 (or any machine with a webcam for local testing). Speech-to-text can run fully offline (Vosk) or via Google Cloud Speech-to-Text (API key required, see `src/config.py`).

## Hardware

Runs on a mobile robot base with:
- Stepper-driven head for tracking
- Motor-driven base
- Ultrasonic obstacle sensing
- Text-to-speech output (espeak-ng, bilingual EN/AR)

## Documentation

- [Architecture guide](docs/architecture/architecture_guide.md)
- [LOSO evaluation summary](docs/LOSO_Evaluation_Summary.md)
- [Full handoff notes](docs/SIGNBOT_HANDOFF.md)

## Author

Omar Mostafa, Mechatronics Engineering, AAST Cairo
