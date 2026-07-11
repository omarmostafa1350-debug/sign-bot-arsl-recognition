"""
Sign-Bot Dashboard — Central Configuration
===========================================
Edit the values in this file to match your wiring. Nothing else in the
project needs to change once these are correct (BCM numbering throughout).
"""

# ═════════════════════════════════════════════════════════════════════════
# Screen
# ═════════════════════════════════════════════════════════════════════════
SCREEN_WIDTH = 800
SCREEN_HEIGHT = 480
FULLSCREEN = True  # set False while developing on a laptop / over VNC

# ═════════════════════════════════════════════════════════════════════════
# Camera (USB webcam)
# ═════════════════════════════════════════════════════════════════════════
CAMERA_INDEX = 0
CAMERA_WIDTH = 640
CAMERA_HEIGHT = 480

# ═════════════════════════════════════════════════════════════════════════
# Sign Language Model (TFLite + MediaPipe Holistic) — same pipeline as
# Sign-bot V2 (omd.py): 21 hand points x 2 hands x (x,y,z) = 126 features,
# 20-frame sequence buffer, predicted every 3 frames.
# ═════════════════════════════════════════════════════════════════════════
MODEL_PATH = "/home/pi/Desktop/signbot_project/model/signbot_arabic_v2_fixed.tflite"
LABELS_PATH = "/home/pi/Desktop/signbot_project/output/dataset/class_names.npy"

NUM_HAND_POINTS = 21
MAX_SEQUENCE_LENGTH = 20
PREDICT_EVERY_N_FRAMES = 3
CONFIDENCE_THRESHOLD = 0.5
STALE_PREDICTION_TIMEOUT = 2.0       # seconds before "Listening..." reappears
STABLE_PREDICTION_REPEAT_GUARD = 3.0  # don't re-speak the same word inside this window

# Language the detected sign labels are written in ("ar" or "en") — used to
# pick the correct espeak-ng voice when speaking a detected word aloud.
SIGN_LABEL_LANGUAGE = "ar"

# ═════════════════════════════════════════════════════════════════════════
# Voice Recognition (Vosk, offline, bilingual)
# ═════════════════════════════════════════════════════════════════════════
VOSK_MODELS_DIR = "/home/pi/Desktop/Voice_rec"
VOSK_MODEL_AR = "vosk-model-ar-mgb2-0.4"
VOSK_MODEL_EN = "vosk-model-small-en-us-0.15"
STT_SAMPLE_RATE = 16000
STT_DEFAULT_LANGUAGE = "ar"  # "ar" or "en"

# ALSA capture device used by `arecord` for mic input.
# Find yours with `arecord -l`: "card 1: B100 [Brio 100], device 0" → plughw:1,0
STT_ALSA_DEVICE = "plughw:1,0"

# Cooldown — identical final result suppressed if heard again within this window
STT_REPEAT_COOLDOWN_S = 1.5

# Voice recognition backend: "whisper" | "google" | "google_cloud" | "vosk"
# "whisper"      — OpenAI Whisper API (best dialect accuracy, requires API key + internet)
# "google"       — Google free web speech (good, no key needed, requires internet)
# "google_cloud" — Google Cloud Speech-to-Text v1 REST API (paid, requires GOOGLE_CLOUD_API_KEY)
# "vosk"         — fully offline Kaldi models above (limited Arabic dialect accuracy)
STT_ENGINE = "google"

# OpenAI API key — only needed if STT_ENGINE = "whisper"
OPENAI_API_KEY = ""

# Google Cloud Speech-to-Text API key — only needed if STT_ENGINE = "google_cloud"
# Enable the "Cloud Speech-to-Text API" in Google Cloud Console, then create an
# API key under APIs & Services → Credentials and paste it here.
GOOGLE_CLOUD_API_KEY = os.environ.get("GOOGLE_CLOUD_API_KEY", "")

# How long a single captured utterance can be, and how long to wait for
# speech to start before looping (also the max delay before stop() takes effect)
STT_PHRASE_TIME_LIMIT_S = 12
STT_LISTEN_TIMEOUT_S    = 5

# ═════════════════════════════════════════════════════════════════════════
# Text-to-Speech (espeak-ng, bilingual EN/AR, plays through whatever is set
# as the default ALSA/PulseAudio output — pair + set your Bluetooth speaker
# as the default sink at the OS level first; this code does not handle
# Bluetooth pairing).
# ═════════════════════════════════════════════════════════════════════════
TTS_VOICE_EN = "en-us"
TTS_VOICE_AR = "ar"
TTS_SPEED_WPM = 160          # words per minute, espeak-ng -s
TTS_AMPLITUDE = 180          # 0-200, espeak-ng -a

# ═════════════════════════════════════════════════════════════════════════
# Drive Motors — BTS7960, 2 motors, differential drive (left / right track)
# Each BTS7960 module needs: RPWM, LPWM, R_EN, L_EN
# ═════════════════════════════════════════════════════════════════════════
MOTOR_PWM_FREQ_HZ = 1000

LEFT_MOTOR_RPWM = 12
LEFT_MOTOR_LPWM = 13
LEFT_MOTOR_R_EN = 5
LEFT_MOTOR_L_EN = 6

RIGHT_MOTOR_RPWM = 18
RIGHT_MOTOR_LPWM = 19
RIGHT_MOTOR_R_EN = 16
RIGHT_MOTOR_L_EN = 26

# Joystick deadzone (ignore tiny accidental touches near center)
JOYSTICK_DEADZONE = 0.08

# ═════════════════════════════════════════════════════════════════════════
# Ultrasonic obstacle sensor — ONE sensor, forward-facing only.
# Used to cut forward drive power; does not affect reverse or turning.
# ═════════════════════════════════════════════════════════════════════════
ULTRASONIC_TRIG_PIN = 23
ULTRASONIC_ECHO_PIN = 24
ULTRASONIC_WARNING_CM = 30   # show warning banner below this distance
ULTRASONIC_STOP_CM = 15      # hard-block forward motion below this distance
ULTRASONIC_POLL_HZ = 10

# ═════════════════════════════════════════════════════════════════════════
# Head Stepper Motor — TB6600 driver (STEP/DIR/ENA), used only for the
# welcome-screen greeting nod.
# ═════════════════════════════════════════════════════════════════════════
STEPPER_STEP_PIN = 17
STEPPER_DIR_PIN = 27
STEPPER_ENA_PIN = 22         # TB6600 ENA is usually active-LOW (enabled when LOW)
STEPPER_ENA_ACTIVE_LOW = True

STEPPER_STEPS_PER_REV = 200  # 1.8° motor, full step
STEPPER_MICROSTEP = 1        # set to match your TB6600 DIP switches (1/2/4/8/16)
STEPPER_STEP_DELAY_S = 0.0015  # delay between step pulses — lower = faster

# Welcome "greeting" motion: rotate this many degrees left, then right, then
# return to center.
WELCOME_NOD_DEGREES = 30

# ═════════════════════════════════════════════════════════════════════════
# Simulation mode — when True, all GPIO/camera/audio hardware calls are
# stubbed out so you can run and design the dashboard UI on a laptop before
# deploying to the Pi. Auto-detects if RPi.GPIO import fails.
# ═════════════════════════════════════════════════════════════════════════
FORCE_SIMULATION_MODE = False
