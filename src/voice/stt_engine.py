"""
Bilingual (AR/EN) speech-to-text. QThread that can be started/stopped by
the Voice switch and supports live language hot-swap.

Four selectable backends (config.STT_ENGINE):

  "whisper" — OpenAI Whisper API.
    The whisper-1 model has excellent multilingual accuracy, handles Arabic
    dialects (including Egyptian colloquial) natively without needing locale
    codes, and is trained on a huge variety of real-world speech rather than
    narrow broadcast corpora. Requires OPENAI_API_KEY and internet access.
    Each complete phrase is saved to a temp WAV and sent to the API;
    response is typically <1s on a fast connection.

  "google" — Google's free web speech API (SpeechRecognition).
    No API key needed. Good accuracy, handles ar-EG well.
    Requires internet access.

  "google_cloud" — Google Cloud Speech-to-Text v1 REST API (paid, with key).
    Superior accuracy over the free Google backend, especially for Arabic
    dialects. Requires GOOGLE_CLOUD_API_KEY in config.py and internet access.
    Each utterance is sent as a base64-encoded FLAC payload; response is
    typically <1s. No extra package needed beyond `requests`.

  "vosk" — Fully offline Kaldi models.
    No internet required, but Arabic accuracy on dialectal/colloquial
    speech is poor (models are broadcast-only). Useful as a fallback.

Audio capture in all modes: `arecord` subprocess via the proven
plughw ALSA path (far more reliable than PyAudio's raw device handling).
"""
import base64
import json
import logging
import os
import subprocess
import tempfile
import time

from PyQt5.QtCore import QThread, pyqtSignal

import config

logger = logging.getLogger(__name__)

# ── Dependency availability flags ──────────────────────────────────────────
try:
    import openai as _openai_mod
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False
    logger.warning("openai package not installed — Whisper backend unavailable; pip install openai")

try:
    import speech_recognition as sr
    SR_AVAILABLE = True
except ImportError:
    SR_AVAILABLE = False
    logger.warning("SpeechRecognition not installed — Google/Whisper backends may be limited")

try:
    from vosk import Model as VoskModel, KaldiRecognizer
    VOSK_AVAILABLE = True
except ImportError:
    VOSK_AVAILABLE = False
    logger.warning("vosk not installed — offline backend unavailable")

try:
    import requests as _requests_mod
    REQUESTS_AVAILABLE = True
except ImportError:
    REQUESTS_AVAILABLE = False
    logger.warning("requests package not installed — Google Cloud STT backend unavailable; pip install requests")


# ── arecord-backed AudioSource for SpeechRecognition ──────────────────────
class _ArecordStream:
    def __init__(self, pipe, sample_width):
        self.pipe = pipe
        self.sample_width = sample_width

    def read(self, frames):
        return self.pipe.read(frames * self.sample_width)


# ArecordAudioSource must inherit from sr.AudioSource so SpeechRecognition's
# isinstance() check passes — it guards every call to listen() and
# adjust_for_ambient_noise(). Use as a context manager (with statement).
if SR_AVAILABLE:
    _SourceBase = sr.AudioSource
else:
    _SourceBase = object


class ArecordAudioSource(_SourceBase):
    """SpeechRecognition AudioSource backed by `arecord` instead of PyAudio."""

    def __init__(self, alsa_device, sample_rate, chunk_frames=1024):
        self.alsa_device = alsa_device
        self.SAMPLE_RATE = sample_rate
        self.SAMPLE_WIDTH = 2          # 16-bit → 2 bytes
        self.CHUNK = chunk_frames
        self.stream = None
        self._proc = None

    def __enter__(self):
        cmd = [
            "arecord", "-D", self.alsa_device,
            "-f", "S16_LE", "-r", str(self.SAMPLE_RATE),
            "-c", "1", "-t", "raw",
        ]
        self._proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
        self.stream = _ArecordStream(self._proc.stdout, self.SAMPLE_WIDTH)
        return self

    # Allow both `with source:` and explicit `source.open()` call patterns
    def open(self):
        return self.__enter__()

    def __exit__(self, exc_type, exc_val, exc_tb):
        self._close()

    def _close(self):
        if self._proc and self._proc.poll() is None:
            self._proc.terminate()
            try:
                self._proc.wait(timeout=2)
            except subprocess.TimeoutExpired:
                self._proc.kill()
        self._proc = None
        self.stream = None


def _find_alsa_device(preferred: str) -> str:
    """Return the preferred ALSA device if arecord can open it, else auto-detect the
    first capture card reported by `arecord -l` and return `plughw:<card>,0`.
    Falls back to 'default' if nothing is found."""
    # First: try a zero-duration test record on the preferred device
    try:
        result = subprocess.run(
            ["arecord", "-D", preferred, "-d", "0", "-f", "S16_LE",
             "-r", "16000", "-c", "1", "/dev/null"],
            stderr=subprocess.PIPE, timeout=3,
        )
        if result.returncode == 0 or b"Broken pipe" in result.stderr:
            return preferred          # preferred device works fine
    except Exception:
        pass

    # Second: scan arecord -l for the first capture card
    try:
        out = subprocess.check_output(["arecord", "-l"], stderr=subprocess.DEVNULL,
                                      timeout=5).decode(errors="replace")
        for line in out.splitlines():
            if line.startswith("card "):
                # "card 1: B100 [Brio 100], device 0: ..."
                parts = line.split(":")
                card_num = parts[0].split()[1]
                logger.info("Auto-detected ALSA capture device: card %s → plughw:%s,0", card_num, card_num)
                return f"plughw:{card_num},0"
    except Exception:
        pass

    logger.warning("Could not auto-detect ALSA device, using 'default'")
    return "default"


# ── Engine thread ──────────────────────────────────────────────────────────
class STTEngine(QThread):
    textReady    = pyqtSignal(str, str, bool)  # text, lang, is_partial
    errorOccurred = pyqtSignal(str)
    modelLoading = pyqtSignal(str)             # lang — mic warming up
    modelReady   = pyqtSignal(str)             # lang — listening
    statusChanged = pyqtSignal(str)            # "listening" | "processing" | "idle"

    def __init__(self, parent=None):
        super().__init__(parent)
        self._running = False
        self.language = config.STT_DEFAULT_LANGUAGE
        self._restart_requested = False
        self._source = None
        self._capture_proc = None        # Vosk backend proc handle for stop()
        self._last_final_text = ""
        self._last_final_ts   = 0.0
        # Resolve the ALSA device once at startup — auto-detects if the
        # configured device isn't accessible (e.g. different card index on
        # a fresh install or different USB port)
        self._alsa_device = _find_alsa_device(config.STT_ALSA_DEVICE)
        logger.info("STTEngine using ALSA device: %s", self._alsa_device)

    def set_language(self, lang):
        if lang == self.language:
            return
        self.language = lang
        self._restart_requested = True

    def stop(self):
        self._running = False
        if self._source:
            self._source._close()
        if self._capture_proc and self._capture_proc.poll() is None:
            self._capture_proc.terminate()
            try:
                self._capture_proc.wait(timeout=2)
            except subprocess.TimeoutExpired:
                self._capture_proc.kill()
        self.wait(3000)

    def _emit_if_new(self, text, language):
        if not text:
            return
        now = time.time()
        repeat = (
            text == self._last_final_text
            and now - self._last_final_ts < config.STT_REPEAT_COOLDOWN_S
        )
        if not repeat:
            self.textReady.emit(text, language, False)
            self._last_final_text = text
            self._last_final_ts   = now

    def run(self):
        engine = config.STT_ENGINE
        if engine == "whisper":
            self._run_whisper()
        elif engine == "google":
            self._run_google()
        elif engine == "google_cloud":
            self._run_google_cloud()
        else:
            self._run_vosk()

    # ── Whisper backend ────────────────────────────────────────────────────
    def _run_whisper(self):
        if not OPENAI_AVAILABLE:
            self.errorOccurred.emit("openai package not installed — run: pip install openai")
            return
        if not SR_AVAILABLE:
            self.errorOccurred.emit("SpeechRecognition not installed — run: pip install SpeechRecognition")
            return
        if not config.OPENAI_API_KEY:
            self.errorOccurred.emit("OPENAI_API_KEY not set in config.py")
            return

        client = _openai_mod.OpenAI(api_key=config.OPENAI_API_KEY)
        self._running = True

        while self._running:
            self._restart_requested = False
            language = self.language
            # Whisper uses ISO 639-1 language codes — "ar" works for all
            # Arabic dialects including Egyptian; "en" for English.
            whisper_lang = "ar" if language == "ar" else "en"

            self.modelLoading.emit(language)
            source = ArecordAudioSource(self._alsa_device, config.STT_SAMPLE_RATE)
            try:
                # Use __enter__ (not a missing .open()) to start arecord
                source.__enter__()
                self._source = source
                recognizer = sr.Recognizer()
                # Calibrate for ambient noise (1 second)
                recognizer.adjust_for_ambient_noise(source, duration=1)
            except Exception as e:
                logger.error("Whisper: could not start microphone: %s", e)
                self.errorOccurred.emit(f"Could not start microphone: {e}")
                source._close()
                self._source = None
                return

            self.modelReady.emit(language)

            while self._running and not self._restart_requested:
                self.statusChanged.emit("listening")
                try:
                    audio = recognizer.listen(
                        source,
                        timeout=config.STT_LISTEN_TIMEOUT_S,
                        phrase_time_limit=config.STT_PHRASE_TIME_LIMIT_S,
                    )
                except sr.WaitTimeoutError:
                    continue
                except Exception:
                    break   # source closed by stop()

                self.statusChanged.emit("processing")
                text = ""
                try:
                    wav_bytes = audio.get_wav_data()
                    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                        tmp.write(wav_bytes)
                        tmp_path = tmp.name
                    try:
                        with open(tmp_path, "rb") as f:
                            transcript = client.audio.transcriptions.create(
                                model="whisper-1",
                                file=f,
                                language=whisper_lang,
                            )
                        text = transcript.text.strip()
                    finally:
                        try:
                            os.unlink(tmp_path)
                        except OSError:
                            pass
                except _openai_mod.APIConnectionError as e:
                    logger.error("Whisper API connection error: %s", e)
                except _openai_mod.AuthenticationError:
                    self.errorOccurred.emit("Invalid OpenAI API key — check OPENAI_API_KEY in config.py")
                    break
                except Exception as e:
                    logger.error("Whisper API error: %s", e)

                self.statusChanged.emit("idle")
                self._emit_if_new(text, language)

            source._close()
            self._source = None

    def _run_google(self):
        if not SR_AVAILABLE:
            msg = "SpeechRecognition not installed — run: pip install SpeechRecognition"
            logger.error(msg)
            self.errorOccurred.emit(msg)
            return

        GOOGLE_LANG = {"ar": "ar-EG", "en": "en-US"}
        self._running = True

        while self._running:
            self._restart_requested = False
            language = self.language
            google_lang = GOOGLE_LANG.get(language, "en-US")

            self.modelLoading.emit(language)
            source = ArecordAudioSource(self._alsa_device, config.STT_SAMPLE_RATE)
            try:
                source.__enter__()
                self._source = source
                recognizer = sr.Recognizer()
                recognizer.adjust_for_ambient_noise(source, duration=1)
            except Exception as e:
                msg = f"Could not start microphone: {e}"
                logger.error(msg)
                self.errorOccurred.emit(msg)
                source._close()
                self._source = None
                return

            self.modelReady.emit(language)

            while self._running and not self._restart_requested:
                self.statusChanged.emit("listening")
                try:
                    audio = recognizer.listen(
                        source,
                        timeout=config.STT_LISTEN_TIMEOUT_S,
                        phrase_time_limit=config.STT_PHRASE_TIME_LIMIT_S,
                    )
                    logger.debug("Google: audio captured, %d bytes WAV",
                                 len(audio.get_wav_data()))
                except sr.WaitTimeoutError:
                    logger.debug("Google: listen timeout (no speech), looping")
                    continue
                except Exception as e:
                    logger.error("Google: listen() raised %s: %s", type(e).__name__, e)
                    break

                self.statusChanged.emit("processing")
                try:
                    text = recognizer.recognize_google(audio, language=google_lang)
                    logger.info("Google STT recognised [%s]: %r", google_lang, text)
                except sr.UnknownValueError:
                    logger.debug("Google: UnknownValueError — audio captured but speech not recognised")
                    text = ""
                except sr.RequestError as e:
                    msg = f"Google STT request failed: {e}"
                    logger.error(msg)
                    self.errorOccurred.emit(msg)
                    text = ""
                self.statusChanged.emit("idle")
                self._emit_if_new(text, language)

            source._close()
            self._source = None

    # ── Google Cloud Speech-to-Text v1 REST backend ───────────────────────
    def _run_google_cloud(self):
        """Google Cloud Speech-to-Text v1 via REST API (requires API key).

        Sends each captured utterance as a base64-encoded LINEAR16 WAV to
        https://speech.googleapis.com/v1/speech:recognize?key=<API_KEY>
        No additional Python packages are needed beyond `requests`.
        """
        if not SR_AVAILABLE:
            msg = "SpeechRecognition not installed — run: pip install SpeechRecognition"
            logger.error(msg)
            self.errorOccurred.emit(msg)
            return
        if not REQUESTS_AVAILABLE:
            msg = "requests not installed — run: pip install requests"
            logger.error(msg)
            self.errorOccurred.emit(msg)
            return
        if not config.GOOGLE_CLOUD_API_KEY:
            self.errorOccurred.emit("GOOGLE_CLOUD_API_KEY not set in config.py")
            return

        # BCP-47 language codes for Google Cloud Speech-to-Text
        GCLOUD_LANG = {"ar": "ar-EG", "en": "en-US"}
        GCLOUD_ENDPOINT = "https://speech.googleapis.com/v1/speech:recognize"

        self._running = True

        while self._running:
            self._restart_requested = False
            language = self.language
            bcp47_lang = GCLOUD_LANG.get(language, "en-US")

            self.modelLoading.emit(language)
            source = ArecordAudioSource(self._alsa_device, config.STT_SAMPLE_RATE)
            try:
                source.__enter__()
                self._source = source
                recognizer = sr.Recognizer()
                recognizer.adjust_for_ambient_noise(source, duration=1)
            except Exception as e:
                msg = f"Could not start microphone: {e}"
                logger.error(msg)
                self.errorOccurred.emit(msg)
                source._close()
                self._source = None
                return

            self.modelReady.emit(language)

            while self._running and not self._restart_requested:
                self.statusChanged.emit("listening")
                try:
                    audio = recognizer.listen(
                        source,
                        timeout=config.STT_LISTEN_TIMEOUT_S,
                        phrase_time_limit=config.STT_PHRASE_TIME_LIMIT_S,
                    )
                except sr.WaitTimeoutError:
                    continue
                except Exception as e:
                    logger.error("Listen error: %s", e)
                    break

                self.statusChanged.emit("processing")
                text = ""
                try:
                    # Get raw WAV bytes and encode to base64 for the REST API
                    wav_bytes = audio.get_wav_data()
                    audio_b64 = base64.b64encode(wav_bytes).decode("utf-8")

                    payload = {
                        "config": {
                            "encoding": "LINEAR16",
                            "sampleRateHertz": config.STT_SAMPLE_RATE,
                            "languageCode": bcp47_lang,
                            "enableAutomaticPunctuation": True,
                            "model": "latest_long",  # best for conversational Arabic
                        },
                        "audio": {
                            "content": audio_b64,
                        },
                    }

                    resp = _requests_mod.post(
                        GCLOUD_ENDPOINT,
                        params={"key": config.GOOGLE_CLOUD_API_KEY},
                        json=payload,
                        timeout=10,
                    )

                    if resp.status_code == 200:
                        data = resp.json()
                        results = data.get("results", [])
                        if results:
                            # Pick the top alternative from the first result
                            text = results[0]["alternatives"][0].get("transcript", "").strip()
                        else:
                            logger.debug("Google Cloud STT: empty results (silence or no match)")
                    elif resp.status_code in (400, 401, 403):
                        # Fatal: bad/unsupported credentials — stop the engine.
                        # NOTE: Google Cloud Speech-to-Text v1 does NOT support plain
                        # API keys; it requires a Service Account JSON (OAuth2).
                        # Use STT_ENGINE = "google" for keyless operation instead.
                        err_body = resp.text[:400]
                        self.errorOccurred.emit(
                            f"Google Cloud STT auth error (HTTP {resp.status_code}). "
                            "API keys are not supported — a Service Account JSON is required. "
                            "Switch STT_ENGINE to 'google' in config.py for keyless operation."
                        )
                        logger.error("Google Cloud STT %s: %s", resp.status_code, resp.text)
                        self._running = False
                        break
                    else:
                        logger.error(
                            "Google Cloud STT HTTP %s: %s", resp.status_code, resp.text[:300]
                        )

                except _requests_mod.ConnectionError as e:
                    logger.error("Google Cloud STT connection error: %s", e)
                except _requests_mod.Timeout:
                    logger.error("Google Cloud STT request timed out")
                except Exception as e:
                    logger.error("Google Cloud STT unexpected error: %s", e)

                self.statusChanged.emit("idle")
                self._emit_if_new(text, language)

            source._close()
            self._source = None

    # ── Vosk offline backend ───────────────────────────────────────────────
    def _run_vosk(self):
        if not VOSK_AVAILABLE:
            self.errorOccurred.emit("vosk not installed")
            return

        self._running = True

        while self._running:
            self._restart_requested = False
            language = self.language
            model_name = config.VOSK_MODEL_AR if language == "ar" else config.VOSK_MODEL_EN
            model_path = os.path.join(config.VOSK_MODELS_DIR, model_name)

            if not os.path.isdir(model_path):
                self.errorOccurred.emit(f"Vosk model not found: {model_path}")
                return

            self.modelLoading.emit(language)
            try:
                vosk_model = VoskModel(model_path)
                recognizer = KaldiRecognizer(vosk_model, config.STT_SAMPLE_RATE)
            except Exception as e:
                self.errorOccurred.emit(f"Vosk model load failed: {e}")
                return

            cmd = [
                "arecord", "-D", self._alsa_device,
                "-f", "S16_LE", "-r", str(config.STT_SAMPLE_RATE),
                "-c", "1", "-t", "raw",
            ]
            try:
                self._capture_proc = subprocess.Popen(
                    cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL
                )
            except Exception as e:
                self.errorOccurred.emit(f"Could not start arecord: {e}")
                return

            self.modelReady.emit(language)
            chunk = 8000   # ~0.25 s of 16-bit mono 16 kHz PCM

            while self._running and not self._restart_requested:
                data = self._capture_proc.stdout.read(chunk)
                if not data:
                    break
                if recognizer.AcceptWaveform(data):
                    result = json.loads(recognizer.Result())
                    self._emit_if_new(result.get("text", "").strip(), language)
                else:
                    partial = json.loads(recognizer.PartialResult())
                    pt = partial.get("partial", "").strip()
                    if pt:
                        self.textReady.emit(pt, language, True)

            if self._capture_proc:
                self._capture_proc.terminate()
                try:
                    self._capture_proc.wait(timeout=2)
                except subprocess.TimeoutExpired:
                    self._capture_proc.kill()
                self._capture_proc = None
