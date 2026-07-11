"""
Bilingual (EN/AR) text-to-speech via espeak-ng, run on a background thread
so speaking never blocks the GUI. Output goes to whatever is set as the
default ALSA/PulseAudio sink — pair your Bluetooth speaker and set it as
the default audio output at the OS level first; this module does not
manage Bluetooth pairing itself.

Install on the Pi:
    sudo apt install espeak-ng
"""
import logging
import queue
import subprocess
import threading

logger = logging.getLogger(__name__)

import config

_VOICES = {"en": config.TTS_VOICE_EN, "ar": config.TTS_VOICE_AR}


class TTSEngine:
    def __init__(self):
        self._queue = queue.Queue()
        self._running = True
        self._thread = threading.Thread(target=self._worker, daemon=True, name="TTSWorker")
        self._thread.start()

    def speak(self, text, lang="en"):
        """Non-blocking — enqueues text to be spoken. Safe to call from any thread."""
        if not text:
            return
        self._queue.put((text, lang))

    def _worker(self):
        while self._running:
            try:
                text, lang = self._queue.get(timeout=0.5)
            except queue.Empty:
                continue
            voice = _VOICES.get(lang, config.TTS_VOICE_EN)
            try:
                subprocess.run(
                    [
                        "espeak-ng", "-v", voice,
                        "-s", str(config.TTS_SPEED_WPM),
                        "-a", str(config.TTS_AMPLITUDE),
                        text,
                    ],
                    check=False,
                    timeout=15,
                )
            except FileNotFoundError:
                logger.error("espeak-ng not found — install with: sudo apt install espeak-ng")
            except Exception as e:
                logger.error("TTS playback error: %s", e)

    def shutdown(self):
        self._running = False
