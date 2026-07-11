"""
HC-SR04 ultrasonic distance sensor — forward-facing only.
Runs a continuous polling thread (started once at app launch) and emits the
live distance reading; the Drive tab uses it to drive the obstacle gauge
and to block forward motion when something is too close.
"""
import logging
import time

from PyQt5.QtCore import QThread, pyqtSignal

logger = logging.getLogger(__name__)

try:
    import RPi.GPIO as GPIO
    GPIO_AVAILABLE = True
except (ImportError, RuntimeError):
    GPIO_AVAILABLE = False
    logger.warning("RPi.GPIO not available — ultrasonic running in SIMULATION mode")

import config

SOUND_SPEED_CM_PER_US = 0.0343
ECHO_TIMEOUT_S = 0.03  # ~5m max range worth of wait


class UltrasonicSensor(QThread):
    distanceReady = pyqtSignal(float)  # cm

    def __init__(self, parent=None):
        super().__init__(parent)
        self._running = False
        if GPIO_AVAILABLE:
            GPIO.setmode(GPIO.BCM)
            GPIO.setwarnings(False)
            GPIO.setup(config.ULTRASONIC_TRIG_PIN, GPIO.OUT)
            GPIO.setup(config.ULTRASONIC_ECHO_PIN, GPIO.IN)
            GPIO.output(config.ULTRASONIC_TRIG_PIN, GPIO.LOW)

    def _read_once(self):
        if not GPIO_AVAILABLE:
            return None
        GPIO.output(config.ULTRASONIC_TRIG_PIN, GPIO.HIGH)
        time.sleep(0.00001)
        GPIO.output(config.ULTRASONIC_TRIG_PIN, GPIO.LOW)

        start_wait = time.time()
        pulse_start = start_wait
        while GPIO.input(config.ULTRASONIC_ECHO_PIN) == GPIO.LOW:
            pulse_start = time.time()
            if pulse_start - start_wait > ECHO_TIMEOUT_S:
                return None

        pulse_end = pulse_start
        while GPIO.input(config.ULTRASONIC_ECHO_PIN) == GPIO.HIGH:
            pulse_end = time.time()
            if pulse_end - pulse_start > ECHO_TIMEOUT_S:
                return None

        elapsed_us = (pulse_end - pulse_start) * 1_000_000
        return (elapsed_us * SOUND_SPEED_CM_PER_US) / 2.0

    def run(self):
        self._running = True
        period = 1.0 / config.ULTRASONIC_POLL_HZ
        while self._running:
            loop_start = time.time()
            distance = self._read_once()
            if distance is not None:
                self.distanceReady.emit(distance)
            elapsed = time.time() - loop_start
            time.sleep(max(0.0, period - elapsed))

    def stop(self):
        self._running = False
        self.wait(1000)
