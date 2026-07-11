"""
TB6600 STEP/DIR stepper driver — drives the head-rotation motor.
Runs step pulses on a background thread so the GUI never blocks while the
head is moving (used both for the welcome-screen greeting nod and for the
manual jog buttons on the Drive tab).
"""
import logging
import threading
import time

logger = logging.getLogger(__name__)

try:
    import RPi.GPIO as GPIO
    GPIO_AVAILABLE = True
except (ImportError, RuntimeError):
    GPIO_AVAILABLE = False
    logger.warning("RPi.GPIO not available — stepper running in SIMULATION mode")

import config


class HeadStepper:
    def __init__(self):
        self._lock = threading.Lock()
        self._busy = False
        self.current_angle = 0  # signed degrees, 0 = center

        if GPIO_AVAILABLE:
            GPIO.setmode(GPIO.BCM)
            GPIO.setwarnings(False)
            GPIO.setup(config.STEPPER_STEP_PIN, GPIO.OUT)
            GPIO.setup(config.STEPPER_DIR_PIN, GPIO.OUT)
            GPIO.setup(config.STEPPER_ENA_PIN, GPIO.OUT)
        self._set_enabled(True)

    def _set_enabled(self, enabled):
        if not GPIO_AVAILABLE:
            return
        on_level = GPIO.LOW if config.STEPPER_ENA_ACTIVE_LOW else GPIO.HIGH
        off_level = GPIO.HIGH if config.STEPPER_ENA_ACTIVE_LOW else GPIO.LOW
        GPIO.output(config.STEPPER_ENA_PIN, on_level if enabled else off_level)

    def _steps_for_degrees(self, degrees):
        steps_per_rev = config.STEPPER_STEPS_PER_REV * config.STEPPER_MICROSTEP
        return int(round(abs(degrees) / 360.0 * steps_per_rev))

    def _pulse(self, n_steps, clockwise):
        if not GPIO_AVAILABLE:
            time.sleep(min(n_steps * config.STEPPER_STEP_DELAY_S * 2, 0.5))
            return
        GPIO.output(config.STEPPER_DIR_PIN, GPIO.HIGH if clockwise else GPIO.LOW)
        for _ in range(n_steps):
            GPIO.output(config.STEPPER_STEP_PIN, GPIO.HIGH)
            time.sleep(config.STEPPER_STEP_DELAY_S)
            GPIO.output(config.STEPPER_STEP_PIN, GPIO.LOW)
            time.sleep(config.STEPPER_STEP_DELAY_S)

    def rotate_to(self, target_angle, on_done=None):
        """Rotate from current_angle to target_angle (signed degrees, 0 = center)."""
        def worker():
            with self._lock:
                self._busy = True
                delta = target_angle - self.current_angle
                n_steps = self._steps_for_degrees(delta)
                if n_steps > 0:
                    self._pulse(n_steps, clockwise=delta > 0)
                self.current_angle = target_angle
                self._busy = False
            if on_done:
                on_done()

        threading.Thread(target=worker, daemon=True, name="HeadStepperWorker").start()

    def greeting_nod(self, on_done=None):
        """Welcome-screen motion: center -> +N deg -> -N deg -> center."""
        def worker():
            with self._lock:
                self._busy = True
                deg = config.WELCOME_NOD_DEGREES
                for target in (deg, -deg, 0):
                    delta = target - self.current_angle
                    n_steps = self._steps_for_degrees(delta)
                    if n_steps > 0:
                        self._pulse(n_steps, clockwise=delta > 0)
                    self.current_angle = target
                self._busy = False
            if on_done:
                on_done()

        threading.Thread(target=worker, daemon=True, name="HeadStepperGreeting").start()

    def is_busy(self):
        return self._busy

    def cleanup(self):
        self._set_enabled(False)
