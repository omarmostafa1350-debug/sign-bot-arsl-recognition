"""
BTS7960 dual-motor differential drive controller (RPi.GPIO, software PWM).

Each BTS7960 module needs 4 control lines: RPWM, LPWM, R_EN, L_EN.
Speed/direction is set by driving ONE of RPWM/LPWM with a PWM duty cycle
(0-100) while the other stays at 0; both EN pins are held HIGH to enable
the driver. If RPi.GPIO isn't importable (e.g. running off-Pi for UI
development) this falls back to a simulation mode that just logs.
"""
import logging

logger = logging.getLogger(__name__)

try:
    import RPi.GPIO as GPIO
    GPIO_AVAILABLE = True
except (ImportError, RuntimeError):
    GPIO_AVAILABLE = False
    logger.warning("RPi.GPIO not available — motor controller running in SIMULATION mode")

import config


class _SingleMotor:
    """One BTS7960 channel (one side of the robot)."""

    def __init__(self, rpwm_pin, lpwm_pin, r_en_pin, l_en_pin, pwm_freq):
        self.rpwm_pin = rpwm_pin
        self.lpwm_pin = lpwm_pin
        self.r_en_pin = r_en_pin
        self.l_en_pin = l_en_pin
        self._rpwm = None
        self._lpwm = None

        if GPIO_AVAILABLE:
            GPIO.setup(rpwm_pin, GPIO.OUT)
            GPIO.setup(lpwm_pin, GPIO.OUT)
            GPIO.setup(r_en_pin, GPIO.OUT)
            GPIO.setup(l_en_pin, GPIO.OUT)
            GPIO.output(r_en_pin, GPIO.HIGH)
            GPIO.output(l_en_pin, GPIO.HIGH)
            self._rpwm = GPIO.PWM(rpwm_pin, pwm_freq)
            self._lpwm = GPIO.PWM(lpwm_pin, pwm_freq)
            self._rpwm.start(0)
            self._lpwm.start(0)

    def set_speed(self, speed):
        """speed: float in [-1.0, 1.0]. Positive = forward, negative = reverse."""
        speed = max(-1.0, min(1.0, speed))
        duty = abs(speed) * 100.0
        if not GPIO_AVAILABLE:
            return
        if speed >= 0:
            self._rpwm.ChangeDutyCycle(duty)
            self._lpwm.ChangeDutyCycle(0)
        else:
            self._rpwm.ChangeDutyCycle(0)
            self._lpwm.ChangeDutyCycle(duty)

    def stop(self):
        self.set_speed(0.0)

    def cleanup(self):
        if GPIO_AVAILABLE and self._rpwm:
            self._rpwm.stop()
            self._lpwm.stop()


class DriveController:
    """Differential-drive controller for 2 BTS7960 modules (left + right track)."""

    def __init__(self):
        if GPIO_AVAILABLE:
            GPIO.setmode(GPIO.BCM)
            GPIO.setwarnings(False)

        self.left = _SingleMotor(
            config.LEFT_MOTOR_RPWM, config.LEFT_MOTOR_LPWM,
            config.LEFT_MOTOR_R_EN, config.LEFT_MOTOR_L_EN,
            config.MOTOR_PWM_FREQ_HZ,
        )
        self.right = _SingleMotor(
            config.RIGHT_MOTOR_RPWM, config.RIGHT_MOTOR_LPWM,
            config.RIGHT_MOTOR_R_EN, config.RIGHT_MOTOR_L_EN,
            config.MOTOR_PWM_FREQ_HZ,
        )
        self._forward_blocked = False

    def set_forward_blocked(self, blocked):
        """Called by the ultrasonic safety logic. Does not affect reverse/turn."""
        self._forward_blocked = blocked

    def drive(self, x, y):
        """
        x, y: joystick axes in [-1.0, 1.0]. y > 0 = forward, x > 0 = right turn.
        Applies differential-drive mixing and the forward-obstacle interlock.
        Returns the (left_speed, right_speed) actually applied.
        """
        if abs(x) < config.JOYSTICK_DEADZONE:
            x = 0.0
        if abs(y) < config.JOYSTICK_DEADZONE:
            y = 0.0

        if y > 0 and self._forward_blocked:
            y = 0.0  # forward obstacle interlock — reverse/turn still allowed

        left_speed = max(-1.0, min(1.0, y + x))
        right_speed = max(-1.0, min(1.0, y - x))

        self.left.set_speed(left_speed)
        self.right.set_speed(right_speed)
        return left_speed, right_speed

    def stop(self):
        self.left.stop()
        self.right.stop()

    def cleanup(self):
        self.left.cleanup()
        self.right.cleanup()
