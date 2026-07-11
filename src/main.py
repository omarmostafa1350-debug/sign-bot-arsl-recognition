#!/usr/bin/env python3
"""
Sign-Bot Dashboard — entry point.

Run on the Pi:
    python3 main.py
"""
import logging
import sys

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (
    QApplication, QWidget, QStackedWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QPushButton
)

import config
from hardware.stepper_head import HeadStepper
from hardware.ultrasonic import UltrasonicSensor
from hardware.tts_engine import TTSEngine
from ui.i18n import Translator
from ui.styles import MAIN_QSS
from ui.welcome_screen import WelcomeScreen
from ui.sign_tab import SignTab
from ui.drive_tab import DriveTab
from ui.voice_tab import VoiceTab

logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


class StatusDot(QLabel):
    """Small persistent dot showing whether Sign/Voice is on, visible from any tab."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedSize(8, 8)
        self.set_active(False)

    def set_active(self, active):
        color = "#22d3a4" if active else "#3a4750"
        self.setStyleSheet(f"background-color:{color}; border-radius:4px;")


class MainWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Sign-Bot Dashboard")
        if config.FULLSCREEN:
            self.setWindowFlags(Qt.FramelessWindowHint)
        self.setFixedSize(config.SCREEN_WIDTH, config.SCREEN_HEIGHT)

        self.translator = Translator(lang="en")

        # Hardware singletons — created once, shared across tabs/screens
        self.head_stepper = HeadStepper()
        self.tts_engine = TTSEngine()
        self.ultrasonic = UltrasonicSensor()

        self._build_ui()

        self.ultrasonic.distanceReady.connect(self.drive_tab.on_ultrasonic_reading)
        self.ultrasonic.start()

    def _build_ui(self):
        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        self.screens = QStackedWidget()
        root.addWidget(self.screens)

        self.welcome_screen = WelcomeScreen(self.translator, self.head_stepper, self.tts_engine)
        self.welcome_screen.startClicked.connect(self._show_main_view)
        self.screens.addWidget(self.welcome_screen)

        self.main_view = QWidget()
        main_layout = QVBoxLayout(self.main_view)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)

        # ── Top status bar: Sign/Voice status dots + EN/AR language toggle ──
        top_bar = QHBoxLayout()
        top_bar.setContentsMargins(10, 6, 10, 6)

        self.sign_dot = StatusDot()
        self.sign_status_label = QLabel()
        self.voice_dot = StatusDot()
        self.voice_status_label = QLabel()

        self.lang_en_btn = QPushButton("EN")
        self.lang_ar_btn = QPushButton("AR")
        for b in (self.lang_en_btn, self.lang_ar_btn):
            b.setCheckable(True)
            b.setFixedSize(36, 24)
        self.lang_en_btn.setChecked(True)
        self.lang_en_btn.clicked.connect(lambda: self._set_language("en"))
        self.lang_ar_btn.clicked.connect(lambda: self._set_language("ar"))

        top_bar.addWidget(self.sign_dot)
        top_bar.addWidget(self.sign_status_label)
        top_bar.addSpacing(12)
        top_bar.addWidget(self.voice_dot)
        top_bar.addWidget(self.voice_status_label)
        top_bar.addStretch()
        top_bar.addWidget(self.lang_en_btn)
        top_bar.addWidget(self.lang_ar_btn)

        top_bar_widget = QWidget()
        top_bar_widget.setLayout(top_bar)
        top_bar_widget.setStyleSheet("border-bottom: 1px solid #1e2a30;")
        main_layout.addWidget(top_bar_widget)

        # ── Tab content ─────────────────────────────────────────────────
        self.tabs_stack = QStackedWidget()
        self.sign_tab = SignTab(self.translator)
        self.drive_tab = DriveTab(self.translator)
        self.voice_tab = VoiceTab(self.translator)

        self.sign_tab.switchedOn.connect(lambda: self.sign_dot.set_active(True))
        self.sign_tab.switchedOff.connect(lambda: self.sign_dot.set_active(False))
        self.sign_tab.speakRequested.connect(self.tts_engine.speak)

        self.voice_tab.switchedOn.connect(lambda: self.voice_dot.set_active(True))
        self.voice_tab.switchedOff.connect(lambda: self.voice_dot.set_active(False))
        self.voice_tab.speakRequested.connect(self.tts_engine.speak)

        self.tabs_stack.addWidget(self.sign_tab)
        self.tabs_stack.addWidget(self.drive_tab)
        self.tabs_stack.addWidget(self.voice_tab)
        main_layout.addWidget(self.tabs_stack, 1)

        # ── Bottom tab bar ──────────────────────────────────────────────
        bottom_bar = QHBoxLayout()
        bottom_bar.setContentsMargins(0, 0, 0, 0)
        bottom_bar.setSpacing(0)

        self.tab_buttons = []
        for index, key in enumerate(("tab_sign", "tab_drive", "tab_voice")):
            btn = QPushButton()
            btn.setObjectName("tabButton")
            btn.setCheckable(True)
            btn.setProperty("i18n_key", key)
            btn.clicked.connect(lambda _checked, i=index: self._switch_tab(i))
            bottom_bar.addWidget(btn)
            self.tab_buttons.append(btn)
        self.tab_buttons[0].setChecked(True)

        bottom_bar_widget = QWidget()
        bottom_bar_widget.setLayout(bottom_bar)
        bottom_bar_widget.setStyleSheet("border-top: 1px solid #1e2a30;")
        main_layout.addWidget(bottom_bar_widget)

        self.screens.addWidget(self.main_view)
        self.retranslate_all()

    def _switch_tab(self, index):
        self.tabs_stack.setCurrentIndex(index)
        for i, btn in enumerate(self.tab_buttons):
            btn.setChecked(i == index)

    def _show_main_view(self):
        self.screens.setCurrentWidget(self.main_view)

    def _set_language(self, lang):
        self.translator.set_language(lang)
        self.lang_en_btn.setChecked(lang == "en")
        self.lang_ar_btn.setChecked(lang == "ar")
        self.retranslate_all()

    def retranslate_all(self):
        t = self.translator.t
        self.sign_status_label.setText(t("tab_sign"))
        self.voice_status_label.setText(t("tab_voice"))
        for btn in self.tab_buttons:
            btn.setText(t(btn.property("i18n_key")))
        self.welcome_screen.retranslate()
        self.sign_tab.retranslate()
        self.drive_tab.retranslate()
        self.voice_tab.retranslate()

    def closeEvent(self, event):
        logger.info("Shutting down...")
        self.sign_tab.shutdown()
        self.voice_tab.shutdown()
        self.ultrasonic.stop()
        self.head_stepper.cleanup()
        self.tts_engine.shutdown()
        try:
            import RPi.GPIO as GPIO
            GPIO.cleanup()
        except ImportError:
            pass
        event.accept()


def main():
    app = QApplication(sys.argv)
    app.setStyleSheet(MAIN_QSS)

    window = MainWindow()
    if config.FULLSCREEN:
        window.showFullScreen()
        app.setOverrideCursor(Qt.BlankCursor)
    else:
        window.show()

    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
