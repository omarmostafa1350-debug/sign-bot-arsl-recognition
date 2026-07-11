"""
Sign-detection tab: camera feed, detected word, confidence, and a switch
(not a push button) that starts/stops the SignEngine — flipping it on spins
up the camera + MediaPipe + TFLite pipeline, flipping it off tears it down.
"""
from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtGui import QPixmap
from PyQt5.QtWidgets import (
    QWidget, QHBoxLayout, QVBoxLayout, QLabel, QPushButton, QFrame, QProgressBar
)

import config
from ui.toggle_switch import ToggleSwitch
from vision.sign_engine import SignEngine


class SignTab(QWidget):
    switchedOn = pyqtSignal()
    switchedOff = pyqtSignal()
    speakRequested = pyqtSignal(str, str)  # text, lang

    def __init__(self, translator, parent=None):
        super().__init__(parent)
        self.translator = translator
        self.engine = None
        self._last_word = ""

        root = QHBoxLayout(self)
        root.setContentsMargins(12, 12, 12, 12)
        root.setSpacing(10)

        self.camera_frame = QFrame()
        self.camera_frame.setObjectName("panel")
        cam_layout = QVBoxLayout(self.camera_frame)
        self.camera_label = QLabel()
        self.camera_label.setAlignment(Qt.AlignCenter)
        self.camera_label.setMinimumSize(300, 280)
        self.camera_status_label = QLabel()
        self.camera_status_label.setObjectName("sectionLabel")
        self.camera_status_label.setAlignment(Qt.AlignCenter)
        cam_layout.addWidget(self.camera_label, 1)
        cam_layout.addWidget(self.camera_status_label)

        right = QVBoxLayout()
        right.setSpacing(8)

        self.pred_frame = QFrame()
        self.pred_frame.setObjectName("panel")
        pred_layout = QVBoxLayout(self.pred_frame)
        pred_layout.setAlignment(Qt.AlignCenter)

        self.detected_label = QLabel()
        self.detected_label.setObjectName("sectionLabel")
        self.detected_label.setAlignment(Qt.AlignCenter)

        self.word_label = QLabel("...")
        self.word_label.setAlignment(Qt.AlignCenter)
        self.word_label.setStyleSheet("font-size: 26px; font-weight: bold; color: #22d3a4;")
        self.word_label.setWordWrap(True)

        self.confidence_bar = QProgressBar()
        self.confidence_bar.setRange(0, 100)
        self.confidence_bar.setTextVisible(False)
        self.confidence_bar.setFixedHeight(6)

        conf_fps_row = QHBoxLayout()
        self.conf_label = QLabel()
        self.conf_label.setObjectName("sectionLabel")
        self.fps_label = QLabel()
        self.fps_label.setObjectName("sectionLabel")
        conf_fps_row.addWidget(self.conf_label)
        conf_fps_row.addStretch()
        conf_fps_row.addWidget(self.fps_label)

        pred_layout.addWidget(self.detected_label)
        pred_layout.addWidget(self.word_label)
        pred_layout.addWidget(self.confidence_bar)
        pred_layout.addLayout(conf_fps_row)

        switch_row = QHBoxLayout()
        self.switch_label = QLabel()
        self.switch_label.setObjectName("sectionLabel")
        self.switch = ToggleSwitch()
        self.switch.toggledOn.connect(self._on_switch_toggled)
        switch_row.addWidget(self.switch_label)
        switch_row.addStretch()
        switch_row.addWidget(self.switch)

        self.speak_button = QPushButton()
        self.speak_button.clicked.connect(self._on_speak_clicked)

        right.addWidget(self.pred_frame, 1)
        right.addLayout(switch_row)
        right.addWidget(self.speak_button)

        right_container = QWidget()
        right_container.setLayout(right)
        right_container.setFixedWidth(260)

        root.addWidget(self.camera_frame, 1)
        root.addWidget(right_container)

        self.retranslate()

    def retranslate(self):
        t = self.translator.t
        self.camera_status_label.setText(t("camera_feed"))
        self.detected_label.setText(t("detected_sign"))
        self.switch_label.setText(t("sign_switch"))
        self.speak_button.setText(t("speak"))
        if not self.engine or not self.engine.isRunning():
            self.word_label.setText(t("listening"))
        self.conf_label.setText(f"{t('confidence')} —")
        self.fps_label.setText("—")

    def _on_switch_toggled(self, checked):
        if checked:
            self._start_engine()
        else:
            self._stop_engine()

    def _start_engine(self):
        self.word_label.setText(self.translator.t("starting"))
        self.engine = SignEngine()
        self.engine.frameReady.connect(self._on_frame)
        self.engine.predictionReady.connect(self._on_prediction)
        self.engine.stableWordDetected.connect(self._on_stable_word)
        self.engine.errorOccurred.connect(self._on_error)
        self.engine.start()
        self.switchedOn.emit()

    def _stop_engine(self):
        if self.engine:
            self.engine.stop()
            self.engine = None
        self.camera_label.clear()
        self.word_label.setText(self.translator.t("listening"))
        self.confidence_bar.setValue(0)
        self.switchedOff.emit()

    def _on_frame(self, qimg):
        pixmap = QPixmap.fromImage(qimg).scaled(
            self.camera_label.width(), self.camera_label.height(),
            Qt.KeepAspectRatio, Qt.SmoothTransformation,
        )
        self.camera_label.setPixmap(pixmap)

    def _on_prediction(self, word, confidence):
        t = self.translator.t
        if word:
            self.word_label.setText(word)
            self._last_word = word
        else:
            self.word_label.setText(t("listening"))
        self.confidence_bar.setValue(int(confidence * 100))
        self.conf_label.setText(f"{t('confidence')} {int(confidence * 100)}%")

    def _on_stable_word(self, word):
        self.speakRequested.emit(word, config.SIGN_LABEL_LANGUAGE)

    def _on_speak_clicked(self):
        if self._last_word:
            self.speakRequested.emit(self._last_word, config.SIGN_LABEL_LANGUAGE)

    def _on_error(self, message):
        self.word_label.setText(f"⚠ {message}")
        self.switch.setChecked(False)

    def is_active(self):
        return self.engine is not None and self.engine.isRunning()

    def shutdown(self):
        if self.engine:
            self.engine.stop()
