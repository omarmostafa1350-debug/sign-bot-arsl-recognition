"""
Voice tab: bilingual live transcript, STT recognition-language toggle, an
on/off switch (not a push button) for the Vosk engine, and a button to
replay the last sentence through the TTS/Bluetooth speaker.
"""
from PyQt5.QtCore import pyqtSignal
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QTextEdit
)

from ui.toggle_switch import ToggleSwitch
from voice.stt_engine import STTEngine


class VoiceTab(QWidget):
    switchedOn = pyqtSignal()
    switchedOff = pyqtSignal()
    speakRequested = pyqtSignal(str, str)  # text, lang

    def __init__(self, translator, parent=None):
        super().__init__(parent)
        self.translator = translator
        self.engine = None
        self._stt_lang = "ar"
        self._last_text = ""

        root = QVBoxLayout(self)
        root.setContentsMargins(12, 12, 12, 12)
        root.setSpacing(8)

        top_row = QHBoxLayout()
        self.switch_label = QLabel()
        self.switch = ToggleSwitch()
        self.switch.toggledOn.connect(self._on_switch_toggled)

        self.lang_button = QPushButton()
        self.lang_button.clicked.connect(self._on_lang_toggle)

        self.speak_button = QPushButton()
        self.speak_button.clicked.connect(self._on_speak_last)

        top_row.addWidget(self.switch_label)
        top_row.addWidget(self.switch)
        top_row.addStretch()
        top_row.addWidget(self.lang_button)
        top_row.addWidget(self.speak_button)

        self.transcript = QTextEdit()
        self.transcript.setReadOnly(True)

        self.partial_label = QLabel()
        self.partial_label.setObjectName("sectionLabel")
        self.partial_label.setStyleSheet("color: #7b9098; font-style: italic;")
        self.partial_label.setWordWrap(True)

        clear_row = QHBoxLayout()
        clear_row.addStretch()
        self.clear_button = QPushButton()
        self.clear_button.clicked.connect(self.transcript.clear)
        clear_row.addWidget(self.clear_button)

        root.addLayout(top_row)
        root.addWidget(self.transcript, 1)
        root.addWidget(self.partial_label)
        root.addLayout(clear_row)

        self.retranslate()

    def retranslate(self):
        t = self.translator.t
        self.switch_label.setText(t("voice_switch"))
        lang_word = t("arabic") if self._stt_lang == "ar" else t("english")
        self.lang_button.setText(f"{t('stt_language')}: {lang_word}")
        self.speak_button.setText(t("repeat"))
        self.clear_button.setText(t("clear"))

    def _on_switch_toggled(self, checked):
        if checked:
            self._start_engine()
        else:
            self._stop_engine()

    def _start_engine(self):
        self.engine = STTEngine()
        self.engine.language = self._stt_lang
        self.engine.textReady.connect(self._on_text)
        self.engine.errorOccurred.connect(self._on_error)
        self.engine.statusChanged.connect(self._on_status_changed)
        self.engine.modelLoading.connect(self._on_model_loading)
        self.engine.modelReady.connect(self._on_model_ready)
        self.partial_label.setText("🎙 Initialising microphone…")
        self.engine.start()
        self.switchedOn.emit()

    def _stop_engine(self):
        if self.engine:
            self.engine.stop()
            self.engine = None
        self.partial_label.setText("")
        self.switchedOff.emit()

    def _on_lang_toggle(self):
        self._stt_lang = "en" if self._stt_lang == "ar" else "ar"
        if self.engine:
            self.engine.set_language(self._stt_lang)
        self.retranslate()

    def _on_model_loading(self, lang):
        self.partial_label.setText("🎙 Starting microphone…")

    def _on_model_ready(self, lang):
        t = self.translator.t
        self.partial_label.setText(f"🟢 {t('listening')}")

    def _on_text(self, text, lang, is_partial):
        if is_partial:
            self.partial_label.setText(text)
            return
        self.partial_label.setText("")
        self._last_text = text
        flag = "🇸🇦" if lang == "ar" else "🇺🇸"
        self.transcript.append(f"{flag} {text}")

    def _on_status_changed(self, status):
        t = self.translator.t
        if status == "listening":
            self.partial_label.setText(f"🎤 {t('listening')}")
        elif status == "processing":
            self.partial_label.setText("…")
        else:
            self.partial_label.setText("")

    def _on_speak_last(self):
        if self._last_text:
            self.speakRequested.emit(self._last_text, self._stt_lang)

    def _on_error(self, message):
        self.partial_label.setText(f"⚠ {message}")
        self.partial_label.setStyleSheet("color: #f0a3a3; font-style: normal;")
        self.switch.setChecked(False)
        self.switchedOff.emit()

    def is_active(self):
        return self.engine is not None and self.engine.isRunning()

    def shutdown(self):
        if self.engine:
            self.engine.stop()
