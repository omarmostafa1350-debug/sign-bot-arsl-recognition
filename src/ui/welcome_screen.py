"""
Welcome screen: logo, tagline, Start button. Triggers a head-stepper
greeting nod and a bilingual spoken welcome the first time it's shown.
"""
from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtWidgets import QWidget, QVBoxLayout, QLabel, QPushButton


class WelcomeScreen(QWidget):
    startClicked = pyqtSignal()

    def __init__(self, translator, head_stepper, tts_engine, parent=None):
        super().__init__(parent)
        self.translator = translator
        self.head_stepper = head_stepper
        self.tts_engine = tts_engine
        self._greeted = False

        layout = QVBoxLayout(self)
        layout.setAlignment(Qt.AlignCenter)
        layout.setSpacing(14)

        self.title_label = QLabel()
        self.title_label.setAlignment(Qt.AlignCenter)
        self.title_label.setStyleSheet("font-size: 22px; font-weight: bold;")

        self.tagline_label = QLabel()
        self.tagline_label.setAlignment(Qt.AlignCenter)
        self.tagline_label.setStyleSheet("font-size: 12px; color: #7b9098;")

        self.start_button = QPushButton()
        self.start_button.setObjectName("primaryButton")
        self.start_button.setFixedSize(160, 44)
        self.start_button.clicked.connect(self.startClicked.emit)

        layout.addWidget(self.title_label)
        layout.addWidget(self.tagline_label)
        layout.addWidget(self.start_button, alignment=Qt.AlignCenter)

        self.retranslate()

    def retranslate(self):
        t = self.translator.t
        self.title_label.setText(t("app_name"))
        self.tagline_label.setText(t("app_tagline"))
        self.start_button.setText(t("start"))

    def showEvent(self, event):
        super().showEvent(event)
        if not self._greeted:
            self._greeted = True
            self.head_stepper.greeting_nod()
            self.tts_engine.speak("Welcome to Sign-Bot", lang="en")
            self.tts_engine.speak("مرحبا بكم في ساين بوت", lang="ar")
