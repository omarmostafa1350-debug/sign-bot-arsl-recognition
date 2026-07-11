"""
A sliding toggle switch (not a push button) for the Sign and Voice on/off
controls — matches the "switch, not push buttons" behavior: flipping it on
starts the camera/mic pipeline, flipping it off cleanly tears it down.
"""
from PyQt5.QtCore import Qt, QPropertyAnimation, QRectF, pyqtProperty, pyqtSignal
from PyQt5.QtGui import QPainter, QColor
from PyQt5.QtWidgets import QAbstractButton


class ToggleSwitch(QAbstractButton):
    toggledOn = pyqtSignal(bool)

    def __init__(self, parent=None, width=52, height=28):
        super().__init__(parent)
        self.setCheckable(True)
        self._w, self._h = width, height
        self.setFixedSize(width, height)
        self._offset = 3.0
        self._anim = QPropertyAnimation(self, b"offset", self)
        self._anim.setDuration(150)
        self.toggled.connect(self._on_toggled)

    def _on_toggled(self, checked):
        end = self._w - self._h + 3.0 if checked else 3.0
        self._anim.stop()
        self._anim.setStartValue(self._offset)
        self._anim.setEndValue(end)
        self._anim.start()
        self.toggledOn.emit(checked)

    def getOffset(self):
        return self._offset

    def setOffset(self, value):
        self._offset = value
        self.update()

    offset = pyqtProperty(float, getOffset, setOffset)

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        track_color = QColor("#17b386") if self.isChecked() else QColor("#223139")
        painter.setPen(Qt.NoPen)
        painter.setBrush(track_color)
        painter.drawRoundedRect(self.rect(), self._h / 2, self._h / 2)

        knob_d = self._h - 6
        painter.setBrush(QColor("#e8f0f2"))
        painter.drawEllipse(QRectF(self._offset, 3, knob_d, knob_d))
