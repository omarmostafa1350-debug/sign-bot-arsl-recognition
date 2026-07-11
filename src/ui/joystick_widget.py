"""
Touch-drag virtual joystick. Emits normalized (x, y) in [-1, 1] while
dragging; auto-recenters and emits (0, 0) on release so the robot can never
keep driving after a finger slips off the screen.
"""
from PyQt5.QtCore import Qt, QPointF, pyqtSignal
from PyQt5.QtGui import QPainter, QColor, QPen
from PyQt5.QtWidgets import QWidget


class JoystickWidget(QWidget):
    positionChanged = pyqtSignal(float, float)  # x, y in [-1, 1]

    def __init__(self, parent=None, diameter=120, knob_diameter=44):
        super().__init__(parent)
        self._diameter = diameter
        self._knob_diameter = knob_diameter
        self._radius = (diameter - knob_diameter) / 2.0
        self._knob_offset = QPointF(0, 0)  # offset from center, pixels
        self._dragging = False
        self.setFixedSize(diameter, diameter)

    def _center(self):
        return QPointF(self.width() / 2.0, self.height() / 2.0)

    def _update_knob(self, offset: QPointF):
        dist = (offset.x() ** 2 + offset.y() ** 2) ** 0.5
        if dist > self._radius and dist > 0:
            scale = self._radius / dist
            offset = QPointF(offset.x() * scale, offset.y() * scale)
        self._knob_offset = offset
        x = offset.x() / self._radius if self._radius else 0.0
        y = -offset.y() / self._radius if self._radius else 0.0  # up = positive (forward)
        self.positionChanged.emit(x, y)
        self.update()

    def mousePressEvent(self, event):
        self._dragging = True
        self._update_knob(event.pos() - self._center())

    def mouseMoveEvent(self, event):
        if self._dragging:
            self._update_knob(event.pos() - self._center())

    def mouseReleaseEvent(self, event):
        self._dragging = False
        self._update_knob(QPointF(0, 0))

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)

        base_rect = self.rect().adjusted(2, 2, -2, -2)
        painter.setBrush(QColor("#0f181c"))
        painter.setPen(QPen(QColor("#223139"), 1))
        painter.drawEllipse(base_rect)

        knob_center = self._center() + self._knob_offset
        painter.setBrush(QColor("#22d3a4"))
        painter.setPen(Qt.NoPen)
        painter.drawEllipse(knob_center, self._knob_diameter / 2, self._knob_diameter / 2)
