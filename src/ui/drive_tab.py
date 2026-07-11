"""
Drive tab — ultrasonic obstacle display only.
Motor control and head stepper jog removed; this tab now acts purely as a
live forward-obstacle monitor that the dashboard always shows regardless of
which mode is active.
"""
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QFrame, QProgressBar
)

import config


class DriveTab(QWidget):
    def __init__(self, translator, parent=None):
        super().__init__(parent)
        self.translator = translator
        self._last_distance = config.ULTRASONIC_WARNING_CM + 10

        root = QVBoxLayout(self)
        root.setContentsMargins(24, 24, 24, 24)
        root.setSpacing(16)
        root.setAlignment(Qt.AlignCenter)

        # ── Main panel ─────────────────────────────────────────────────────
        panel = QFrame()
        panel.setObjectName("panel")
        panel_layout = QVBoxLayout(panel)
        panel_layout.setContentsMargins(32, 28, 32, 28)
        panel_layout.setSpacing(16)
        panel_layout.setAlignment(Qt.AlignCenter)

        # Section label
        self.us_label = QLabel()
        self.us_label.setObjectName("sectionLabel")
        self.us_label.setAlignment(Qt.AlignCenter)

        # Large distance readout
        self.distance_value = QLabel("— cm")
        self.distance_value.setAlignment(Qt.AlignCenter)
        self.distance_value.setStyleSheet(
            "font-size: 52px; font-weight: bold; color: #22d3a4;"
        )

        # Distance progress bar (0–200 cm range)
        self.us_bar = QProgressBar()
        self.us_bar.setRange(0, 200)
        self.us_bar.setValue(200)
        self.us_bar.setTextVisible(False)
        self.us_bar.setFixedHeight(10)

        # Min / max range labels beneath bar
        range_row = QHBoxLayout()
        min_label = QLabel("0 cm")
        min_label.setObjectName("sectionLabel")
        max_label = QLabel("200 cm")
        max_label.setObjectName("sectionLabel")
        range_row.addWidget(min_label)
        range_row.addStretch()
        range_row.addWidget(max_label)

        # Status banner
        self.us_banner = QLabel()
        self.us_banner.setAlignment(Qt.AlignCenter)
        self.us_banner.setFixedHeight(36)
        self.us_banner.setStyleSheet(
            "font-size: 13px; font-weight: bold; border-radius: 8px;"
        )

        panel_layout.addWidget(self.us_label)
        panel_layout.addWidget(self.distance_value)
        panel_layout.addWidget(self.us_bar)
        panel_layout.addLayout(range_row)
        panel_layout.addWidget(self.us_banner)

        root.addStretch()
        root.addWidget(panel)
        root.addStretch()

        self.retranslate()

    def retranslate(self):
        t = self.translator.t
        self.us_label.setText(t("forward_obstacle"))
        self._update_banner(self._last_distance)

    def on_ultrasonic_reading(self, distance_cm):
        """Slot wired to UltrasonicSensor.distanceReady in main.py."""
        self._last_distance = distance_cm
        self.distance_value.setText(f"{int(distance_cm)} cm")
        self.us_bar.setValue(int(min(distance_cm, 200)))
        self._update_banner(distance_cm)

    def _update_banner(self, distance_cm):
        t = self.translator.t
        if distance_cm < config.ULTRASONIC_STOP_CM:
            self.us_banner.setText(t("obstacle_blocked"))
            self.us_banner.setStyleSheet(
                "background:#2a1414; color:#f0a3a3; "
                "border-radius:8px; font-size:13px; font-weight:bold;"
            )
        elif distance_cm < config.ULTRASONIC_WARNING_CM:
            self.us_banner.setText(t("obstacle_slow"))
            self.us_banner.setStyleSheet(
                "background:#2a2210; color:#f0c87a; "
                "border-radius:8px; font-size:13px; font-weight:bold;"
            )
        else:
            self.us_banner.setText(t("obstacle_clear"))
            self.us_banner.setStyleSheet(
                "background:#11241d; color:#4fd19c; "
                "border-radius:8px; font-size:13px; font-weight:bold;"
            )
