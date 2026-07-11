"""Shared dark-theme QSS for the Sign-Bot dashboard, 800x480 touchscreen."""

COLORS = {
    "bg_screen": "#0b1216",
    "panel_bg": "#131c21",
    "border": "#223139",
    "text_primary": "#e8f0f2",
    "text_secondary": "#7b9098",
    "accent": "#22d3a4",
    "accent_dark": "#17b386",
    "danger_bg": "#2a1414",
    "danger_text": "#f0a3a3",
    "warning_bg": "#2a2210",
    "warning_text": "#f0c87a",
    "success_bg": "#11241d",
    "success_text": "#4fd19c",
}

MAIN_QSS = f"""
QWidget {{
    background-color: {COLORS['bg_screen']};
    color: {COLORS['text_primary']};
    font-family: 'DejaVu Sans', sans-serif;
    font-size: 12px;
}}
QFrame#panel {{
    background-color: {COLORS['panel_bg']};
    border: 1px solid {COLORS['border']};
    border-radius: 10px;
}}
QPushButton {{
    background-color: {COLORS['panel_bg']};
    border: 1px solid {COLORS['border']};
    border-radius: 8px;
    color: {COLORS['text_primary']};
    padding: 8px;
    font-size: 12px;
}}
QPushButton:pressed {{
    background-color: {COLORS['border']};
}}
QPushButton#primaryButton {{
    background-color: {COLORS['accent_dark']};
    color: #06140f;
    font-weight: bold;
    border: none;
}}
QPushButton#stopButton {{
    background-color: {COLORS['danger_bg']};
    border: 1px solid #5a2424;
    color: {COLORS['danger_text']};
    font-weight: bold;
}}
QPushButton#tabButton {{
    background: none;
    border: none;
    border-top: 1px solid transparent;
    color: {COLORS['text_secondary']};
    font-size: 11px;
    padding: 8px 0;
}}
QPushButton#tabButton:checked {{
    color: {COLORS['accent']};
}}
QLabel#sectionLabel {{
    color: {COLORS['text_secondary']};
    font-size: 11px;
}}
QTextEdit {{
    background-color: {COLORS['panel_bg']};
    border: 1px solid {COLORS['border']};
    border-radius: 10px;
    padding: 6px;
}}
QSlider::groove:horizontal {{
    height: 6px;
    background: {COLORS['border']};
    border-radius: 3px;
}}
QSlider::handle:horizontal {{
    background: {COLORS['accent']};
    width: 16px;
    margin: -5px 0;
    border-radius: 8px;
}}
QProgressBar {{
    background-color: {COLORS['border']};
    border: none;
    border-radius: 3px;
    height: 6px;
    text-align: center;
}}
QProgressBar::chunk {{
    background-color: {COLORS['accent']};
    border-radius: 3px;
}}
"""
