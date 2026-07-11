"""
Bilingual UI strings (English / Arabic) for the dashboard chrome.
This does NOT translate detected sign labels or live transcript text —
those stay in whatever language they were actually signed/spoken in.
"""

STRINGS = {
    "en": {
        "app_name": "Sign-Bot",
        "app_tagline": "Sign language communication & mobility kiosk",
        "start": "Start",
        "tab_sign": "Sign",
        "tab_drive": "Drive",
        "tab_voice": "Voice",
        "camera_feed": "USB camera feed",
        "detected_sign": "Detected sign",
        "listening": "Listening...",
        "confidence": "Confidence",
        "fps": "FPS",
        "sign_switch": "Sign detection",
        "speak": "Speak",
        "head_angle": "Head angle",
        "left_motor": "Left motor",
        "right_motor": "Right motor",
        "forward_obstacle": "Forward obstacle",
        "emergency_stop": "Emergency stop",
        "obstacle_blocked": "Obstacle too close — forward blocked",
        "obstacle_slow": "Obstacle ahead — slow down",
        "obstacle_clear": "Path clear",
        "voice_switch": "Voice recognition",
        "stt_language": "Recognition language",
        "arabic": "Arabic",
        "english": "English",
        "repeat": "Repeat",
        "clear": "Clear",
        "starting": "Starting...",
    },
    "ar": {
        "app_name": "Sign-Bot",
        "app_tagline": "منصة تواصل بلغة الإشارة وتنقل",
        "start": "ابدأ",
        "tab_sign": "الإشارة",
        "tab_drive": "القيادة",
        "tab_voice": "الصوت",
        "camera_feed": "بث كاميرا USB",
        "detected_sign": "الإشارة المكتشفة",
        "listening": "بانتظار إشارة...",
        "confidence": "الثقة",
        "fps": "إطار/ث",
        "sign_switch": "كشف الإشارة",
        "speak": "نطق",
        "head_angle": "زاوية الرأس",
        "left_motor": "المحرك الأيسر",
        "right_motor": "المحرك الأيمن",
        "forward_obstacle": "عائق أمامي",
        "emergency_stop": "إيقاف طارئ",
        "obstacle_blocked": "عائق قريب جدًا — تم إيقاف الحركة الأمامية",
        "obstacle_slow": "عائق أمامي — أبطئ",
        "obstacle_clear": "الطريق آمن",
        "voice_switch": "التعرف على الصوت",
        "stt_language": "لغة التعرف",
        "arabic": "العربية",
        "english": "الإنجليزية",
        "repeat": "إعادة",
        "clear": "مسح",
        "starting": "جارٍ البدء...",
    },
}


class Translator:
    def __init__(self, lang="en"):
        self.lang = lang

    def set_language(self, lang):
        self.lang = lang

    def t(self, key):
        return STRINGS.get(self.lang, STRINGS["en"]).get(key, key)
