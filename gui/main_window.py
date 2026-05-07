"""
gui/main_window.py
------------------
Visual layout ONLY — no logic here.
Just buttons, labels, and panels.
"""

from PyQt5.QtWidgets import (
    QWidget, QLabel, QPushButton, QTextEdit,
    QVBoxLayout, QHBoxLayout, QFrame,
)
from PyQt5.QtCore import Qt


# ── Colors ────────────────────────────────────────────────────────────────────

BG      = "#F8F9FA"
SURFACE = "#FFFFFF"
BORDER  = "#DADCE0"
BLUE    = "#1A73E8"
BLUE_D  = "#1557B0"
BLUE_L  = "#E8F0FE"
GREEN   = "#1E8E3E"
GREEN_L = "#E6F4EA"
RED     = "#D93025"
RED_L   = "#FCE8E6"
AMBER   = "#F29900"
AMBER_L = "#FEF7E0"
TEXT    = "#202124"
TEXT2   = "#5F6368"

DECISION_COLORS = {
    "authorized":   (GREEN, GREEN_L, "Authorized"),
    "unauthorized": (RED,   RED_L,   "Refused"),
    "unknown":      (AMBER, AMBER_L, "Unknown"),
    "info":         (BLUE,  BLUE_L,  "Info"),
    "error":        (RED,   RED_L,   "Error"),
    "partial": ("#E37400", "#FEF0E0", "Partial"),
}


# ── Button helpers ────────────────────────────────────────────────────────────

def _btn_filled(text, color=BLUE, hover=BLUE_D):
    b = QPushButton(text)
    b.setStyleSheet(
        f"QPushButton {{ background:{color}; color:white; border:none;"
        f" border-radius:20px; padding:10px 22px; font-size:13px; font-weight:500; }}"
        f"QPushButton:hover {{ background:{hover}; }}"
        f"QPushButton:disabled {{ background:{BORDER}; color:{TEXT2}; }}")
    b.setCursor(Qt.PointingHandCursor)
    return b


def _btn_outlined(text, color=BLUE):
    b = QPushButton(text)
    b.setStyleSheet(
        f"QPushButton {{ background:transparent; color:{color};"
        f" border:1.5px solid {color}; border-radius:20px;"
        f" padding:9px 22px; font-size:13px; font-weight:500; }}"
        f"QPushButton:hover {{ background:{BLUE_L}; }}"
        f"QPushButton:disabled {{ color:{TEXT2}; border-color:{BORDER}; }}")
    b.setCursor(Qt.PointingHandCursor)
    return b


def _label(text, size=13, color=TEXT, bold=False):
    l = QLabel(text)
    l.setStyleSheet(
        f"color:{color}; font-size:{size}px;"
        f" font-weight:{'500' if bold else '400'};"
        f" background:transparent; border:none;")
    return l


def _divider():
    f = QFrame()
    f.setFrameShape(QFrame.HLine)
    f.setStyleSheet(f"background:{BORDER}; max-height:1px; border:none;")
    f.setFixedHeight(1)
    return f


def _section_title(text):
    return _label(text, size=11, color=TEXT2)


# ── Build UI ──────────────────────────────────────────────────────────────────

def build_ui(window):
    """
    Builds the full UI and attaches all widgets to `window`.
    All widgets become attributes of `window` (e.g. window.btn_enroll)
    so that app_logic.py can access them.
    """
    root = QWidget()
    window.setCentralWidget(root)
    root.setStyleSheet(f"background:{BG};")

    main_layout = QVBoxLayout(root)
    main_layout.setContentsMargins(24, 16, 24, 16)
    main_layout.setSpacing(14)

    main_layout.addWidget(_make_header(window))

    body = QHBoxLayout()
    body.setSpacing(14)
    body.addLayout(_make_left_panel(window), stretch=5)
    body.addLayout(_make_right_panel(window), stretch=2)
    main_layout.addLayout(body, stretch=1)


# ── Header ────────────────────────────────────────────────────────────────────

def _make_header(window):
    w = QWidget()
    w.setFixedHeight(44)
    w.setStyleSheet("background:transparent; border:none;")
    lay = QHBoxLayout(w)
    lay.setContentsMargins(0, 0, 0, 0)

    dot = QLabel("◉")
    dot.setStyleSheet(f"color:{BLUE}; font-size:18px; background:transparent; border:none;")
    lay.addWidget(dot)
    lay.addWidget(_label("BioAuth", size=16, bold=True))
    lay.addWidget(_label("·", size=14, color=TEXT2))
    lay.addWidget(_label("Multimodal Biometric Identification", size=13, color=TEXT2))
    lay.addStretch()

    window.status_chip = QLabel("Inactive")
    window.status_chip.setStyleSheet(
        f"background:{BORDER}; color:{TEXT2}; border-radius:12px;"
        f" padding:4px 16px; font-size:12px; font-weight:500; border:none;")
    lay.addWidget(window.status_chip)
    return w


# ── Left panel ────────────────────────────────────────────────────────────────

def _make_left_panel(window):
    col = QVBoxLayout()
    col.setSpacing(10)

    # Camera card
    camera_card = QWidget()
    camera_card.setStyleSheet(
        f"background:{SURFACE}; border-radius:16px; border:1px solid {BORDER};")
    card_layout = QVBoxLayout(camera_card)
    card_layout.setContentsMargins(8, 8, 8, 8)

    window.video_label = QLabel("No active session")
    window.video_label.setAlignment(Qt.AlignCenter)
    window.video_label.setMinimumSize(640, 430)
    window.video_label.setStyleSheet(
        f"background:#F1F3F4; border-radius:10px; color:{TEXT2}; font-size:14px; border:none;")
    card_layout.addWidget(window.video_label)
    col.addWidget(camera_card, stretch=1)

    # Result bar
    result_bar = QWidget()
    result_bar.setFixedHeight(62)
    result_bar.setStyleSheet(
        f"background:{SURFACE}; border-radius:16px; border:1px solid {BORDER};")
    bar_layout = QHBoxLayout(result_bar)
    bar_layout.setContentsMargins(20, 0, 20, 0)
    bar_layout.setSpacing(14)

    window.result_chip = QLabel("—")
    window.result_chip.setFixedWidth(90)
    window.result_chip.setAlignment(Qt.AlignCenter)
    window.result_chip.setStyleSheet(
        f"background:{BORDER}; color:{TEXT2}; border-radius:12px;"
        f" padding:4px 10px; font-size:12px; font-weight:500; border:none;")
    bar_layout.addWidget(window.result_chip)

    window.result_name = _label("Waiting…", size=15, color=TEXT2)
    bar_layout.addWidget(window.result_name)
    bar_layout.addStretch()

    window.result_score = _label("", size=13, color=TEXT2)
    bar_layout.addWidget(window.result_score)

    col.addWidget(result_bar)
    return col


# ── Right panel ───────────────────────────────────────────────────────────────

def _make_right_panel(window):
    col = QVBoxLayout()
    col.setSpacing(10)

    # Actions card
    actions_card = QWidget()
    actions_card.setStyleSheet(
        f"background:{SURFACE}; border-radius:16px; border:1px solid {BORDER};")
    actions_layout = QVBoxLayout(actions_card)
    actions_layout.setContentsMargins(16, 16, 16, 16)
    actions_layout.setSpacing(8)

    actions_layout.addWidget(_section_title("Management"))
    actions_layout.addSpacing(2)
    
    window.btn_enroll = _btn_filled("+ Enroll person")
    actions_layout.addWidget(window.btn_enroll)
    
    window.btn_enroll_fp = _btn_outlined("+ Enroll fingerprint")
    actions_layout.addWidget(window.btn_enroll_fp)
    
    window.btn_train = _btn_outlined("Train SVM model")
    actions_layout.addWidget(window.btn_train)

    actions_layout.addSpacing(6)
    actions_layout.addWidget(_divider())
    actions_layout.addSpacing(6)

    actions_layout.addWidget(_section_title("Session"))
    actions_layout.addSpacing(2)
    window.btn_start = _btn_filled("▶  Start session", color=GREEN, hover="#156A31")
    actions_layout.addWidget(window.btn_start)
    window.btn_stop = _btn_outlined("■  Stop", color=RED)
    window.btn_stop.setEnabled(False)
    actions_layout.addWidget(window.btn_stop)

    actions_layout.addSpacing(6)
    actions_layout.addWidget(_divider())
    actions_layout.addSpacing(6)

    actions_layout.addWidget(_section_title("Tools"))
    actions_layout.addSpacing(2)
    window.btn_integrity = _btn_outlined("Check DB integrity", color=TEXT2)
    actions_layout.addWidget(window.btn_integrity)
    window.btn_history = _btn_outlined("Session history", color=TEXT2)
    actions_layout.addWidget(window.btn_history)

    col.addWidget(actions_card)

    # Log card
    log_card = QWidget()
    log_card.setStyleSheet(
        f"background:{SURFACE}; border-radius:16px; border:1px solid {BORDER};")
    log_layout = QVBoxLayout(log_card)
    log_layout.setContentsMargins(14, 12, 14, 12)
    log_layout.setSpacing(8)
    log_layout.addWidget(_section_title("Log"))

    window.log_box = QTextEdit()
    window.log_box.setReadOnly(True)
    window.log_box.setStyleSheet(
        f"background:{BG}; color:{TEXT}; font-size:11px;"
        f" font-family:monospace; border:none; border-radius:8px;")
    log_layout.addWidget(window.log_box, stretch=1)

    col.addWidget(log_card, stretch=1)
    return col