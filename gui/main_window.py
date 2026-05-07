"""
gui/main_window.py
Interface PyQt5 — Système d'Identification Biométrique Multimodale.
Style : moderne, épuré, inspiré Material Design (Google/Gemini).
"""

import cv2
import time
import threading
from datetime import datetime

from PyQt5.QtWidgets import (
    QMainWindow, QWidget, QLabel, QPushButton, QTextEdit,
    QVBoxLayout, QHBoxLayout, QFrame,
    QInputDialog, QMessageBox, QTableWidget, QTableWidgetItem,
    QDialog, QHeaderView, QGraphicsDropShadowEffect,
)
from PyQt5.QtCore import Qt, pyqtSignal, QObject
from PyQt5.QtGui import QImage, QPixmap, QColor

from core.detection import FaceDetector
from core.recognition import FaceRecognizer
from core.enrollment import EnrollmentPipeline
from core.fusion import decide
from core.template_watermark import TemplateWatermarker
from database.db_manager import DatabaseManager


# ── Palette Google / Gemini ──────────────────────────────────────────────────

BG      = "#F8F9FA"
SURFACE = "#FFFFFF"
BLUE    = "#1A73E8"
BLUE_L  = "#E8F0FE"
BLUE_D  = "#1557B0"
TEXT    = "#202124"
TEXT2   = "#5F6368"
BORDER  = "#DADCE0"
GREEN   = "#1E8E3E"
GREEN_L = "#E6F4EA"
RED     = "#D93025"
RED_L   = "#FCE8E6"
AMBER   = "#F29900"
AMBER_L = "#FEF7E0"

DECISION = {
    "authorized":   (GREEN, GREEN_L, "Autorisé"),
    "unauthorized": (RED,   RED_L,   "Refusé"),
    "unknown":      (AMBER, AMBER_L, "Inconnu"),
    "info":         (BLUE,  BLUE_L,  "Info"),
    "error":        (RED,   RED_L,   "Erreur"),
}


# ── Signaux ──────────────────────────────────────────────────────────────────

class Signals(QObject):
    frame_ready = pyqtSignal(object)
    log_ready   = pyqtSignal(str, str)


# ── Composants ───────────────────────────────────────────────────────────────

def _btn_filled(text, slot, color=BLUE, hover=BLUE_D):
    b = QPushButton(text)
    b.setStyleSheet(
        f"QPushButton {{ background:{color}; color:white; border:none;"
        f" border-radius:20px; padding:10px 22px; font-size:13px; font-weight:500; }}"
        f"QPushButton:hover {{ background:{hover}; }}"
        f"QPushButton:disabled {{ background:{BORDER}; color:{TEXT2}; }}")
    b.setCursor(Qt.PointingHandCursor)
    b.clicked.connect(slot)
    return b


def _btn_text(text, slot, color=BLUE):
    b = QPushButton(text)
    b.setStyleSheet(
        f"QPushButton {{ background:transparent; color:{color}; border:1.5px solid {color};"
        f" border-radius:20px; padding:9px 22px; font-size:13px; font-weight:500; }}"
        f"QPushButton:hover {{ background:{BLUE_L}; }}"
        f"QPushButton:disabled {{ color:{TEXT2}; border-color:{BORDER}; }}")
    b.setCursor(Qt.PointingHandCursor)
    b.clicked.connect(slot)
    return b


def _lbl(text, size=13, color=TEXT, bold=False):
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


def _section(title):
    return _lbl(title, 11, TEXT2)


# ── Fenêtre principale ───────────────────────────────────────────────────────

class MainWindow(QMainWindow):

    def __init__(self):
        super().__init__()
        self.setWindowTitle("BioAuth — Identification Biométrique")
        self.setMinimumSize(1080, 660)
        self.setStyleSheet(f"QMainWindow, QWidget {{ background:{BG}; }}")

        self.db         = DatabaseManager()
        self.detector   = FaceDetector()
        self.recognizer = FaceRecognizer()
        self.twm        = TemplateWatermarker()
        self.enrollment = EnrollmentPipeline(db=self.db)
        self.camera_id  = "CAM-001"
        self.cap        = None
        self.running    = False

        self.sig = Signals()
        self.sig.frame_ready.connect(self._show_frame)
        self.sig.log_ready.connect(self._on_log)

        self._build_ui()

    # ── UI ───────────────────────────────────────────────────────────────────

    def _build_ui(self):
        root = QWidget()
        self.setCentralWidget(root)
        main = QVBoxLayout(root)
        main.setContentsMargins(24, 16, 24, 16)
        main.setSpacing(14)
        main.addWidget(self._make_header())
        body = QHBoxLayout()
        body.setSpacing(14)
        body.addLayout(self._make_left(), stretch=5)
        body.addLayout(self._make_right(), stretch=2)
        main.addLayout(body, stretch=1)

    def _make_header(self):
        w = QWidget()
        w.setFixedHeight(44)
        w.setStyleSheet("background:transparent; border:none;")
        lay = QHBoxLayout(w)
        lay.setContentsMargins(0, 0, 0, 0)

        dot = QLabel("◉")
        dot.setStyleSheet(f"color:{BLUE}; font-size:18px; background:transparent; border:none;")
        lay.addWidget(dot)

        lay.addWidget(_lbl("BioAuth", 16, TEXT, bold=True))
        lay.addWidget(_lbl("·", 14, TEXT2))
        lay.addWidget(_lbl("Identification Biométrique Multimodale", 13, TEXT2))
        lay.addStretch()

        self.status_chip = QLabel("Inactif")
        self.status_chip.setStyleSheet(
            f"background:{BORDER}; color:{TEXT2}; border-radius:12px;"
            f" padding:4px 16px; font-size:12px; font-weight:500; border:none;")
        lay.addWidget(self.status_chip)
        return w

    def _make_left(self):
        col = QVBoxLayout()
        col.setSpacing(10)

        # Carte vidéo
        vc = QWidget()
        vc.setStyleSheet(
            f"background:{SURFACE}; border-radius:16px; border:1px solid {BORDER};")
        vcl = QVBoxLayout(vc)
        vcl.setContentsMargins(8, 8, 8, 8)

        self.video_label = QLabel("Aucune session en cours")
        self.video_label.setAlignment(Qt.AlignCenter)
        self.video_label.setMinimumSize(640, 430)
        self.video_label.setStyleSheet(
            f"background:#F1F3F4; border-radius:10px; color:{TEXT2};"
            f" font-size:14px; border:none;")
        vcl.addWidget(self.video_label)
        col.addWidget(vc, stretch=1)

        # Bandeau résultat
        rc = QWidget()
        rc.setFixedHeight(62)
        rc.setStyleSheet(
            f"background:{SURFACE}; border-radius:16px; border:1px solid {BORDER};")
        rcl = QHBoxLayout(rc)
        rcl.setContentsMargins(20, 0, 20, 0)
        rcl.setSpacing(14)

        self.result_chip = QLabel("—")
        self.result_chip.setFixedWidth(88)
        self.result_chip.setAlignment(Qt.AlignCenter)
        self.result_chip.setStyleSheet(
            f"background:{BORDER}; color:{TEXT2}; border-radius:12px;"
            f" padding:4px 10px; font-size:12px; font-weight:500; border:none;")
        rcl.addWidget(self.result_chip)

        self.result_name = _lbl("En attente…", 15, TEXT2)
        rcl.addWidget(self.result_name)
        rcl.addStretch()

        self.result_score = _lbl("", 13, TEXT2)
        rcl.addWidget(self.result_score)

        col.addWidget(rc)
        return col

    def _make_right(self):
        col = QVBoxLayout()
        col.setSpacing(10)

        # Carte actions
        ac = QWidget()
        ac.setStyleSheet(
            f"background:{SURFACE}; border-radius:16px; border:1px solid {BORDER};")
        acl = QVBoxLayout(ac)
        acl.setContentsMargins(16, 16, 16, 16)
        acl.setSpacing(8)

        acl.addWidget(_section("Gestion"))
        acl.addSpacing(2)
        self.btn_enroll = _btn_filled("+ Enrôler une personne", self.enroll_person)
        acl.addWidget(self.btn_enroll)
        self.btn_train = _btn_text("Entraîner le modèle", self.train_model)
        acl.addWidget(self.btn_train)

        acl.addSpacing(6)
        acl.addWidget(_divider())
        acl.addSpacing(6)

        acl.addWidget(_section("Session"))
        acl.addSpacing(2)
        self.btn_start = _btn_filled("▶  Lancer la session", self.start_auth, GREEN, "#156A31")
        acl.addWidget(self.btn_start)
        self.btn_stop = _btn_text("■  Arrêter", self.stop_auth, RED)
        self.btn_stop.setEnabled(False)
        acl.addWidget(self.btn_stop)

        acl.addSpacing(6)
        acl.addWidget(_divider())
        acl.addSpacing(6)

        acl.addWidget(_section("Outils"))
        acl.addSpacing(2)
        self.btn_integrity = _btn_text("Vérifier intégrité BDD", self.verify_integrity, TEXT2)
        acl.addWidget(self.btn_integrity)
        self.btn_history = _btn_text("Historique sessions", self.show_history, TEXT2)
        acl.addWidget(self.btn_history)

        col.addWidget(ac)

        # Carte journal
        lc = QWidget()
        lc.setStyleSheet(
            f"background:{SURFACE}; border-radius:16px; border:1px solid {BORDER};")
        lcl = QVBoxLayout(lc)
        lcl.setContentsMargins(14, 12, 14, 12)
        lcl.setSpacing(8)
        lcl.addWidget(_section("Journal"))
        self.log_box = QTextEdit()
        self.log_box.setReadOnly(True)
        self.log_box.setStyleSheet(
            f"background:{BG}; color:{TEXT}; font-size:11px;"
            f" font-family:monospace; border:none; border-radius:8px;")
        lcl.addWidget(self.log_box, stretch=1)
        col.addWidget(lc, stretch=1)

        return col

    # ── Actions ───────────────────────────────────────────────────────────────

    def enroll_person(self):
        if self.running:
            self.stop_auth()
        name, ok = QInputDialog.getText(self, "Enrôlement", "Nom de la personne :")
        if not ok or not name.strip():
            return
        status, ok = QInputDialog.getItem(
            self, "Statut", "Statut :", ["authorized", "unauthorized"], 0, False)
        if not ok:
            return
        name = name.strip()
        self._log(f"Enrôlement de {name} ({status})…", "info")
        def _run():
            ok2, msg = self.enrollment.enroll(name, status, self.camera_id)
            self.sig.log_ready.emit(msg, "info" if ok2 else "error")
        threading.Thread(target=_run, daemon=True).start()

    def train_model(self):
        if self.running:
            self.stop_auth()
        self._log("Entraînement SVM HOG+LBP…", "info")
        def _run():
            try:
                self.recognizer.train()
                self.sig.log_ready.emit("Modèle entraîné avec succès.", "info")
            except Exception as e:
                self.sig.log_ready.emit(f"Erreur : {e}", "error")
        threading.Thread(target=_run, daemon=True).start()

    def start_auth(self):
        if self.running:
            return
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            QMessageBox.critical(self, "Erreur", "Impossible d'ouvrir la caméra.")
            return
        self.running = True
        self.btn_start.setEnabled(False)
        self.btn_stop.setEnabled(True)
        self._set_chip(self.status_chip, "Session active", BLUE, BLUE_L)
        threading.Thread(target=self._auth_loop, daemon=True).start()
        self._log("Session démarrée.", "info")

    def stop_auth(self):
        self.running = False
        time.sleep(0.3)
        if self.cap:
            self.cap.release()
            self.cap = None
        self.btn_start.setEnabled(True)
        self.btn_stop.setEnabled(False)
        self._set_chip(self.status_chip, "Inactif", TEXT2, BORDER)
        self.video_label.setText("Session terminée.")
        self._log("Session arrêtée.", "info")

    def _auth_loop(self):
        frame_idx  = 0
        last_faces = []
        last_label = ""
        last_color = (180, 180, 180)

        while self.running:
            ret, frame = self.cap.read()
            if not ret:
                break
            frame_idx += 1

            if frame_idx % 3 == 0:
                small = cv2.resize(frame, (0, 0), fx=0.75, fy=0.75)
                sc    = 1 / 0.75
                last_faces = [
                    (int(x*sc), int(y*sc), int(w*sc), int(h*sc))
                    for x, y, w, h in self.detector.detect(small)
                ]

            if frame_idx % 60 == 0 and last_faces:
                x, y, w, h = last_faces[0]
                roi = frame[y:y+h, x:x+w]
                if roi.size > 0:
                    face_result = self.recognizer.predict(roi)
                    result      = decide(face_result)
                    integrity   = self._check_integrity(result["name"])

                    last_label = f"{result['name']}  {result['score']*100:.0f}%"
                    cmap = {
                        "authorized":   (30, 142, 62),
                        "unauthorized": (217, 48, 37),
                        "unknown":      (242, 153, 0),
                    }
                    last_color = cmap.get(result["decision"], (150, 150, 150))

                    self.db.log_auth(
                        result["name"], result["decision"],
                        result["face_score"], result["fp_score"],
                        result["score"], result["detail"])

                    msg = result["detail"] + (f" | {integrity}" if integrity else "")
                    self.sig.log_ready.emit(msg, result["decision"])
                    self.sig.log_ready.emit(
                        f"__result__|{result['decision']}|{result['name']}|{result['score']*100:.0f}%",
                        "__result__")

            display = frame.copy()
            self.detector.draw(display, last_faces, last_label, last_color)
            self.sig.frame_ready.emit(display)

        self.running = False

    def _check_integrity(self, name):
        if name == "Inconnu":
            return ""
        pid = self.db.get_person_id(name)
        if not pid:
            return ""
        tmpl = self.db.load_template(pid, "face")
        if not tmpl:
            return ""
        _, msg, _ = self.twm.verify(tmpl["vector"], tmpl["wm_payload"], tmpl["checksum"])
        return msg

    def verify_integrity(self):
        persons = self.db.list_persons()
        if not persons:
            QMessageBox.information(self, "Intégrité", "Aucune personne en base.")
            return
        self._log("Vérification des gabarits…", "info")
        for pid, name, status in persons:
            tmpl = self.db.load_template(pid, "face")
            if not tmpl:
                self._log(f"{name} — aucun gabarit", "unknown")
                continue
            ok, msg, _ = self.twm.verify(tmpl["vector"], tmpl["wm_payload"], tmpl["checksum"])
            self._log(f"{name} : {msg}", "authorized" if ok else "unauthorized")

    def show_history(self):
        events = self.db.get_auth_events(limit=40)
        dlg = QDialog(self)
        dlg.setWindowTitle("Historique des authentifications")
        dlg.setStyleSheet(f"background:{BG};")
        dlg.resize(960, 460)
        lay  = QVBoxLayout(dlg)
        cols = ["ID", "Nom", "Décision", "Score visage", "Score fusion", "Détail", "Horodatage"]
        table = QTableWidget(len(events), len(cols))
        table.setHorizontalHeaderLabels(cols)
        table.setStyleSheet(
            f"QTableWidget {{ background:{SURFACE}; color:{TEXT}; font-size:12px;"
            f" gridline-color:{BORDER}; border:1px solid {BORDER}; border-radius:12px; }}"
            f"QHeaderView::section {{ background:{BG}; color:{TEXT2}; font-size:11px;"
            f" font-weight:500; padding:6px; border:none;"
            f" border-bottom:1px solid {BORDER}; }}")
        table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        table.verticalHeader().setVisible(False)
        table.setAlternatingRowColors(True)
        table.setSelectionBehavior(QTableWidget.SelectRows)
        for row, e in enumerate(events):
            vals = [str(e[0]), str(e[1]), str(e[2]),
                    f"{(e[3] or 0)*100:.1f}%", f"{(e[5] or 0)*100:.1f}%",
                    str(e[6]), str(e[7])]
            for col, val in enumerate(vals):
                item = QTableWidgetItem(val)
                item.setFlags(item.flags() & ~Qt.ItemIsEditable)
                if col == 2:
                    cm = {"authorized": GREEN, "unauthorized": RED, "unknown": AMBER}
                    item.setForeground(QColor(cm.get(val, TEXT2)))
                table.setItem(row, col, item)
        lay.addWidget(table)
        dlg.exec_()

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _set_chip(self, chip, text, fg, bg):
        chip.setText(text)
        chip.setStyleSheet(
            f"background:{bg}; color:{fg}; border-radius:12px;"
            f" padding:4px 16px; font-size:12px; font-weight:500; border:none;")

    def _show_frame(self, frame):
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w = rgb.shape[:2]
        img  = QImage(rgb.data, w, h, w * 3, QImage.Format_RGB888)
        pix  = QPixmap.fromImage(img).scaled(
            640, 430, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self.video_label.setPixmap(pix)

    def _on_log(self, message, decision):
        if decision == "__result__":
            parts = message.split("|")
            if len(parts) >= 4:
                dec, name, score = parts[1], parts[2], parts[3]
                fg, bg, lbl = DECISION.get(dec, (TEXT2, BORDER, "—"))
                self._set_chip(self.result_chip, lbl, fg, bg)
                self.result_name.setText(name)
                self.result_name.setStyleSheet(
                    f"color:{fg}; font-size:15px; font-weight:500;"
                    f" background:transparent; border:none;")
                self.result_score.setText(f"score {score}")
            return

        color = DECISION.get(decision, (TEXT2, BORDER, ""))[0]
        ts    = datetime.now().strftime("%H:%M:%S")
        self.log_box.append(
            f'<span style="color:{TEXT2}">[{ts}]</span>'
            f' <span style="color:{color}">{message}</span>')
        self.log_box.verticalScrollBar().setValue(
            self.log_box.verticalScrollBar().maximum())

    def _log(self, msg, typ="info"):
        self.sig.log_ready.emit(msg, typ)

    def closeEvent(self, event):
        self.running = False
        time.sleep(0.3)
        if self.cap:
            self.cap.release()
        self.db.close()
        event.accept()