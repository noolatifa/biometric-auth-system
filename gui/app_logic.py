"""
gui/app_logic.py
----------------
Logic ONLY — no visual code here.
Each function does ONE thing. Read top to bottom.
"""

import cv2
import time
import threading
from datetime import datetime

from PyQt5.QtWidgets import QMainWindow, QInputDialog, QMessageBox, QFileDialog
from PyQt5.QtCore import Qt, pyqtSignal, QObject
from PyQt5.QtGui import QImage, QPixmap

from core.detection import FaceDetector
from core.recognition import FaceRecognizer
from core.fingerprint import FingerprintRecognizer
from core.enrollment import EnrollmentPipeline
from core.fusion import decide
from core.template_watermark import TemplateWatermarker
from database.db_manager import DatabaseManager

# Import the UI builder and colors from main_window.py
from gui.main_window import build_ui, DECISION_COLORS, BORDER, TEXT, TEXT2


# ── Signals ───────────────────────────────────────────────────────────────────
# Signals let the camera thread safely update the UI
# Rule: never update the UI directly from a background thread

class Signals(QObject):
    new_frame = pyqtSignal(object)    # sends a camera frame to display
    new_log   = pyqtSignal(str, str)  # sends (message, type) to the log
    ask_fingerprint = pyqtSignal(dict)   # sends {"name": ..., "face_score": ...} to ask for fingerprint scan


# ── Main class ────────────────────────────────────────────────────────────────

class AppLogic(QMainWindow):

    def __init__(self):
        super().__init__()
        self.setWindowTitle("BioAuth — Biometric Identification")
        self.setMinimumSize(1080, 660)

        # Load all modules
        self.db         = DatabaseManager()
        self.detector   = FaceDetector()
        self.recognizer = FaceRecognizer()
        self.twm        = TemplateWatermarker()
        self.enrollment = EnrollmentPipeline(db=self.db)
        self._fp_event     = threading.Event()
        self._fp_result    = None

        # Camera state
        self.camera  = None
        self.running = False
        self._last_face_pos = None

        # Build the visual layout (defined in main_window.py)
        build_ui(self)

        # Setup signals
        self.sig = Signals()
        self.sig.new_frame.connect(self._display_frame)
        self.sig.new_log.connect(self._handle_log)

        # Connect buttons → functions
        self.btn_enroll.clicked.connect(self.enroll)
        self.btn_train.clicked.connect(self.train)
        self.btn_start.clicked.connect(self.start)
        self.btn_stop.clicked.connect(self.stop)
        self.btn_integrity.clicked.connect(self.check_integrity)
        self.btn_enroll_fp.clicked.connect(self.enroll_fingerprint)
        self.sig.ask_fingerprint.connect(self._ask_fingerprint_popup)
        self.btn_delete.clicked.connect(self.delete_person)
        self.fp_recognizer = FingerprintRecognizer()
    # ── 1. Enroll a person ────────────────────────────────────────────────────

    def enroll(self):
        # Ask for name
        name, ok = QInputDialog.getText(self, "Enroll", "Name:")
        if not ok or not name.strip():
            return

        # Ask for status
        status, ok = QInputDialog.getItem(
            self, "Status", "Status:", ["authorized", "unauthorized"], 0, False)
        if not ok:
            return

        # Run in background so the UI doesn't freeze
        def run():
            ok2, msg = self.enrollment.enroll(name.strip(), status)
            self.sig.new_log.emit(msg, "info" if ok2 else "error")

        threading.Thread(target=run, daemon=True).start()

    # ── 2. Train the SVM model ────────────────────────────────────────────────

    def train(self):
        def run():
            try:
                self.recognizer.train()
                self.sig.new_log.emit("Model trained successfully.", "info")
            except Exception as e:
                self.sig.new_log.emit(f"Training error: {e}", "error")

        threading.Thread(target=run, daemon=True).start()

    # ── 3. Start authentication session ──────────────────────────────────────

    def start(self):
        self.camera = cv2.VideoCapture(0)
        if not self.camera.isOpened():
            QMessageBox.critical(self, "Error", "Cannot open camera.")
            return

        self.running = True
        self.btn_start.setEnabled(False)
        self.btn_stop.setEnabled(True)

        threading.Thread(target=self._camera_loop, daemon=True).start()
        self.sig.new_log.emit("Session started.", "info")

    # ── 4. Stop authentication session ───────────────────────────────────────

    def stop(self):
        self.running = False
        time.sleep(0.3)
        if self.camera:
            self.camera.release()
            self.camera = None

        self.btn_start.setEnabled(True)
        self.btn_stop.setEnabled(False)
        self.sig.new_log.emit("Session stopped.", "info")

    # ── 5. Camera loop (runs in background thread) ────────────────────────────
    def _camera_loop(self):
            count = 0
            faces = []
            label = ""
            color = (180, 180, 180)

            while self.running:
                ok, frame = self.camera.read()
                if not ok:
                    break
                count += 1

                # Detect faces every 3 frames
                if count % 3 == 0:
                    small = cv2.resize(frame, (0, 0), fx=0.75, fy=0.75)
                    scale = 1 / 0.75
                    faces = [
                        (int(x*scale), int(y*scale), int(w*scale), int(h*scale))
                        for x, y, w, h in self.detector.detect(small)
                    ]
                    if not faces:
                        label = "No face detected"
                        color = (180, 180, 180)

                # Try to recognize every 60 frames
                if count % 60 == 0 and faces:
                    x, y, w, h = faces[0]

                    # Skip if face hasn't moved
                    pos = (x//20, y//20)
                    if pos == self._last_face_pos:
                        self.detector.draw(frame, faces, label, color)
                        self.sig.new_frame.emit(frame)
                        continue
                    self._last_face_pos = pos

                    roi         = frame[y:y+h, x:x+w]
                    face_result = self.recognizer.predict(roi)

                    if face_result["status"] == "unknown":
                        label = "Unknown"
                        color = (242, 153, 0)
                        self.sig.new_log.emit("Unknown face — intrusion.", "unknown")
                        self.sig.new_log.emit("__result__|unknown|Unknown|0%", "__result__")

                    elif face_result["status"] == "unauthorized":
                        label  = f"{face_result['name']} — REFUSED"
                        color  = (217, 48, 37)
                        result = decide(face_result)
                        self.db.log_auth(result["name"], result["decision"],
                            result["face_score"], result["fp_score"],
                            result["score"], result["detail"])
                        self.sig.new_log.emit(result["detail"], "unauthorized")
                        self.sig.new_log.emit(
                            f"__result__|unauthorized|{face_result['name']}|{face_result['confidence']*100:.0f}%",
                            "__result__")

                    else:
                        label = f"{face_result['name']} — scan fingerprint"
                        color = (30, 142, 62)
                        self.sig.new_log.emit(
                            f"Face recognized: {face_result['name']} — please upload fingerprint",
                            "info")
                        self.sig.ask_fingerprint.emit(face_result)

                        self._fp_event.clear()
                        self._fp_event.wait(timeout=30)

                        result = decide(face_result, fingerprint_result=self._fp_result)
                        self._fp_result = None

                        color_map = {
                            "authorized":   (30, 142, 62),
                            "partial":      (227, 116, 0),
                            "unauthorized": (217, 48, 37),
                            "unknown":      (242, 153, 0),
                        }
                        color = color_map.get(result["decision"], (150, 150, 150))
                        label = f"{result['name']}  {result['score']*100:.0f}%"

                        self.db.log_auth(result["name"], result["decision"],
                            result["face_score"], result["fp_score"],
                            result["score"], result["detail"])
                        self.sig.new_log.emit(result["detail"], result["decision"])
                        self.sig.new_log.emit(
                            f"__result__|{result['decision']}|{result['name']}|{result['score']*100:.0f}%",
                            "__result__")

                self.detector.draw(frame, faces, label, color)
                self.sig.new_frame.emit(frame)

            self.running = False
    # ── 6. Check DB integrity ─────────────────────────────────────────────────

    def check_integrity(self):
        for pid, name, status in self.db.list_persons():
            tmpl = self.db.load_template(pid, "face")
            if not tmpl:
                self.sig.new_log.emit(f"{name} — no template", "unknown")
                continue
            ok, msg, _ = self.twm.verify(
                tmpl["vector"], tmpl["wm_payload"], tmpl["checksum"])
            self.sig.new_log.emit(f"{name}: {msg}", "authorized" if ok else "error")

    # ── UI updaters (called by signals) ──────────────────────────────────────

    def _display_frame(self, frame):
        """Convert camera frame and show it in the video label."""
        rgb   = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w  = rgb.shape[:2]
        img   = QImage(rgb.data, w, h, w * 3, QImage.Format_RGB888)
        pix   = QPixmap.fromImage(img).scaled(640, 430, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self.video_label.setPixmap(pix)

    def _handle_log(self, message, decision):
        """Route log messages: update result bar OR append to log box."""

        # Update the result bar below the camera
        if decision == "__result__":
            parts = message.split("|")
            if len(parts) >= 4:
                dec, name, score = parts[1], parts[2], parts[3]
                fg, bg, label = DECISION_COLORS.get(dec, (TEXT2, BORDER, "—"))
                self.result_chip.setText(label)
                self.result_chip.setStyleSheet(
                    f"background:{bg}; color:{fg}; border-radius:12px;"
                    f" padding:4px 10px; font-size:12px; font-weight:500; border:none;")
                self.result_name.setText(name)
                self.result_name.setStyleSheet(
                    f"color:{fg}; font-size:15px; background:transparent; border:none;")
                self.result_score.setText(f"score {score}")
            return

        # Append colored line to log box
        fg = DECISION_COLORS.get(decision, (TEXT2, BORDER, ""))[0]
        ts = datetime.now().strftime("%H:%M:%S")
        self.log_box.append(
            f'<span style="color:{TEXT2}">[{ts}]</span>'
            f' <span style="color:{fg}">{message}</span>')
        self.log_box.verticalScrollBar().setValue(
            self.log_box.verticalScrollBar().maximum())
        
    # ── Fingerprint popup ─────────────────────────────────────────────────────
    def enroll_fingerprint(self):
        """Register a fingerprint from an image file."""
        name, ok = QInputDialog.getText(self, "Enroll fingerprint", "Name:")
        if not ok or not name.strip():
            return
        status, ok = QInputDialog.getItem(
            self, "Status", "Status:", ["authorized", "unauthorized"], 0, False)
        if not ok:
            return
        path, _ = QFileDialog.getOpenFileName(
            self, "Select fingerprint image", "", "Images (*.bmp *.png *.jpg)")
        if not path:
            return
        ok2, msg = self.fp_recognizer.enroll(name.strip(), status, path)
        self.sig.new_log.emit(msg, "info" if ok2 else "error")

    def _ask_fingerprint_popup(self, face_result):
        """Called automatically after face is recognized — asks for fingerprint image."""
        path, _ = QFileDialog.getOpenFileName(
            self,
            f"Face: {face_result['name']} — Select your fingerprint image",
            "",
            "Images (*.bmp *.png *.jpg)")
        if path:
            self._fp_result = self.fp_recognizer.predict(path)
            self.sig.new_log.emit(
                f"Fingerprint score: {self._fp_result['confidence']*100:.0f}%", "info")
        else:
            self._fp_result = None
            self.sig.new_log.emit("Fingerprint cancelled.", "info")
        self._fp_event.set()  # unblock the camera loop
        
    # ── Delete person (RGPD) ─────────────────────────────────────────────────
    def delete_person(self):
        import shutil
        persons = self.db.list_persons()
        if not persons:
            QMessageBox.information(self, "Delete", "No persons in database.")
            return
        names = [p[1] for p in persons]
        name, ok = QInputDialog.getItem(
            self, "Delete person", "Choose:", names, 0, False)
        if not ok:
            return
        confirm = QMessageBox.question(
            self, "Confirm",
            f"Permanently delete {name} and all their data?",
            QMessageBox.Yes | QMessageBox.No)
        if confirm != QMessageBox.Yes:
            return

        # 1. Delete from SQLite (person + templates)
        self.db.delete_person(name)

        # 2. Delete from fingerprint pickle
        if name in self.fp_recognizer.db:
            del self.fp_recognizer.db[name]
            self.fp_recognizer._save()

        # 3. Delete face images from disk
        import os
        for status in ("authorized", "unauthorized"):
            folder = os.path.join("data", "known_faces", status, name)
            if os.path.isdir(folder):
                shutil.rmtree(folder)

        self.sig.new_log.emit(f"{name} fully deleted (GDPR).", "info")
    # ── Cleanup ───────────────────────────────────────────────────────────────

    def closeEvent(self, event):
        self.running = False
        time.sleep(0.3)
        if self.camera:
            self.camera.release()
        self.db.close()
        event.accept()