"""
core/enrollment.py
Enrôlement : capture webcam → extraction HOG+LBP → tatouage → BDD.
"""

import cv2
import os
import time
import numpy as np

from core.detection import FaceDetector
from core.recognition import extract_features
from core.template_watermark import TemplateWatermarker

N_PHOTOS   = 10
SAVE_DIR   = "data/known_faces"


class EnrollmentPipeline:
    def __init__(self, db):
        self.db       = db
        self.detector = FaceDetector()
        self.twm      = TemplateWatermarker()

    def enroll(self, name, status, camera_id="CAM-001"):
        """
        Ouvre la webcam.
        Appuyer ESPACE pour capturer, Q pour annuler.
        Stocke le gabarit moyen tatoué en BDD.
        """
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            return False, "Impossible d'ouvrir la webcam."

        vectors  = []
        save_dir = os.path.join(SAVE_DIR, status, name)
        os.makedirs(save_dir, exist_ok=True)

        try:
            while len(vectors) < N_PHOTOS:
                ret, frame = cap.read()
                if not ret:
                    break

                faces   = self.detector.detect(frame)
                display = frame.copy()
                cv2.putText(display,
                    f"Captures : {len(vectors)}/{N_PHOTOS}   ESPACE=photo  Q=quitter",
                    (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                self.detector.draw(display, faces)
                cv2.imshow(f"Enrôlement — {name}", display)

                key = cv2.waitKey(1)
                if key in (ord("q"), 27):
                    return False, "Enrôlement annulé."
                if key == ord(" ") and faces:
                    x, y, w, h = faces[0]
                    roi = frame[y:y+h, x:x+w]
                    vectors.append(extract_features(roi))
                    cv2.imwrite(os.path.join(save_dir, f"{len(vectors):03d}.jpg"), roi)
                    time.sleep(0.4)
        finally:
            cap.release()
            cv2.destroyAllWindows()

        if len(vectors) < 3:
            return False, "Pas assez de captures (minimum 3)."

        # Gabarit = moyenne des vecteurs capturés
        mean_vec  = np.mean(vectors, axis=0)
        person_id = self.db.add_person(name, status)

        payload  = self.twm.build_payload(person_id, name, camera_id)
        wm_vec   = self.twm.embed(mean_vec, payload)
        checksum = self.twm.compute_checksum(wm_vec)

        self.db.store_template(person_id, "face", wm_vec, checksum, payload)
        return True, f"{name} enrôlé ({len(mean_vec)} dims, checksum: {checksum[:12]}…)"
