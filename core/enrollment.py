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

N_PHOTOS   = 60
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

        if self.db.get_person_id(name) is not None:
            return False, f"{name} is already enrolled. Delete first to re-enroll."
            

        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            return False, "Impossible d'ouvrir la webcam."

       

        vectors  = []
        save_dir = os.path.join(SAVE_DIR, status, name)
        os.makedirs(save_dir, exist_ok=True)

        try:
            last_capture = time.time()

            while len(vectors) < N_PHOTOS:
                ret, frame = cap.read()
                if not ret:
                    break

                faces   = self.detector.detect(frame)
                display = frame.copy()
                cv2.putText(display,
                    f"Captures : {len(vectors)}/{N_PHOTOS} — move slightly",
                    (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                self.detector.draw(display, faces)
                cv2.imshow(f"Enrollment — {name}", display)

                key = cv2.waitKey(1)
                if key in (ord("q"), 27):
                    return False, "Enrôlement annulé."

                # Auto-capture every 0.5 sec if face detected
                if faces and (time.time() - last_capture) > 0.5:
                    x, y, w, h = faces[0]
                    roi = frame[y:y+h, x:x+w]

                    # 1 capture → 4 vecteurs (variations lumière + flip)
                    vectors.append(extract_features(roi))
                    vectors.append(extract_features(cv2.flip(roi, 1)))
                    vectors.append(extract_features(cv2.convertScaleAbs(roi, alpha=1.3, beta=30)))
                    vectors.append(extract_features(cv2.convertScaleAbs(roi, alpha=0.7, beta=-20)))

                    n = len(vectors) // 4
                    for suffix, img in [("orig", roi),
                                        ("flip",   cv2.flip(roi, 1)),
                                        ("bright", cv2.convertScaleAbs(roi, alpha=1.3, beta=30)),
                                        ("dark",   cv2.convertScaleAbs(roi, alpha=0.7, beta=-20))]:
                        cv2.imwrite(os.path.join(save_dir, f"{n:03d}_{suffix}.jpg"), img)                    
                    last_capture = time.time()

        finally:
            cap.release()
            cv2.destroyAllWindows()
            print(f"Vectors captured: {len(vectors)}")  # ← ajoute ici


        if len(vectors) < 3:
            return False, "Pas assez de captures (minimum 3)."

        # Gabarit = moyenne des vecteurs capturés
        mean_vec  = np.mean(vectors, axis=0)
        
        is_dup, dup_name = self._is_duplicate_face(mean_vec)
        if is_dup:
            return False, f"This face is already enrolled as '{dup_name}'."


        person_id = self.db.add_person(name, status)

       
        payload  = self.twm.build_payload(person_id, name, camera_id)
        wm_vec   = self.twm.embed(mean_vec, payload)
        checksum = self.twm.compute_checksum(wm_vec)

        self.db.store_template(person_id, "face", wm_vec, checksum, payload)
        return True, f"{name} enrôlé ({len(mean_vec)} dims, checksum: {checksum[:12]}…)"
     
     
     
    def _is_duplicate_face(self, new_vector):
        """Returns True if a very similar face template already exists in DB."""
        from numpy.linalg import norm
        persons = self.db.list_persons()
        for pid, name, status in persons:
            tmpl = self.db.load_template(pid, "face")
            if tmpl is None:
                continue
            # cosine similarity between the two vectors
            a = new_vector
            b = tmpl["vector"]
            similarity = float(np.dot(a, b) / (norm(a) * norm(b) + 1e-8))
            if similarity > 0.92:   # threshold — same face
                return True, name
        return False, None