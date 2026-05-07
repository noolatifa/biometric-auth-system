"""
core/recognition.py
Extraction HOG+LBP et classification SVM.
"""

import os
import cv2
import pickle
import numpy as np
from skimage.feature import hog, local_binary_pattern
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.pipeline import Pipeline

FACE_SIZE  = (128, 128)
MODEL_PATH = "models/svm_face.pkl"
THRESHOLD  = 0.80  # Score minimum pour valider la reconnaissance


def extract_features(bgr_face):
    """Retourne un vecteur HOG + LBP normalisé."""
    face = cv2.resize(bgr_face, FACE_SIZE)
    gray = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)

    # HOG : capture la forme et les contours
    hog_vec = hog(gray, orientations=9, pixels_per_cell=(8, 8),
                  cells_per_block=(2, 2), block_norm="L2-Hys",
                  visualize=False, feature_vector=True)

    # LBP : capture la texture locale
    lbp      = local_binary_pattern(gray, P=8, R=1, method="uniform")
    hist, _  = np.histogram(lbp.ravel(), bins=np.arange(0, 12), range=(0, 10))
    lbp_vec  = hist.astype(float) / (hist.sum() + 1e-7)

    return np.concatenate([hog_vec, lbp_vec])


class FaceRecognizer:
    def __init__(self):
        self.pipeline      = None
        self.label_encoder = LabelEncoder()
        self.statuses      = {}  # {name -> "authorized" | "unauthorized"}
        if os.path.exists(MODEL_PATH):
            self.load()

    def train(self, data_dir="data/known_faces"):
        """
        Parcourt data/known_faces/{authorized,unauthorized}/{nom}/
        et entraîne le SVM.
        """
        features, labels = [], []
        self.statuses     = {}

        for status in ("authorized", "unauthorized"):
            status_dir = os.path.join(data_dir, status)
            if not os.path.isdir(status_dir):
                continue
            for name in os.listdir(status_dir):
                person_dir = os.path.join(status_dir, name)
                if not os.path.isdir(person_dir):
                    continue
                self.statuses[name] = status
                count = 0
                for f in os.listdir(person_dir):
                    if not f.lower().endswith((".jpg", ".png", ".jpeg")):
                        continue
                    img = cv2.imread(os.path.join(person_dir, f))
                    if img is None:
                        continue
                    features.append(extract_features(img))
                    labels.append(name)
                    count += 1
                print(f"  ✓ {name} ({status}) — {count} images")

        if len(features) < 2:
            raise ValueError("Pas assez d'images. Enrôlez d'abord des personnes.")

        X = np.array(features)
        y = self.label_encoder.fit_transform(labels)

        self.pipeline = Pipeline([
            ("scaler", StandardScaler()),
            ("svm",    SVC(kernel="rbf", C=10, gamma="scale",
                           probability=True, random_state=42)),
        ])
        self.pipeline.fit(X, y)
        self.save()
        print(f"Modèle entraîné : {len(X)} images, {len(self.label_encoder.classes_)} personnes.")

    def predict(self, face_bgr):
        """
        Retourne dict :
          name       : str
          status     : "authorized" | "unauthorized" | "unknown"
          confidence : float  (0.0 – 1.0)
        """
        if self.pipeline is None:
            return {"name": "Inconnu", "status": "unknown", "confidence": 0.0}

        feat  = extract_features(face_bgr)
        proba = self.pipeline.predict_proba(feat.reshape(1, -1))[0]
        idx   = int(np.argmax(proba))
        conf  = float(proba[idx])

        if conf < THRESHOLD:
            return {"name": "Inconnu", "status": "unknown", "confidence": conf}

        name   = self.label_encoder.inverse_transform([idx])[0]
        status = self.statuses.get(name, "unknown")
        return {"name": name, "status": status, "confidence": conf}

    def save(self):
        os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
        with open(MODEL_PATH, "wb") as f:
            pickle.dump({
                "pipeline":      self.pipeline,
                "label_encoder": self.label_encoder,
                "statuses":      self.statuses,
            }, f)

    def load(self):
        with open(MODEL_PATH, "rb") as f:
            d = pickle.load(f)
        self.pipeline      = d["pipeline"]
        self.label_encoder = d["label_encoder"]
        self.statuses      = d["statuses"]
        print(f"Modèle chargé : {len(self.label_encoder.classes_)} personnes.")
