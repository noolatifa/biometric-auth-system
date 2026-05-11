"""
core/recognition.py
Extraction HOG+LBP et classification SVM.
"""

from math import dist
import os
import cv2
import pickle
import numpy as np
from skimage.feature import hog, local_binary_pattern
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV


FACE_SIZE  = (128, 128)
MODEL_PATH = "models/svm_face.pkl"
THRESHOLD  = 0.55  # Score minimum pour valider la reconnaissance


def extract_features(bgr_face):
    """Retourne un vecteur HOG + LBP normalisé."""
    face = cv2.resize(bgr_face, FACE_SIZE)
    gray = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8)) #clahe pour améliorer les contrastes locaux (utile pour les visages sombres ou éclairés de travers)
    gray  = clahe.apply(gray)

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
      
      # adding a secondary classier LBPH cause svm is failing due to low nb of images and high intra-class variance (different lighting, angles, etc.)
        self.lbph         = cv2.face.LBPHFaceRecognizer_create()
        self.lbph_trained = False
        self.LBPH_PATH    = "models/lbph_face.yml"
        if os.path.exists(self.LBPH_PATH):
            self.lbph.read(self.LBPH_PATH)
            self.lbph_trained = True
        
      
      
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
        
        # Entraînement du classifieur LBPH
        lbph_faces  = []
        lbph_labels = []
        for status in ("authorized", "unauthorized"):
            status_dir = os.path.join(data_dir, status)
            if not os.path.isdir(status_dir):
                continue
            for name in os.listdir(status_dir):
                person_dir = os.path.join(status_dir, name)
                if not os.path.isdir(person_dir):
                    continue
                for f in os.listdir(person_dir):
                    if not f.lower().endswith((".jpg", ".png")):
                        continue
                    img = cv2.imread(os.path.join(person_dir, f), cv2.IMREAD_GRAYSCALE)
                    if img is not None:
                        lbph_faces.append(cv2.resize(img, (128, 128)))
                        lbph_labels.append(self.label_encoder.transform([name])[0])

        if lbph_faces:
            self.lbph.train(lbph_faces, np.array(lbph_labels))
            self.lbph.save(self.LBPH_PATH)
            self.lbph_trained = True
            print("LBPH model trained.")
        
        
        print(f"Modèle entraîné : {len(X)} images, {len(self.label_encoder.classes_)} personnes.")
        self.save()

    def predict(self, face_bgr):
        """
        Retourne dict :
          name       : str
          status     : "authorized" | "unauthorized" | "unknown"
          confidence : float  (0.0 – 1.0)

        Pipeline : LBPH (rapide) → SVM (confirme) → décision
        """
        if not self.lbph_trained or self.pipeline is None:
            return {"name": "Inconnu", "status": "unknown", "confidence": 0.0}

        # Step 1 — LBPH : rapide, s'arrête si inconnu
        gray          = cv2.cvtColor(cv2.resize(face_bgr, (128, 128)), cv2.COLOR_BGR2GRAY)
        lbph_id, dist = self.lbph.predict(gray)

        if dist > 80:
            return {"name": "Inconnu", "status": "unknown", "confidence": 0.0}

        lbph_conf = max(0.0, 1.0 - dist / 100)

        # Step 2 — SVM : confirme l'identité
        feat  = extract_features(face_bgr)
        proba = self.pipeline.predict_proba(feat.reshape(1, -1))[0]
        idx   = int(np.argmax(proba))
        conf  = float(proba[idx])

        # Les deux doivent être d'accord
        if idx != lbph_id:
            return {"name": "Inconnu", "status": "unknown", "confidence": 0.0}

        # Confiance finale = moyenne des deux
        final_conf = (conf + lbph_conf) / 2

        if final_conf < THRESHOLD:
            return {"name": "Inconnu", "status": "unknown", "confidence": final_conf}

        name   = self.label_encoder.inverse_transform([idx])[0]
        status = self.statuses.get(name, "unknown")
        return {"name": name, "status": status, "confidence": final_conf}
        
        


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
