"""
core/fingerprint.py
-------------------
Fingerprint recognition using REAL minutiae extraction.

Method :
    1. Skeletonize the fingerprint image
    2. Detect minutiae : ridge endings (1 neighbor) and bifurcations (3 neighbors)
    3. Compute orientation angle at each minutia point
    4. Match two sets of minutiae using position + orientation

This respects the project requirement:
    "Reconnaissance empreinte : Extraction de minuties + matching"
"""

import cv2
import pickle
import os
import numpy as np
from skimage.morphology import skeletonize
from skimage.filters import threshold_otsu

MODEL_PATH         = "models/fingerprint_db.pkl"
THRESHOLD          = 0.45   # minimum score to accept a match
DISTANCE_THRESHOLD = 12     # pixels — position must match within this distance
ANGLE_THRESHOLD    = 0.5    # radians (~28°) — orientation must match within this


class FingerprintRecognizer:

    def __init__(self):
        self.db = {}   # { name: { "minutiae": [...], "status": str } }
        if os.path.exists(MODEL_PATH):
            self._load()

    # ── Enroll ────────────────────────────────────────────────────────────────

    def enroll(self, name, status, image_path):
        """
        Register a person's fingerprint from an image file.
        Returns (True, message) or (False, error).
        """
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            return False, f"Image not found: {image_path}"

        minutiae = self._extract_minutiae(img)

        if len(minutiae) < 5:
            return False, f"Not enough minutiae ({len(minutiae)}). Try a clearer image."

        #----------------------------Check duplicate fingerprint-------------------        
        for existing_name, data in self.db.items():
            if existing_name == name:
                continue
            score = self._match(minutiae, data["minutiae"])
            if score >= THRESHOLD:
                return False, f"This fingerprint is already enrolled as '{existing_name}'."

        self.db[name] = {"minutiae": minutiae, "status": status}  
        self._save()
        
        return True, f"Fingerprint enrolled for {name} ({len(minutiae)} minutiae)."
    # ── Predict ───────────────────────────────────────────────────────────────

    def predict(self, image_path):
        """
        Compare a fingerprint image against all enrolled fingerprints.
        Returns dict: { name, status, confidence }
        """
        unknown = {"name": "Unknown", "status": "unknown", "confidence": 0.0}

        if not self.db:
            return unknown

        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            return unknown

        minutiae = self._extract_minutiae(img)
        if len(minutiae) < 5:
            return unknown

        best_name  = "Unknown"
        best_score = 0.0

        for name, data in self.db.items():
            score = self._match(minutiae, data["minutiae"])
            if score > best_score:
                best_score = score
                best_name  = name

        if best_score < THRESHOLD:
            return unknown

        return {
            "name":       best_name,
            "status":     self.db[best_name]["status"],
            "confidence": best_score,
        }

    # ── Minutiae extraction ───────────────────────────────────────────────────

    def _extract_minutiae(self, gray):
        """
        Extracts minutiae points from a fingerprint image.

        Steps:
            1. Enhance contrast (CLAHE + Gaussian blur)
            2. Binarize (Otsu threshold)
            3. Skeletonize (thin ridges to 1 pixel)
            4. Detect minutiae by counting 8-connected neighbors:
                 1 neighbor → ridge ending
                 3 neighbors → ridge bifurcation
            5. Compute orientation angle at each minutia
        """
        # Step 1 - enhance contrast
        gray = cv2.resize(gray, (256, 256))  # ← add this line FIRST
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        gray  = clahe.apply(gray)
        gray  = cv2.GaussianBlur(gray, (3, 3), 0)

        # Step 2 - binarize
        thresh = threshold_otsu(gray)
        binary = gray > thresh

        # Step 3 - skeletonize
        skeleton = skeletonize(binary)

        # Step 4 + 5 - find minutiae + orientation
        minutiae = []
        rows, cols = skeleton.shape

        for y in range(1, rows - 1):
            for x in range(1, cols - 1):
                if not skeleton[y, x]:
                    continue

                neighbors = (
                    int(skeleton[y-1, x-1]) + int(skeleton[y-1, x]) +
                    int(skeleton[y-1, x+1]) + int(skeleton[y,   x-1]) +
                    int(skeleton[y,   x+1]) + int(skeleton[y+1, x-1]) +
                    int(skeleton[y+1, x])   + int(skeleton[y+1, x+1])
                )

                if neighbors == 1:
                    angle = self._compute_angle(skeleton, x, y)
                    minutiae.append((x, y, "ending", angle))
                elif neighbors == 3:
                    angle = self._compute_angle(skeleton, x, y)
                    minutiae.append((x, y, "bifurcation", angle))

        return minutiae

    # ── Orientation ───────────────────────────────────────────────────────────

    @staticmethod
    def _compute_angle(skeleton, x, y):
        """
        Computes the local ridge orientation at a minutia point.
        Uses image gradients in a 7x7 neighborhood.
        """
        region = skeleton[max(0, y-3):y+4, max(0, x-3):x+4].astype(np.float32)
        if region.sum() == 0:
            return 0.0
        gy, gx = np.gradient(region)
        angle  = np.arctan2(float(gy.mean()), float(gx.mean()))
        return angle

    # ── Matching ──────────────────────────────────────────────────────────────

    def _match(self, minutiae_a, minutiae_b):
        """
        Compares two sets of minutiae.
        A match requires BOTH position AND orientation to agree.

        Score = matched / max(|A|, |B|)
        """
        if not minutiae_a or not minutiae_b:
            return 0.0

        matched = 0
        for (x1, y1, type1, angle1) in minutiae_a:
            for (x2, y2, type2, angle2) in minutiae_b:
                if type1 != type2:
                    continue
                dist       = np.sqrt((x1 - x2)**2 + (y1 - y2)**2)
                angle_diff = abs(angle1 - angle2)
                if dist < DISTANCE_THRESHOLD and angle_diff < ANGLE_THRESHOLD:
                    matched += 1
                    break   # each minutia counted once

        score = matched / max(len(minutiae_a), len(minutiae_b))
        return min(score, 1.0)

    # ── Save / Load ───────────────────────────────────────────────────────────

    def _save(self):
        os.makedirs("models", exist_ok=True)
        with open(MODEL_PATH, "wb") as f:
            pickle.dump(self.db, f)

    def _load(self):
        with open(MODEL_PATH, "rb") as f:
            self.db = pickle.load(f)
        print(f"Fingerprint DB loaded: {len(self.db)} persons.")