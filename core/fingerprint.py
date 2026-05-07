"""
core/fingerprint.py
-------------------
Fingerprint recognition using REAL minutiae extraction.
Method : skimage morphology → minutiae points (bifurcations + endings) + matching

Workflow:
    ENROLL  → load image → extract minutiae → save to disk
    PREDICT → load image → extract minutiae → compare with enrolled
"""

import cv2
import pickle
import os
import numpy as np
from skimage.morphology import skeletonize
from skimage.filters import threshold_otsu

MODEL_PATH = "models/fingerprint_db.pkl"
THRESHOLD  = 0.30   # minimum score to accept a match


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
            return False, f"Not enough minutiae detected ({len(minutiae)}). Try a clearer image."

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
            1. Enhance contrast
            2. Binarize (Otsu threshold)
            3. Skeletonize (thin ridges to 1 pixel)
            4. Find minutiae = pixels with exactly 1 neighbor (ending)
                                              or exactly 3 neighbors (bifurcation)
        """
        # Step 1 - enhance contrast
        gray = cv2.equalizeHist(gray)
        gray = cv2.GaussianBlur(gray, (3, 3), 0)

        # Step 2 - binarize
        thresh = threshold_otsu(gray)
        binary = gray > thresh   # True = ridge, False = valley

        # Step 3 - skeletonize
        skeleton = skeletonize(binary)

        # Step 4 - find minutiae points
        minutiae = []
        rows, cols = skeleton.shape

        for y in range(1, rows - 1):
            for x in range(1, cols - 1):
                if not skeleton[y, x]:
                    continue   # not a ridge pixel

                # Count neighbors that are also ridge pixels
                neighbors = int(skeleton[y-1, x-1]) + int(skeleton[y-1, x]) + \
                            int(skeleton[y-1, x+1]) + int(skeleton[y,   x-1]) + \
                            int(skeleton[y,   x+1]) + int(skeleton[y+1, x-1]) + \
                            int(skeleton[y+1, x])   + int(skeleton[y+1, x+1])

                if neighbors == 1:
                    minutiae.append((x, y, "ending"))         # ridge ending
                elif neighbors == 3:
                    minutiae.append((x, y, "bifurcation"))    # ridge bifurcation

        return minutiae

    # ── Matching ──────────────────────────────────────────────────────────────

    def _match(self, minutiae_a, minutiae_b):
        """
        Compares two sets of minutiae points.
        Score = proportion of points from A that have a close match in B.
        """
        if not minutiae_a or not minutiae_b:
            return 0.0

        DISTANCE_THRESHOLD = 20   # pixels — two points are "matching" if closer than this

        matched = 0
        for (x1, y1, type1) in minutiae_a:
            for (x2, y2, type2) in minutiae_b:
                if type1 != type2:
                    continue
                dist = np.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)
                if dist < DISTANCE_THRESHOLD:
                    matched += 1
                    break   # count each point only once

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