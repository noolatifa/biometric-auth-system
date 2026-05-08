"""
test_metrics.py
Mesure FAR, FRR, EER sur les images enregistrées.

Utilise les images dans data/known_faces/ en leave-one-out :
  - pour chaque personne, teste chaque image contre le modèle
  - genuine attempt  = bonne personne
  - impostor attempt = mauvaise personne
"""

import os
import cv2
import numpy as np
from core.recognition import FaceRecognizer, extract_features

THRESHOLD_RANGE = np.arange(0.1, 1.0, 0.05)
DATA_DIR        = "data/known_faces"


def load_all_images():
    """
    Returns list of (image, true_name, true_status).
    """
    samples = []
    for status in ("authorized", "unauthorized"):
        status_dir = os.path.join(DATA_DIR, status)
        if not os.path.isdir(status_dir):
            continue
        for name in os.listdir(status_dir):
            person_dir = os.path.join(status_dir, name)
            if not os.path.isdir(person_dir):
                continue
            for f in os.listdir(person_dir):
                if not f.lower().endswith((".jpg", ".png")):
                    continue
                img = cv2.imread(os.path.join(person_dir, f))
                if img is not None:
                    samples.append((img, name, status))
    return samples


def compute_far_frr(recognizer, samples, threshold):
    """
    For a given threshold, compute FAR and FRR.
    """
    genuine_total   = 0
    genuine_reject  = 0   # FRR numerator
    impostor_total  = 0
    impostor_accept = 0   # FAR numerator

    for img, true_name, true_status in samples:
        result = recognizer.predict(img)
        predicted_name = result["name"]
        confidence     = result["confidence"]

        if true_status == "authorized":
            genuine_total += 1
            # Genuine rejected if below threshold or wrong name
            if confidence < threshold or predicted_name != true_name:
                genuine_reject += 1
        else:
            impostor_total += 1
            # Impostor accepted if above threshold
            if confidence >= threshold and predicted_name != "Inconnu":
                impostor_accept += 1

    far = impostor_accept / impostor_total if impostor_total > 0 else 0.0
    frr = genuine_reject  / genuine_total  if genuine_total  > 0 else 0.0
    return far, frr


def find_eer(far_list, frr_list, thresholds):
    """Find the threshold where FAR ≈ FRR."""
    min_diff = float("inf")
    eer_threshold = 0
    eer = 0
    for t, far, frr in zip(thresholds, far_list, frr_list):
        diff = abs(far - frr)
        if diff < min_diff:
            min_diff = diff
            eer_threshold = t
            eer = (far + frr) / 2
    return eer, eer_threshold


if __name__ == "__main__":
    print("Loading model...")
    recognizer = FaceRecognizer()

    print("Loading test images...")
    samples = load_all_images()
    print(f"Total samples: {len(samples)}")

    far_list = []
    frr_list = []

    print("\nComputing FAR/FRR across thresholds...")
    for t in THRESHOLD_RANGE:
        far, frr = compute_far_frr(recognizer, samples, t)
        far_list.append(far)
        frr_list.append(frr)
        print(f"  threshold={t:.2f}  FAR={far*100:.1f}%  FRR={frr*100:.1f}%")

    eer, eer_t = find_eer(far_list, frr_list, THRESHOLD_RANGE)
    print(f"\n── Results ──────────────────────────")
    print(f"EER       : {eer*100:.1f}%")
    print(f"At threshold : {eer_t:.2f}")

    # Current threshold performance
    far_cur, frr_cur = compute_far_frr(recognizer, samples, 0.65)
    print(f"\nAt current threshold (0.65):")
    print(f"  FAR = {far_cur*100:.1f}%")
    print(f"  FRR = {frr_cur*100:.1f}%")