"""
core/detection.py
Détection de visage avec Haar Cascade.
"""

import cv2


class FaceDetector:
    def __init__(self):
        cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        self.detector = cv2.CascadeClassifier(cascade_path)

    def detect(self, frame):
        """Retourne liste de (x, y, w, h). Vide si aucun visage."""
        gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray  = cv2.equalizeHist(gray)
        faces = self.detector.detectMultiScale(
            gray, scaleFactor=1.2, minNeighbors=6, minSize=(80, 80))
        return list(faces) if len(faces) > 0 else []

    def draw(self, frame, faces, label="", color=(0, 255, 0)):
        """Dessine les rectangles + label sur l'image."""
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            if label:
                cv2.rectangle(frame, (x, y - 26), (x + w, y), (0, 0, 0), -1)
                cv2.putText(frame, label, (x + 4, y - 7),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 1)
        return frame
