"""
core/template_watermark.py
Tatouage DCT 1D des gabarits biométriques (vecteurs HOG+LBP).
Objectif : détecter toute modification du gabarit stocké en base.
"""

import numpy as np
import hashlib
from datetime import datetime

ALPHA       = 0.5   # Amplitude du tatouage dans les coefficients DCT
START_COEFF = 50    # Premier coefficient DCT modifié


class TemplateWatermarker:

    def build_payload(self, person_id, name, camera="CAM-001"):
        """Chaîne à cacher dans le gabarit."""
        ts = datetime.now().strftime("%Y%m%d%H%M%S")
        return f"ID:{person_id}|NM:{name[:8]}|CAM:{camera}|TS:{ts}"

    def compute_checksum(self, vector):
        """SHA-256 du vecteur — sert de second mécanisme de vérification."""
        return hashlib.sha256(vector.astype(np.float32).tobytes()).hexdigest()

    def embed(self, feature_vector, payload):
        """
        Cache le payload dans les coefficients DCT du vecteur.
        Retourne le vecteur tatoué.
        """
        vec  = feature_vector.astype(np.float64).copy()
        bits = self._to_bits(payload)
        bits = bits[:len(vec) - START_COEFF]

        dct = self._dct(vec)
        for i, bit in enumerate(bits):
            idx      = START_COEFF + i
            val      = abs(dct[idx])
            dct[idx] = (val + ALPHA) if bit == "1" else -(val + ALPHA)

        return self._idct(dct)

    def verify(self, stored_vector, expected_payload, stored_checksum):
        """
        Vérifie l'intégrité d'un gabarit.
        Retourne (ok: bool, message: str, tampered: bool).
        """
        # Vérification 1 : checksum SHA-256
        checksum_ok = self.compute_checksum(stored_vector) == stored_checksum

        # Vérification 2 : extraction du tatouage DCT
        expected_bits = self._to_bits(expected_payload)
        dct           = self._dct(stored_vector.astype(np.float64))
        extracted     = []
        for i in range(len(expected_bits)):
            idx = START_COEFF + i
            if idx >= len(dct):
                break
            extracted.append("1" if dct[idx] > 0 else "0")

        matches      = sum(a == b for a, b in zip(expected_bits, extracted))
        accuracy     = matches / max(len(expected_bits), 1) * 100
        watermark_ok = accuracy >= 75.0

        tampered = not checksum_ok or not watermark_ok

        if tampered:
            msg = f"⚠ GABARIT ALTÉRÉ (checksum={'OK' if checksum_ok else 'FAIL'}, tatouage={accuracy:.0f}%)"
        else:
            msg = f"✓ Intègre ({accuracy:.0f}%)"

        return not tampered, msg, tampered

    # ── DCT 1D (numpy pur, sans scipy) ──────────────────────────────────────

    @staticmethod
    def _dct(x):
        N = len(x)
        n = np.arange(N)
        k = n.reshape((N, 1))
        M = np.cos(np.pi * k * (2 * n + 1) / (2 * N))
        r = 2 * M @ x
        r[0] *= np.sqrt(1 / (4 * N))
        r[1:] *= np.sqrt(1 / (2 * N))
        return r

    @staticmethod
    def _idct(X):
        N  = len(X)
        X2 = X.copy()
        X2[0] *= np.sqrt(1 / (4 * N))
        X2[1:] *= np.sqrt(1 / (2 * N))
        n = np.arange(N)
        k = n.reshape((N, 1))
        M = np.cos(np.pi * k * (2 * n + 1) / (2 * N))
        return 2 * (M.T @ X2)

    @staticmethod
    def _to_bits(text):
        return list("".join(format(b, "08b") for b in text.encode("utf-8")))
