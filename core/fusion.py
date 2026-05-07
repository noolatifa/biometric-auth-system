"""
core/fusion.py
Fusion des scores biométriques (règle de la somme pondérée).

Actuellement : visage uniquement.
Empreinte    : à brancher plus tard dans authenticate().
"""

# Poids de chaque modalité (doivent sommer à 1.0)
WEIGHT_FACE        = 1.0   # → 0.6 quand l'empreinte sera ajoutée
WEIGHT_FINGERPRINT = 0.0   # → 0.4 quand l'empreinte sera ajoutée

# Seuil de décision final
FUSION_THRESHOLD = 0.80


def fuse(face_score: float, fingerprint_score: float = 0.0) -> float:
    """
    Retourne le score fusionné entre 0.0 et 1.0.
    """
    return WEIGHT_FACE * face_score + WEIGHT_FINGERPRINT * fingerprint_score


def decide(face_result: dict, fingerprint_result: dict = None) -> dict:
    """
    Prend les résultats bruts de chaque modalité et retourne la décision finale.

    face_result        : {"name", "status", "confidence"}  (depuis FaceRecognizer)
    fingerprint_result : même structure — None si non disponible

    Retourne :
      decision : "authorized" | "unauthorized" | "unknown"
      score    : float (score fusionné)
      detail   : str   (explication lisible)
    """
    face_score = face_result.get("confidence", 0.0)
    fp_score   = fingerprint_result.get("confidence", 0.0) if fingerprint_result else 0.0

    fused = fuse(face_score, fp_score)

    # ── Règles de décision ────────────────────────────────────────────────
    face_ok = face_result.get("status") == "authorized" and face_score >= FUSION_THRESHOLD
    # fp_ok = fingerprint_result ... (à compléter)

    if face_ok:
        decision = "authorized"
        detail   = f"Accès accordé — visage {face_score*100:.0f}%"
    elif face_result.get("status") == "unauthorized":
        decision = "unauthorized"
        detail   = f"Personne non autorisée — visage {face_score*100:.0f}%"
    else:
        decision = "unknown"
        detail   = f"Identité inconnue — score {fused*100:.0f}%"

    return {
        "decision":   decision,
        "score":      fused,
        "face_score": face_score,
        "fp_score":   fp_score,
        "name":       face_result.get("name", "Inconnu"),
        "detail":     detail,
    }
