"""
core/fusion.py
Fusion des scores biométriques (règle de la somme pondérée).
"""

WEIGHT_FACE        = 0.6
WEIGHT_FINGERPRINT = 0.4
THRESHOLD          = 0.65


def fuse(face_score, fingerprint_score=0.0):
    return WEIGHT_FACE * face_score + WEIGHT_FINGERPRINT * fingerprint_score


def decide(face_result, fingerprint_result=None):
    face_score = face_result.get("confidence", 0.0)
    fp_score   = fingerprint_result.get("confidence", 0.0) if fingerprint_result else 0.0
    fused      = fuse(face_score, fp_score)

    face_ok       = face_result.get("status") == "authorized" and face_score >= THRESHOLD
    fp_ok         = fingerprint_result is not None and \
                    fingerprint_result.get("status") == "authorized" and fp_score >= THRESHOLD
    fp_configured = fingerprint_result is not None

    if face_result.get("status") == "unauthorized":
        decision = "unauthorized"
        detail   = f"Accès refusé — {face_result.get('name')} non autorisé"

    elif face_ok and fp_configured and fp_ok:
        decision = "authorized"
        detail   = f"Accès accordé — {face_result.get('name')} (visage {face_score*100:.0f}% · empreinte {fp_score*100:.0f}%)"

    elif face_ok and fp_configured and not fp_ok:
        decision = "partial"
        detail   = f"Alerte — visage OK ({face_result.get('name')}) mais empreinte non confirmée"

    elif face_ok and not fp_configured:
        decision = "authorized"
        detail   = f"Accès accordé — visage {face_score*100:.0f}% (pas d'empreinte configurée)"

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