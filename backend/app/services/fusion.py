from __future__ import annotations

from typing import Any

from app.config import get_settings
from app.services.ecapa import cosine_similarity


def fuse_scores(
    spoof_probability: float,
    identity_cosine: float | None,
    spectral_artifact_mean: float,
    enrolled: bool,
) -> dict[str, Any]:
    """Stage 3–4 of the deck: fuse spoof + identity + spectral into one risk."""
    settings = get_settings()
    identity_risk = 0.35
    if enrolled and identity_cosine is not None:
        # Cosine 1.0 = same speaker. Drift / clone often lands well below 0.7.
        identity_risk = float(max(0.0, min(1.0, (0.85 - identity_cosine) / 0.85)))

    risk = (
        settings.risk_spoof_weight * spoof_probability
        + settings.risk_identity_weight * identity_risk
        + settings.risk_spectral_weight * spectral_artifact_mean
    )
    risk_pct = round(100.0 * max(0.0, min(1.0, risk)), 1)

    if risk_pct >= settings.alert_hard_threshold:
        level = "critical"
        action = "callback"
        label = "High risk — likely synthetic or cloned voice"
    elif risk_pct >= settings.alert_soft_threshold:
        level = "elevated"
        action = "otp"
        label = "Elevated risk — verify identity before proceeding"
    else:
        level = "clear"
        action = "none"
        label = "Consistent with a live human speaker"

    return {
        "risk_score_percentage": risk_pct,
        "risk_level": level,
        "risk_label": label,
        "recommended_action": action,
        "identity_risk": round(identity_risk, 4),
        "spoof_probability": round(spoof_probability, 4),
        "spectral_prior": round(spectral_artifact_mean, 4),
        "identity_cosine": None if identity_cosine is None else round(float(identity_cosine), 4),
        "enrolled_speaker_present": enrolled,
        "weights": {
            "spoof": settings.risk_spoof_weight,
            "identity": settings.risk_identity_weight,
            "spectral": settings.risk_spectral_weight,
        },
    }


def identity_from_embeddings(live: list[float], enrolled: list[float] | None) -> tuple[float | None, bool]:
    if not enrolled:
        return None, False
    return cosine_similarity(live, enrolled), True
