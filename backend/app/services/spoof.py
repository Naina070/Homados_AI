from __future__ import annotations

import logging
from typing import Any

import numpy as np

from app.config import get_settings
from app.services.spectral import spoof_prior_from_artifacts

logger = logging.getLogger("homados.spoof")


class SpoofDetector:
    """AASIST integration boundary.

    A raw ``.pth`` is not enough to safely claim AASIST inference: it also
    needs the exact official architecture and checkpoint format. Until that
    verified inference adapter is supplied, this service deliberately uses
    the explainable spectral prior instead of reporting a misleading model
    result.
    """

    def __init__(self) -> None:
        self.settings = get_settings()
        self.available = False
        self.reason = "spectral_prior"
        if self.settings.enable_aasist:
            self._try_load()

    def _try_load(self) -> None:
        try:
            ckpt = self.settings.aasist_dir / "aasist.pth"
            self.reason = (
                "aasist_checkpoint_needs_official_inference_adapter"
                if ckpt.exists()
                else "aasist_checkpoint_not_downloaded"
            )
            logger.info("AASIST %s — using spectral prior.", self.reason)
        except Exception as exc:  # pragma: no cover
            self.available = False
            self.reason = f"aasist_unavailable:{type(exc).__name__}"
            logger.warning("AASIST not loaded: %s", exc)

    def score(self, y: np.ndarray, sr: int, artifacts: list[dict[str, Any]]) -> dict[str, Any]:
        prior = spoof_prior_from_artifacts(artifacts)
        if not self.available:
            return {
                "available": False,
                "backend": "librosa_spectral_prior",
                "source": "clovaai/aasist (checkpoint pending)",
                "spoof_probability": round(prior, 4),
                "bonafide_probability": round(1.0 - prior, 4),
                "reason": self.reason,
            }

_detector: SpoofDetector | None = None


def get_spoof_detector() -> SpoofDetector:
    global _detector
    if _detector is None:
        _detector = SpoofDetector()
    return _detector
