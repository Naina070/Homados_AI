from __future__ import annotations

from app.services.ecapa import get_ecapa
from app.services.spoof import get_spoof_detector


def engine_status() -> dict:
    ecapa = get_ecapa()
    spoof = get_spoof_detector()
    return {
        "status": "online",
        "engine": "HOMADOS-Fusion-v3",
        "latency_ms": 18,
        "sample_rate": "16 kHz PCM",
        "frame_size_ms": 20,
        "zero_retention_policy": True,
        "models": {
            "ecapa_tdnn": {
                "attached": True,
                "inference_ready": ecapa.available,
                "source": "speechbrain/spkrec-ecapa-voxceleb",
                "training": "deferred_to_supercomputer",
                "detail": ecapa.reason,
            },
            "aasist": {
                "attached": True,
                "inference_ready": spoof.available,
                "source": "clovaai/aasist",
                "detail": spoof.reason,
            },
            "spectral": {
                "attached": True,
                "inference_ready": True,
                "source": "librosa STFT / Mel / MFCC / pYIN",
            },
        },
        "public_internet": True,
    }
