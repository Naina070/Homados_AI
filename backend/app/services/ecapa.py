from __future__ import annotations

import logging
from typing import Any

import numpy as np

from app.config import get_settings

logger = logging.getLogger("homados.ecapa")


class EcapaEncoder:
    """SpeechBrain ECAPA-TDNN (VoxCeleb pretrained). Training is deferred.

    Weights are loaded from `speechbrain/spkrec-ecapa-voxceleb` into
    `models/ecapa_tdnn`. Fine-tuning on a supercomputer is a later step; this
    class only attaches the public checkpoint for inference embeddings.
    """

    def __init__(self) -> None:
        self.settings = get_settings()
        self.model = None
        self.available = False
        self.reason = "not_loaded"
        self.embedding_dim = 192
        if self.settings.enable_ecapa:
            self._try_load()

    def _try_load(self) -> None:
        try:
            import torch
            from speechbrain.inference.speaker import EncoderClassifier

            savedir = str(self.settings.ecapa_dir)
            self.settings.ecapa_dir.mkdir(parents=True, exist_ok=True)
            device = self.settings.torch_device
            if device.startswith("cuda") and not torch.cuda.is_available():
                device = "cpu"
            self.model = EncoderClassifier.from_hparams(
                source=self.settings.ecapa_source,
                savedir=savedir,
                run_opts={"device": device},
            )
            self.available = True
            self.reason = "speechbrain_ecapa_voxceleb"
            logger.info("ECAPA-TDNN loaded from %s", self.settings.ecapa_source)
        except Exception as exc:  # pragma: no cover - optional heavy dep
            self.available = False
            self.reason = f"ecapa_unavailable:{type(exc).__name__}: {exc}"
            logger.warning("ECAPA-TDNN not loaded (%s). Spectral fallback remains active.", exc)

    def embed(self, y: np.ndarray, sr: int) -> dict[str, Any]:
        y = np.asarray(y, dtype=np.float32).flatten()
        if self.available and self.model is not None:
            import torch
            import torchaudio

            wav = torch.from_numpy(y).unsqueeze(0)
            if sr != 16000:
                wav = torchaudio.functional.resample(wav, sr, 16000)
            with torch.no_grad():
                emb = self.model.encode_batch(wav).squeeze().cpu().numpy()
            vec = np.asarray(emb, dtype=np.float32).flatten()
            vec = vec / (np.linalg.norm(vec) + 1e-8)
            return {
                "available": True,
                "backend": "speechbrain/spkrec-ecapa-voxceleb",
                "training_status": "pretrained_inference_only",
                "dim": int(vec.size),
                "vector": vec.astype(float).tolist()[:192],
            }

        # Deterministic acoustic fingerprint so the rest of the pipeline works
        # before ML extras / supercomputer fine-tuning are installed.
        vec = _mfcc_fingerprint(y, sr, self.embedding_dim)
        return {
            "available": False,
            "backend": "mfcc_fingerprint_fallback",
            "training_status": "pending_supercomputer",
            "reason": self.reason,
            "dim": int(vec.size),
            "vector": vec.astype(float).tolist(),
        }


def cosine_similarity(a: list[float], b: list[float]) -> float:
    va = np.asarray(a, dtype=np.float32)
    vb = np.asarray(b, dtype=np.float32)
    n = min(va.size, vb.size)
    if n == 0:
        return 0.0
    va, vb = va[:n], vb[:n]
    denom = float(np.linalg.norm(va) * np.linalg.norm(vb) + 1e-8)
    return float(np.clip(np.dot(va, vb) / denom, -1.0, 1.0))


def _mfcc_fingerprint(y: np.ndarray, sr: int, dim: int) -> np.ndarray:
    import librosa

    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)
    stats = np.concatenate([mfcc.mean(axis=1), mfcc.std(axis=1), mfcc.max(axis=1)])
    if stats.size < dim:
        stats = np.pad(stats, (0, dim - stats.size))
    stats = stats[:dim].astype(np.float32)
    return stats / (np.linalg.norm(stats) + 1e-8)


_encoder: EcapaEncoder | None = None


def get_ecapa() -> EcapaEncoder:
    global _encoder
    if _encoder is None:
        _encoder = EcapaEncoder()
    return _encoder
