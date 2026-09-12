from __future__ import annotations

import io
from typing import Any

import numpy as np

from app.services.ecapa import get_ecapa
from app.services.fusion import fuse_scores, identity_from_embeddings
from app.services.spectral import extract_spectral_features
from app.services.spoof import get_spoof_detector

TARGET_SR = 16000


def load_audio(data: bytes, filename: str = "audio.wav") -> tuple[np.ndarray, int]:
    import librosa
    import soundfile as sf

    buffer = io.BytesIO(data)
    try:
        y, sr = sf.read(buffer, always_2d=False)
        y = np.asarray(y, dtype=np.float32)
        if y.ndim > 1:
            y = y.mean(axis=1)
    except Exception:
        buffer.seek(0)
        y, sr = librosa.load(buffer, sr=TARGET_SR, mono=True)
        return np.asarray(y, dtype=np.float32), TARGET_SR

    if sr != TARGET_SR:
        y = librosa.resample(y, orig_sr=sr, target_sr=TARGET_SR)
        sr = TARGET_SR
    peak = np.max(np.abs(y)) + 1e-8
    y = y / peak
    return y, sr


def analyze_audio(
    data: bytes,
    filename: str = "clip.wav",
    enrolled_embedding: list[float] | None = None,
) -> dict[str, Any]:
    y, sr = load_audio(data, filename)
    spectral = extract_spectral_features(y, sr)
    spoof = get_spoof_detector().score(y, sr, spectral["artifacts"])
    ecapa = get_ecapa().embed(y, sr)
    cosine, enrolled = identity_from_embeddings(ecapa["vector"], enrolled_embedding)
    artifact_mean = float(np.mean([a["score"] for a in spectral["artifacts"]]))
    fused = fuse_scores(
        spoof_probability=float(spoof["spoof_probability"]),
        identity_cosine=cosine,
        spectral_artifact_mean=artifact_mean,
        enrolled=enrolled,
    )
    return {
        "filename": filename,
        "engine": "HOMADOS-Fusion-v3",
        "zero_retention": True,
        "spectral": spectral,
        "spoof": spoof,
        "speaker": {
            **{k: v for k, v in ecapa.items() if k != "vector"},
            "embedding_preview": ecapa["vector"][:16],
            "cosine_to_enrolled": fused["identity_cosine"],
        },
        "fusion": fused,
        "artifacts": spectral["artifacts"],
    }
