from __future__ import annotations

from typing import Any

import numpy as np


def _safe_mean(x: np.ndarray) -> float:
    if x.size == 0:
        return 0.0
    value = float(np.nanmean(x))
    return 0.0 if np.isnan(value) else value


def _safe_std(x: np.ndarray) -> float:
    if x.size == 0:
        return 0.0
    value = float(np.nanstd(x))
    return 0.0 if np.isnan(value) else value


def extract_spectral_features(y: np.ndarray, sr: int) -> dict[str, Any]:
    """Librosa spectral / prosodic analysis used by the fusion engine.

    These features implement the PPT Stage-2 extractors (mel, MFCC, pitch,
    discontinuity around vocoder cutoff bands) without requiring a GPU.
    """
    import librosa

    y = np.asarray(y, dtype=np.float32).flatten()
    if y.size < sr // 10:
        pad = np.zeros(sr // 4, dtype=np.float32)
        y = np.concatenate([y, pad])

    n_fft = 512
    hop = 160
    stft = np.abs(librosa.stft(y, n_fft=n_fft, hop_length=hop))
    freqs = librosa.fft_frequencies(sr=sr, n_fft=n_fft)

    mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=80, n_fft=n_fft, hop_length=hop)
    log_mel = librosa.power_to_db(mel, ref=np.max)
    mfcc = librosa.feature.mfcc(S=librosa.power_to_db(mel), n_mfcc=20)
    centroid = librosa.feature.spectral_centroid(y=y, sr=sr, n_fft=n_fft, hop_length=hop)
    rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr, n_fft=n_fft, hop_length=hop)
    bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr, n_fft=n_fft, hop_length=hop)
    contrast = librosa.feature.spectral_contrast(y=y, sr=sr, n_fft=n_fft, hop_length=hop)
    zcr = librosa.feature.zero_crossing_rate(y, hop_length=hop)
    rms = librosa.feature.rms(y=y, hop_length=hop)

    flux = np.sqrt(np.mean(np.diff(stft, axis=1) ** 2, axis=0)) if stft.shape[1] > 1 else np.array([0.0])

    band_low = (freqs >= 300) & (freqs < 1000)
    band_mid = (freqs >= 1000) & (freqs < 3000)
    band_cut = (freqs >= 3000) & (freqs <= 3600)
    band_high = freqs > 3600
    energy = np.mean(stft, axis=1) + 1e-8
    cutoff_ratio = float(energy[band_cut].sum() / (energy.sum() + 1e-8))
    high_drop = float(energy[band_high].sum() / (energy[band_mid].sum() + 1e-8)) if band_mid.any() else 0.0

    try:
        f0, voiced_flag, _ = librosa.pyin(
            y,
            fmin=librosa.note_to_hz("C2"),
            fmax=librosa.note_to_hz("C7"),
            sr=sr,
        )
        f0_voiced = f0[np.isfinite(f0)]
        f0_std = _safe_std(f0_voiced)
        f0_mean = _safe_mean(f0_voiced)
        voiced_ratio = float(np.mean(voiced_flag)) if voiced_flag is not None else 0.0
    except Exception:
        f0_std, f0_mean, voiced_ratio = 0.0, 0.0, 0.0

    mfcc_delta = librosa.feature.delta(mfcc)
    artifacts = _score_artifacts(
        cutoff_ratio=cutoff_ratio,
        high_drop=high_drop,
        flux_mean=_safe_mean(flux),
        f0_std=f0_std,
        centroid_std=_safe_std(centroid),
        zcr_mean=_safe_mean(zcr),
    )

    return {
        "sample_rate": sr,
        "duration_sec": round(len(y) / sr, 3),
        "mfcc_mean": mfcc.mean(axis=1).astype(float).tolist(),
        "mfcc_std": mfcc.std(axis=1).astype(float).tolist(),
        "log_mel_stats": {
            "mean": _safe_mean(log_mel),
            "std": _safe_std(log_mel),
        },
        "prosody": {
            "f0_mean_hz": round(f0_mean, 2),
            "f0_std_hz": round(f0_std, 2),
            "voiced_ratio": round(voiced_ratio, 3),
            "zcr": round(_safe_mean(zcr), 4),
            "rms": round(_safe_mean(rms), 5),
        },
        "spectrum": {
            "centroid_hz": round(_safe_mean(centroid), 1),
            "rolloff_hz": round(_safe_mean(rolloff), 1),
            "bandwidth_hz": round(_safe_mean(bandwidth), 1),
            "contrast_mean": round(_safe_mean(contrast), 3),
            "spectral_flux": round(_safe_mean(flux), 4),
            "low_band_energy": round(float(energy[band_low].sum()), 4),
            "mid_band_energy": round(float(energy[band_mid].sum()), 4),
            "cutoff_band_3k2_ratio": round(cutoff_ratio, 4),
            "highband_to_mid_ratio": round(high_drop, 4),
            "mfcc_delta_energy": round(float(np.mean(np.abs(mfcc_delta))), 4),
        },
        "artifacts": artifacts,
        "spectrogram_preview": _preview_log_mel(log_mel),
    }


def _preview_log_mel(log_mel: np.ndarray, bands: int = 48, frames: int = 64) -> list[list[float]]:
    """Downsampled log-mel for the frontend waterfall (not stored as raw audio)."""
    import librosa

    reduced = librosa.util.fix_length(log_mel, size=frames, axis=1)
    if reduced.shape[0] > bands:
        idx = np.linspace(0, reduced.shape[0] - 1, bands).astype(int)
        reduced = reduced[idx]
    reduced = reduced[:, :frames]
    reduced = (reduced - reduced.min()) / (np.ptp(reduced) + 1e-8)
    return np.round(reduced, 3).astype(float).tolist()


def _score_artifacts(
    cutoff_ratio: float,
    high_drop: float,
    flux_mean: float,
    f0_std: float,
    centroid_std: float,
    zcr_mean: float,
) -> list[dict[str, Any]]:
    spectral_disc = float(np.clip(cutoff_ratio * 4.5 + (0.35 - min(high_drop, 0.35)) * 1.8, 0, 1))
    pitch_flat = float(np.clip((18.0 - min(f0_std, 18.0)) / 18.0, 0, 1))
    vocoder = float(np.clip(flux_mean / 8.0 + zcr_mean * 2.2, 0, 1))
    jittery = float(np.clip(centroid_std / 900.0, 0, 1))

    return [
        {
            "type": "Acoustic Artifact",
            "name": "Spectral discontinuity near 3.2 kHz vocoder cutoff",
            "severity": "high" if spectral_disc > 0.65 else "medium" if spectral_disc > 0.4 else "low",
            "score": round(spectral_disc, 3),
            "description": "Energy collapse / spike in the 3.0–3.6 kHz band, a common neural vocoder fingerprint.",
        },
        {
            "type": "Prosody & Pitch Anomaly",
            "name": "Unnatural pitch micro-variation",
            "severity": "high" if pitch_flat > 0.7 else "medium" if pitch_flat > 0.45 else "low",
            "score": round(pitch_flat, 3),
            "description": "Low F0 variance relative to conversational human speech (flattened contour).",
        },
        {
            "type": "Vocoder Fingerprint",
            "name": "Phase / flux instability (HiFi-GAN-like traces)",
            "severity": "high" if vocoder > 0.65 else "medium" if vocoder > 0.4 else "low",
            "score": round(vocoder, 3),
            "description": "Elevated spectral flux and zero-crossing rate in non-stationary frames.",
        },
        {
            "type": "Channel Instability",
            "name": "Centroid jitter across frames",
            "severity": "medium" if jittery > 0.5 else "low",
            "score": round(jittery, 3),
            "description": "Frame-to-frame spectral centroid movement used as a telephony-channel cue.",
        },
    ]


def spoof_prior_from_artifacts(artifacts: list[dict[str, Any]]) -> float:
    if not artifacts:
        return 0.35
    weights = [0.34, 0.30, 0.26, 0.10]
    scores = [float(a.get("score", 0)) for a in artifacts]
    while len(scores) < len(weights):
        scores.append(0.0)
    return float(np.clip(np.dot(scores[:4], weights), 0, 1))
