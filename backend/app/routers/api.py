from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Optional

from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from pydantic import BaseModel, Field

from app.services.pipeline import analyze_audio, load_audio
from app.services.ecapa import get_ecapa
from app.services.status import engine_status

router = APIRouter(prefix="/api/v1")

DEMO_SESSION = {
    "session_id": "#HOM-2026-9810",
    "incoming_number": "+91 98XXX XX210",
    "call_duration_seconds": 134,
    "claimed_caller": {
        "name": "Rajesh Kumar",
        "account_type": "Priority Banking",
        "account_number_masked": "••••4912",
        "kyc_status": "Verified",
        "customer_since": "2018",
        "registered_sim": "+91 98201 ••••7",
        "avatar_url": "/images/caller.jpg",
    },
    "transaction_context": {
        "type": "Urgent RTGS Outward Transfer",
        "amount_inr": "₹4,50,000",
        "beneficiary": "New Beneficiary (Axis Bank)",
        "urgency": "High",
        "initiated_at": "10:11:48 IST",
    },
    "telephony_flags": {
        "channel": "VoIP, Unregistered SIM",
        "carrier_spoof_detected": True,
        "carrier_gateway": "Airtel Gateway Ingress",
        "jitter_burst_ms": 42,
    },
}


class EscalationRequest(BaseModel):
    action: str
    session_id: str = "#HOM-2026-9810"
    notes: Optional[str] = None


class EnrollResponse(BaseModel):
    status: str
    dim: int
    backend: str
    training_status: str
    embedding_preview: list[float] = Field(default_factory=list)


@router.get("/health")
def health() -> dict[str, Any]:
    return engine_status()


@router.get("/session")
def session() -> dict[str, Any]:
    payload = dict(DEMO_SESSION)
    payload["synthesis_risk"] = {
        "risk_score_percentage": 87,
        "risk_level": "critical",
        "risk_label": "High Risk Likely Synthetic Voice",
        "statistical_confidence": 0.994,
        "recommended_action": "callback",
    }
    return payload


@router.get("/telemetry")
def telemetry() -> dict[str, Any]:
    return {
        "model_version": "HOMADOS-Fusion-v3",
        "evaluated_features_count": 80,
        "sampling_rate_hz": 16000,
        "frame_length_ms": 20,
        "spectral_discontinuity_freq_khz": 3.2,
        "artifacts": [
            {
                "type": "Acoustic Artifact",
                "name": "Spectral Discontinuity at 3.2 kHz",
                "severity": "high",
                "score": 0.94,
                "description": "Sudden frequency cutoff in high-order harmonic resonance bands characteristic of generative vocoders.",
            },
            {
                "type": "Prosody & Pitch Anomaly",
                "name": "Unnatural Pitch Flattening (Score: 0.91)",
                "severity": "high",
                "score": 0.91,
                "description": "Fundamental frequency (F0) contour shows absence of emotional micro-tremors.",
            },
            {
                "type": "Vocoder Fingerprint",
                "name": "Neural Vocoder Traces (HiFi-GAN / DiffSinger)",
                "severity": "high",
                "score": 0.88,
                "description": "Checkerboard phase noise artifacts detected in non-vocal transition frames.",
            },
            {
                "type": "Biometric Divergence",
                "name": "Voiceprint Divergence: 43.8%",
                "severity": "medium",
                "score": 0.438,
                "description": "Cosine distance mismatch against customer enrolled baseline voice vector.",
            },
        ],
    }


@router.get("/reports")
def reports() -> list[dict[str, Any]]:
    return [
        {
            "session_id": "#HOM-2026-9810",
            "timestamp": "10:14:02 IST",
            "caller": "+91 98XXX XX210",
            "risk_score": 87,
            "risk_label": "Critical (Synthetic Voice)",
            "action_taken": "Callback Verification Recommended",
            "action_sub": "VoIP spoof flagged, out-of-band fallback invoked",
            "status": "danger",
        },
        {
            "session_id": "#HOM-2026-9809",
            "timestamp": "09:48:19 IST",
            "caller": "+91 97XXX XX542",
            "risk_score": 14,
            "risk_label": "Safe (Genuine Human)",
            "action_taken": "Voice Biometrics Cleared",
            "action_sub": "Acoustic envelope matches enrolled customer profile",
            "status": "safe",
        },
        {
            "session_id": "#HOM-2026-9808",
            "timestamp": "09:12:45 IST",
            "caller": "+91 91XXX XX890",
            "risk_score": 62,
            "risk_label": "Suspicious (Elevated Jitter)",
            "action_taken": "OTP Step-Up Challenge Dispatched",
            "action_sub": "User passed in-app push auth, transaction approved",
            "status": "warning",
        },
    ]


@router.post("/escalate")
def escalate(request: EscalationRequest) -> dict[str, Any]:
    messages = {
        "callback": "VoIP stream severed. Out-of-band PSTN call initiated to the customer's registered SIM.",
        "otp": "Step-up authentication challenge pushed to the registered device.",
        "supervisor": "Live audio features and fusion telemetry transferred to Senior Cybercrime Desk.",
        "none": "No escalation required.",
    }
    if request.action not in messages:
        raise HTTPException(status_code=400, detail="Unknown action")
    return {
        "status": "success",
        "action_executed": request.action,
        "session_id": request.session_id,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "message": messages[request.action],
        "audit_logged": True,
        "notes": request.notes,
    }


@router.post("/analyze")
async def analyze(
    file: UploadFile = File(...),
    enrolled_csv: Optional[str] = Form(default=None),
) -> dict[str, Any]:
    data = await file.read()
    if not data:
        raise HTTPException(status_code=400, detail="Empty audio upload")
    enrolled = None
    if enrolled_csv:
        try:
            enrolled = [float(x) for x in enrolled_csv.split(",") if x.strip()]
        except ValueError as exc:
            raise HTTPException(status_code=400, detail="Invalid enrolled embedding") from exc
    return analyze_audio(data, file.filename or "clip.wav", enrolled)


@router.post("/enroll", response_model=EnrollResponse)
async def enroll(file: UploadFile = File(...)) -> EnrollResponse:
    data = await file.read()
    y, sr = load_audio(data, file.filename or "enroll.wav")
    result = get_ecapa().embed(y, sr)
    return EnrollResponse(
        status="ok",
        dim=result["dim"],
        backend=result["backend"],
        training_status=result["training_status"],
        embedding_preview=result["vector"][:16],
    )
