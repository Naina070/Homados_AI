#!/usr/bin/env python3
"""Download public pretrained weights used by HOMADOS AI.

ECAPA-TDNN (SpeechBrain, VoxCeleb1+2) is attached for inference only.
Fine-tuning / full training is intentionally deferred to HPC / PARAM access.

AASIST: this script records the official GitHub source. Place `aasist.pth`
from the clovaai/aasist release into models/aasist/ when you have it.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ECAPA_DIR = ROOT / "models" / "ecapa_tdnn"
AASIST_DIR = ROOT / "models" / "aasist"


def download_ecapa() -> None:
    ECAPA_DIR.mkdir(parents=True, exist_ok=True)
    try:
        from speechbrain.inference.speaker import EncoderClassifier
    except ImportError:
        print("Install ML extras first: pip install -r backend/requirements-ml.txt")
        print("Python 3.10–3.12 is recommended for PyTorch + SpeechBrain.")
        sys.exit(1)

    print("Fetching speechbrain/spkrec-ecapa-voxceleb into", ECAPA_DIR)
    EncoderClassifier.from_hparams(
        source="speechbrain/spkrec-ecapa-voxceleb",
        savedir=str(ECAPA_DIR),
        run_opts={"device": "cpu"},
    )
    print("ECAPA-TDNN pretrained checkpoint is ready (inference only).")


def prepare_aasist() -> None:
    AASIST_DIR.mkdir(parents=True, exist_ok=True)
    note = AASIST_DIR / "SOURCE.txt"
    note.write_text(
        "Official implementation: https://github.com/clovaai/aasist\n"
        "Copy a pretrained ASVspoof 2019 LA checkpoint here as aasist.pth\n"
        "Do not train from scratch on a laptop. Use HPC for fine-tuning.\n",
        encoding="utf-8",
    )
    print("AASIST folder prepared at", AASIST_DIR)


if __name__ == "__main__":
    prepare_aasist()
    if "--ecapa" in sys.argv or "--all" in sys.argv:
        download_ecapa()
    else:
        print("Run with --ecapa after installing backend/requirements-ml.txt")
