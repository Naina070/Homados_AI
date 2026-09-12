# Supercomputer / HPC training (later)

ECAPA-TDNN and AASIST are **attached as pretrained public checkpoints**. This
repository does not train them on a laptop.

## What is already wired

| Model | Public source | Role now | Later on HPC |
|---|---|---|---|
| ECAPA-TDNN | SpeechBrain `spkrec-ecapa-voxceleb` | 192-D embeddings + cosine identity | Fine-tune last layers on telephony / Indian-language call audio |
| AASIST | [clovaai/aasist](https://github.com/clovaai/aasist) | Folder + loader; spectral prior until `.pth` | Fine-tune on ASVspoof 2019/2021 + in-house clones |
| Spectral | librosa | Live now | Keep as explainability branch |

## Suggested PARAM / NSM job (when allocation exists)

1. Python 3.11 module + CUDA PyTorch matching the cluster.
2. `pip install -r backend/requirements-ml.txt`
3. Do **not** retrain ECAPA from random weights on VoxCeleb2 unless the cluster
   time is explicitly granted for that (multi-GPU days). Prefer:
   - freeze frame layers, train pooling + embedding on 16 kHz telephony audio
   - AASIST 10–20 epochs on ASVspoof 2019 LA (~4–10 GPU-hours)
   - fusion head on the team's 60/20/20 call-session set (< 1 GPU-hour)
4. Export `models/ecapa_tdnn/` and `models/aasist/aasist.pth` back into this repo's
   ignore rules (store weights on the cluster or Hugging Face private space).

See `ECAPA-TDNN_and_VoiceGuardAI_Technical_Deep_Dive.md` in `docs/` for the
full SIH data and compute narrative.
