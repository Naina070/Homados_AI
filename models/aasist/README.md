# AASIST spoof detector

Official code: https://github.com/clovaai/aasist

Until the official inference adapter **and** a compatible checkpoint are
validated, HOMADOS uses its librosa spectral / prosodic branch (mel, MFCC,
pYIN, 3.2 kHz cutoff band). A raw `.pth` file alone is not treated as an
active detector.

## Safe integration hand-off

1. Pin a commit of the official repository and retain its licence notices.
2. Add its model definition and the checkpoint's exact preprocessing to a
   separately reviewed inference adapter.
3. Validate it on held-out bona-fide and spoofed ASVspoof samples before
   enabling it with `ENABLE_AASIST=true`.
4. Put the verified checkpoint at `models/aasist/aasist.pth` (it remains
   ignored by Git). Fine-tuning is an HPC task, not a laptop task.
