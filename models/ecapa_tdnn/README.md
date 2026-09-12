# ECAPA-TDNN (pretrained, inference only)

Source: [speechbrain/spkrec-ecapa-voxceleb](https://huggingface.co/speechbrain/spkrec-ecapa-voxceleb)

```powershell
pip install -r backend/requirements-ml.txt
python scripts/download_pretrained.py --ecapa
```

Do **not** commit `.ckpt` / `.pt` files. Training and fine-tuning belong on a supercomputer (see `docs/SUPERCOMPUTER_TRAINING.md`).
