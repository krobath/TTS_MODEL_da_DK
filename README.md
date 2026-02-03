# TTS_MODEL (Danish) — Training Repo Stub

This folder is designed to be copied into its own Git repository, so you can clone it to another Mac
(e.g. your Mac Studio) and run voice training there while continuing app development on your laptop.

It contains:
- Dataset preparation (CoRal-TTS → WS “PUA phonemes” dataset)
- Coqui VITS config generation
- Coqui VITS training wrapper (offline-friendly + macOS friendly)
- Export tool (Coqui checkpoint → sherpa-onnx voice pack)
- Diagnostics to confirm whether PyTorch can use MPS (Apple GPU)
- `environment.yml` / `requirements.txt` for recreating the Python environment

It intentionally does **not** contain:
- CoRal dataset audio (downloaded on demand; large)
- Training outputs (checkpoints/logs; large)
- Your Danish G2P model (you can copy it in; see below)

## Expected layout

- `scripts/` contains everything you need to prepare/train/export.
- `scripts/py/sitecustomize.py` is injected into the training process to make `torchaudio.load()`
  use `soundfile` (avoids TorchCodec/FFmpeg issues on macOS).
- `baselines/vits-coqui-da-cv/config.json` is used as the hyperparameter baseline.
- `g2p-models/da-DK/` must exist (copy from your main WordSuggestor repo).

## Quick start (high level)

1) Create/activate your training environment (Coqui TTS + torch + torchaudio + soundfile + datasets + onnx).
2) Copy `g2p-models/da-DK/` from the WordSuggestor repo into `TTS_MODEL/g2p-models/da-DK/`.
3) Prepare dataset: `scripts/prepare_da_coral_ws_pua_dataset.sh`.
4) Generate config: `scripts/generate_coqui_vits_config_ws_pua.py`.
5) Train: `scripts/train_da_coral_ws_pua_vits.sh` (optionally with `--use_accelerate true`).
6) Export: `scripts/export_coqui_vits_to_ws_voicepack.py`.

If you want a single “do the standard thing” command (downloads CoRal on first run), use:
- `scripts/run_da_coral_mic_train.sh`

### Important: WS‑PUA “fail early” checks

WS‑PUA voices are conditioned on the *exact* phoneme-token-id sequences produced by our G2P model.
If the G2P outputs used during dataset generation (ONNX) differ from the G2P outputs used at runtime
in the app (CoreML), the trained voice will usually sound like “gibberish”.

To catch this before wasting days training, `scripts/run_da_coral_mic_train.sh` runs:
- `scripts/compare_g2p_onnx_coreml.py ... --strict` and will fail if any mismatch is detected.

If you explicitly want to bypass this check (not recommended), run with:
- `WS_SKIP_G2P_PARITY=1 scripts/run_da_coral_mic_train.sh`

## Why we disable language embeddings

This project trains a **single-language** Danish model. Coqui configs often enable language embeddings,
even with `num_languages=1`. That path has caused crashes on Apple Silicon when using MPS/Accelerate,
so the generated config explicitly disables language embeddings.

## Detailed notes

See `docs/da_phoneme_voice_training.md`.
