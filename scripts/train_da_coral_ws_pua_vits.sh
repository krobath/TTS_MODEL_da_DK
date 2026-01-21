#!/usr/bin/env bash
set -euo pipefail

# Train a Coqui VITS model for the Danish CoRal "WS PUA phonemes" dataset.
#
# This wrapper:
# - disables Coqui/Trainer telemetry (network calls break offline training)
# - sets writable cache dirs for Numba/Matplotlib (avoids permission issues)
# - injects `sitecustomize.py` to patch torchaudio decoding (avoids torchcodec/ffmpeg)
#
# Usage:
#   conda activate ws-tts
#   ./scripts/train_da_coral_ws_pua_vits.sh tts-training/da/coral_mic_ws_pua/config.json
#
# Optional:
#   WS_TTS_DIAG=1   prints a short environment report (torch/mps availability) before training.
#   WS_TTS_DEBUG=1  enables verbose logging from our `sitecustomize.py` patches.
#   WS_TORCH_NUM_THREADS / WS_TORCH_NUM_INTEROP_THREADS tune PyTorch CPU thread pools.

CONFIG_PATH="${1:-}"
if [[ -z "${CONFIG_PATH}" ]]; then
  echo "Usage: train_da_coral_ws_pua_vits.sh <config.json> [extra TTS args...]" >&2
  exit 2
fi
shift 1

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

export TRAINER_TELEMETRY=0
export NUMBA_CACHE_DIR="${NUMBA_CACHE_DIR:-/tmp/numba-cache}"
export NUMBA_DISABLE_CACHE="${NUMBA_DISABLE_CACHE:-1}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/mplconfig}"

export PYTHONPATH="${PYTHONPATH:-}"
export PYTHONPATH="${ROOT_DIR}/scripts/py:${PYTHONPATH}"

if [[ "${WS_TTS_DIAG:-0}" == "1" ]]; then
  python "${ROOT_DIR}/scripts/ws_tts_diag.py" || true
fi

python -m TTS.bin.train_tts --config_path "${CONFIG_PATH}" "$@"

