#!/usr/bin/env bash
set -euo pipefail

# Convenience "one command" runner for Mac Studio / training machines.
# It creates a dataset in /tmp, writes the generated config under ./work/,
# and then starts training.
#
# You can interrupt training (Ctrl+C) and later resume by re-running the same
# command and pointing Coqui to the last checkpoint.

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATASET_DIR="${DATASET_DIR:-/tmp/ws-coral-mic-ws-pua}"
OUT_DIR="${OUT_DIR:-${ROOT_DIR}/work/tts-training/da/coral_mic_ws_pua}"
RUN_NAME="${RUN_NAME:-coral_mic_ws_pua}"

if [[ ! -f "${ROOT_DIR}/g2p-models/da-DK/da_ctc.onnx" ]]; then
  echo "Missing g2p-models/da-DK/da_ctc.onnx" >&2
  echo "Copy it from your WordSuggestor repo into: ${ROOT_DIR}/g2p-models/da-DK/" >&2
  exit 2
fi

mkdir -p "${ROOT_DIR}/work"

echo "== (1) Diagnostics =="
python "${ROOT_DIR}/scripts/ws_tts_diag.py" || true

echo "== (2) Prepare dataset (downloads CoRal on first run) =="
"${ROOT_DIR}/scripts/prepare_da_coral_ws_pua_dataset.sh" --out "${DATASET_DIR}" --speaker mic --resample 22050 --min-frames-per-char "${WS_MIN_FRAMES_PER_CHAR:-1.0}"

echo "== (3) Generate Coqui config =="
python "${ROOT_DIR}/scripts/generate_coqui_vits_config_ws_pua.py" \
  --dataset "${DATASET_DIR}" \
  --out-dir "${OUT_DIR}" \
  --run-name "${RUN_NAME}" \
  --epochs 500 \
  --batch-size 16 \
  --min-audio-sec "${WS_MIN_AUDIO_SEC:-1.5}" \
  --max-audio-sec "${WS_MAX_AUDIO_SEC:-20.0}" \
  --max-text-len "${WS_MAX_TEXT_LEN:-120}" \
  ${WS_DISABLE_EVAL:+--disable-eval}

echo "== (4) Train =="
WS_TTS_DIAG=1 "${ROOT_DIR}/scripts/train_da_coral_ws_pua_vits.sh" "${OUT_DIR}/config.json" --use_accelerate true
