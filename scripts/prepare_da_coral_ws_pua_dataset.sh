#!/usr/bin/env bash
set -euo pipefail

# Prepare CoRal-TTS (CC0) for training a Danish voice pack using the WordSuggestor
# "WS PUA phonemes" approach.
#
# This script does NOT train the voice. It only materializes:
#   - wavs/*.wav
#   - metadata.csv (id|original_text|pua_text)  (Coqui `ljspeech` formatter compatible)
#   - ws_voicepack.json
#
# Usage:
#   ./scripts/prepare_da_coral_ws_pua_dataset.sh --out /tmp/ws-coral-mic-ws-pua --speaker mic --resample 22050
#
# Notes:
# - Requires the Danish G2P model at `g2p-models/da-DK/` (copy from your WordSuggestor repo).
# - Downloads ~15GB on first run (CoRal-TTS).

OUT=""
SPEAKER=""
RESAMPLE="22050"
LIMIT="0"

# Default to a repo-local HF cache.
CACHE_DIR="hf-cache"
MIN_FRAMES_PER_CHAR="1.0"
HOP_LENGTH="256"

usage() {
  cat <<'EOF'
Usage:
  prepare_da_coral_ws_pua_dataset.sh --out <dir> [--speaker <id>] [--resample <hz>] [--limit N] [--cache-dir <dir>]

Examples:
  ./scripts/prepare_da_coral_ws_pua_dataset.sh --out /tmp/ws-coral-mic-ws-pua --speaker mic --resample 22050
  ./scripts/prepare_da_coral_ws_pua_dataset.sh --out /tmp/ws-coral-all-ws-pua --resample 0 --limit 100
EOF
}

if [[ "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then
  usage
  exit 0
fi

while [[ $# -gt 0 ]]; do
  case "$1" in
    --out) OUT="${2:-}"; shift 2 ;;
    --speaker) SPEAKER="${2:-}"; shift 2 ;;
    --resample) RESAMPLE="${2:-}"; shift 2 ;;
    --limit) LIMIT="${2:-}"; shift 2 ;;
    --cache-dir) CACHE_DIR="${2:-}"; shift 2 ;;
    --min-frames-per-char) MIN_FRAMES_PER_CHAR="${2:-}"; shift 2 ;;
    --hop-length) HOP_LENGTH="${2:-}"; shift 2 ;;
    *) echo "Unknown arg: $1" >&2; usage >&2; exit 2 ;;
  esac
done

if [[ -z "${OUT}" ]]; then
  echo "Error: missing --out" >&2
  usage >&2
  exit 2
fi

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

args=( "--out" "${OUT}" "--g2p-dir" "${ROOT_DIR}/g2p-models/da-DK" "--resample" "${RESAMPLE}" "--limit" "${LIMIT}" )
args+=( "--min-frames-per-char" "${MIN_FRAMES_PER_CHAR}" "--hop-length" "${HOP_LENGTH}" )
if [[ -n "${SPEAKER}" ]]; then
  args+=( "--speaker" "${SPEAKER}" )
fi
if [[ -n "${CACHE_DIR}" ]]; then
  mkdir -p "${CACHE_DIR}"
  args+=( "--cache-dir" "${CACHE_DIR}" )
fi

python3 "${ROOT_DIR}/scripts/prepare_da_coral_ws_pua_dataset.py" "${args[@]}"
