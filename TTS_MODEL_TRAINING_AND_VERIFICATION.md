# TTS Model (da-DK) — Training & Verification (WS‑PUA)

This document is the canonical, continuously updated reference for **how we train and verify** our Danish text-to-speech (TTS) voice using the **WS‑PUA phoneme frontend**.

## Current product direction (recommended)

For now, WordSuggestor should use **built-in platform TTS/STT**:

- **TTS:** `AVSpeechSynthesizer` (macOS/iOS/iPadOS) with user-installed “Enhanced/Forbedret” voices.
- **STT:** platform Speech APIs (Apple Speech framework on Apple platforms; Windows/ChromeOS later).

Why:
- Highest quality with lowest app complexity today (no model downloads/runtimes).
- Licensing is simple and commercial-friendly.

WS‑PUA training remains valuable R&D for consistent cross-platform offline voices, but it is currently **experimental**
and should not be the default shipped path until G2P determinism and voice quality are validated.

## What we are building

We train a Danish VITS voice model (Coqui TTS) on **CoRal‑TTS (CC0)**. The key design is:

- We do **not** use a traditional phonemizer at runtime (e.g. eSpeak) due to licensing constraints.
- Instead, WordSuggestor performs: **Text → G2P token IDs → PUA characters → sherpa‑onnx VITS inference**.
- Each G2P token ID `i` becomes one Private Use Area character `U+E000 + i`.

This is referred to as **WS‑PUA** in the codebase and voice pack metadata.

## Why verification is non‑negotiable (the “gibberish” failure mode)

WS‑PUA voice conditioning depends on the **exact** token-id sequence produced by the G2P model.

If the G2P output used during dataset preparation (currently **ONNXRuntime**) differs from what the app uses at runtime (currently **CoreML**), the voice will typically sound like **unrecognizable gibberish**, even if training “succeeds”.

Therefore, we must run an **ONNX vs CoreML G2P parity check** before multi‑day training runs.

## System voices (Enhanced) — user instructions

We cannot programmatically download Apple “Enhanced/Forbedret” voices. Users must install them in system settings.

macOS (Danish UI varies slightly by version):
- Systemindstillinger → Tilgængelighed → Oplæst indhold
- Vælg en **systemstemme** → Administrer stemmer…
- Installer “Dansk” (forbedret) / “Enhanced”

In-app:
- Settings → `Lydindstillinger`
- Choose `Oplæsningsmotor = System`
- Use the “Åbn systemindstillinger for oplæsning…” button (best-effort deep link)
- Optionally disable “Brug systemets oplæsningsindstillinger” and pick a specific installed voice.

## Repo locations

Main app repo (developer machine):
- `/Users/krobath/XcodeProjects/WordSuggestor`

Training repo stub (copy to separate Git repo for Mac Studio training):
- `/Users/krobath/XcodeProjects/WordSuggestor/TTS_MODEL_da_DK`

## Required artifacts

### 1) Danish G2P model (must be present in training repo)

The training repo does not ship the G2P model by default. Copy from the main repo:

mkdir -p /Users/krobath/XcodeProjects/WordSuggestor/TTS_MODEL_da_DK/g2p-models/da-DK
rsync -a /Users/krobath/XcodeProjects/WordSuggestor/g2p-models/da-DK/ /Users/krobath/XcodeProjects/WordSuggestor/TTS_MODEL_da_DK/g2p-models/da-DK/

Expected files:
- `TTS_MODEL_da_DK/g2p-models/da-DK/vocab.json`
- `TTS_MODEL_da_DK/g2p-models/da-DK/da_ctc.onnx`
- `TTS_MODEL_da_DK/g2p-models/da-DK/da_ctc.mlpackage`

## Environment setup (training machine)

From the training repo:

cd /Users/krobath/XcodeProjects/WordSuggestor/TTS_MODEL_da_DK

Create/activate environment:

conda env create -f environment.yml
conda activate ws-tts

If you need a clean rebuild of the environment:

conda env remove -n ws-tts -y
conda env create -f environment.yml
conda activate ws-tts

Notes:
- `TTS_MODEL_da_DK/environment.yml` includes Coqui TTS + torch/torchaudio + datasets + soundfile + onnx/onnxruntime + coremltools.
- Training wrapper injects `scripts/py/sitecustomize.py` via `PYTHONPATH` to force torchaudio decoding via `soundfile` (avoids TorchCodec/FFmpeg issues).

## Verification: ONNX vs CoreML G2P parity (fail early)

### Quick sanity parity check (default small word set)

cd /Users/krobath/XcodeProjects/WordSuggestor/TTS_MODEL_da_DK
conda activate ws-tts
python scripts/compare_g2p_onnx_coreml.py --g2p-dir g2p-models/da-DK --pua-base 0xE000 --strict

### Large parity check (top 2000 words)

We generate a top‑frequency wordlist from the Danish lexicon SQLite and use it for parity checking.

#### Create the top‑2000 list from SQLite

Run from the main WordSuggestor repo root:

cd /Users/krobath/XcodeProjects/WordSuggestor
python3 TTS_MODEL_da_DK/scripts/extract_top_words_from_lexicon_sqlite.py --db WordSuggestor/da_lexicon.sqlite --n 2000 --word-only --out-dir TTS_MODEL_da_DK/wordlists

Outputs:
- `TTS_MODEL_da_DK/wordlists/da_top2000_display.txt` (lowercased words; recommended input to parity check)
- `TTS_MODEL_da_DK/wordlists/da_top2000_terms.tsv` (rank + term_norm + display + freq)

#### Run parity check on the full list

cd /Users/krobath/XcodeProjects/WordSuggestor/TTS_MODEL_da_DK
conda activate ws-tts
python scripts/compare_g2p_onnx_coreml.py --g2p-dir g2p-models/da-DK --pua-base 0xE000 --words-file wordlists/da_top2000_display.txt --limit 2000 --strict

If this fails:
- Do **not** start training a WS‑PUA voice pack.
- The training pipeline must be aligned so the same G2P outputs are used in both training and the app.

Note:
- The parity tool runs CoreML inference with `cpu-only` compute units by default to make results
  deterministic across machines. The WordSuggestor app also forces `MLModelConfiguration.computeUnits = .cpuOnly`
  for G2P for the same reason.

If you see different mismatch sets on different Apple machines even when:
- `vocab.json sha256`, `da_ctc.onnx sha256`, and `da_ctc.mlpackage tree_sha256` are identical, and
- CoreML is forced to `cpu-only`,

then CoreML inference is still producing numerically different logits across devices/OS versions (enough to flip CTC decoding).
That is a **hard blocker** for WS‑PUA portability if the app relies on CoreML G2P.

In that situation, the recommended path is to pick a **single portable G2P backend** used everywhere:
- Prefer ONNXRuntime for G2P on all platforms (macOS/iPadOS/Windows/ChromeOS), so training + app share the same backend.
- Or avoid neural G2P for WS‑PUA entirely by using a deterministic pronunciation lexicon (where coverage permits).

## G2P model health checks (before training any WS‑PUA voice)

WS‑PUA voice training only works if the Danish neural G2P model is “healthy” and **consistent across runtimes**.

“Healthy” means:
- `vocab.json` is present and sane (has `chars`, `tokens`, `blank_id`, `max_word_len`).
- The ONNX export and the CoreML export produce the **same decoded token-id sequence** for the same input word.
- Token lengths look plausible (not “always max_word_len” for most words).

### 1) Sanity check the G2P artifact bundle

These files must exist in `TTS_MODEL_da_DK/g2p-models/da-DK/`:
- `vocab.json`
- `da_ctc.onnx`
- `da_ctc.mlpackage`

Quick sanity (prints counts and key params):

cd /Users/krobath/XcodeProjects/WordSuggestor/TTS_MODEL_da_DK
python -c "import json; v=json.load(open('g2p-models/da-DK/vocab.json','r',encoding='utf-8')); print('blank_id=',v.get('blank_id'),'max_word_len=',v.get('max_word_len'),'chars=',len(v.get('chars',[])),'tokens=',len(v.get('tokens',[])))"

Expected:
- `blank_id` is `0`
- `max_word_len` is `32` (our current fixed-length design)

### 2) Parity check: ONNX vs CoreML (strict)

This is the most important “fail early” check:

cd /Users/krobath/XcodeProjects/WordSuggestor/TTS_MODEL_da_DK
conda activate ws-tts
python scripts/compare_g2p_onnx_coreml.py --g2p-dir g2p-models/da-DK --pua-base 0xE000 --strict

Recommended stronger check (top 2000 words):

python scripts/compare_g2p_onnx_coreml.py --g2p-dir g2p-models/da-DK --pua-base 0xE000 --words-file wordlists/da_top2000_display.txt --limit 2000 --strict

If this fails:
- Do **not** train WS‑PUA voices.
- Rebuild/re-export the G2P model so ONNX and CoreML match (see “Rebuilding the Danish neural G2P model (CTC)”).

### 3) Quick “does decoding look reasonable?” smoke-check (optional)

If you see most words decoding to ~32 tokens (max_word_len), something is very wrong even if training “runs”.

python scripts/compare_g2p_onnx_coreml.py --g2p-dir g2p-models/da-DK --pua-base 0xE000 --words "hej,ikke,det,arbejde,arbejder,arbejdede,forvirring,kampagne,kampagnen,tømrer,schæfer,administration" --strict

## Dataset preparation (CoRal → WS‑PUA dataset)

This produces an LJSpeech-style dataset directory with:
- `wavs/*.wav`
- `metadata.csv` (id|original_text|pua_text)
- `ws_voicepack.json` (frontend metadata for the app)

Run (training repo):

cd /Users/krobath/XcodeProjects/WordSuggestor/TTS_MODEL_da_DK
conda activate ws-tts
rm -rf /tmp/ws-coral-mic-ws-pua
./scripts/prepare_da_coral_ws_pua_dataset.sh --out /tmp/ws-coral-mic-ws-pua --speaker mic --resample 22050

## Config generation + verification (required PUA alphabet)

Generate config:

cd /Users/krobath/XcodeProjects/WordSuggestor/TTS_MODEL_da_DK
conda activate ws-tts
mkdir -p work/tts-training/da/coral_mic_ws_pua
python scripts/generate_coqui_vits_config_ws_pua.py --dataset /tmp/ws-coral-mic-ws-pua --out-dir work/tts-training/da/coral_mic_ws_pua --run-name coral_mic_ws_pua --epochs 500 --batch-size 16

Patch config so `characters.characters` contains the full required PUA alphabet:

python scripts/patch_coqui_config_add_required_pua.py --config work/tts-training/da/coral_mic_ws_pua/config.json --g2p-dir g2p-models/da-DK --pua-base 0xE000

Verify dataset + config compatibility (strict):

python scripts/verify_ws_pua_pipeline.py --g2p-dir g2p-models/da-DK --pua-base 0xE000 --dataset /tmp/ws-coral-mic-ws-pua --config work/tts-training/da/coral_mic_ws_pua/config.json --strict

## Training (full run)

Start training:

cd /Users/krobath/XcodeProjects/WordSuggestor/TTS_MODEL_da_DK
conda activate ws-tts
WS_TTS_DIAG=1 ./scripts/train_da_coral_ws_pua_vits.sh work/tts-training/da/coral_mic_ws_pua/config.json --use_accelerate true

### Minimum “smoke test” training (to confirm it’s not gibberish)

If WS‑PUA parity checks pass, the output should become “speech‑like” quickly.

Practical recommendation:
- Train **3–5 epochs** on the full dataset, export, and test inside the app before committing to multi‑day training.

Even faster option:
- Use a small dataset (e.g. `--limit 2000`) and train ~10 epochs as a pipeline smoke test.

Example (small dataset):

rm -rf /tmp/ws-coral-mic-ws-pua
./scripts/prepare_da_coral_ws_pua_dataset.sh --out /tmp/ws-coral-mic-ws-pua --speaker mic --resample 22050 --limit 2000

Then generate config and train as usual (reduce epochs in config generation if desired).

## Export to a WordSuggestor voice pack

After training, pick a checkpoint (often `best_model.pth`) from your run directory and export:

CHECKPOINT_DIR=/Users/krobath/XcodeProjects/WordSuggestor/TTS_MODEL_da_DK/work/tts-training/da/coral_mic_ws_pua/<your-run-folder>
python scripts/export_coqui_vits_to_ws_voicepack.py --config ${CHECKPOINT_DIR}/config.json --checkpoint ${CHECKPOINT_DIR}/best_model.pth --out-root /Users/krobath/XcodeProjects/WordSuggestor/TTS_MODEL_da_DK/work/tts-voices --pack-name vits-ws-da-coral-mic-pua --language da --speaker-label coral_mic --ws-frontend ws_pua_phonemes_v1 --ws-pua-base 0xE000 --ws-g2p-language da-DK

## Verify the exported voice pack tokens.txt

This confirms the exported voice pack’s `tokens.txt` contains the full required PUA alphabet:

python scripts/verify_ws_pua_pipeline.py --g2p-dir g2p-models/da-DK --pua-base 0xE000 --voicepack /Users/krobath/XcodeProjects/WordSuggestor/TTS_MODEL_da_DK/work/tts-voices/vits-ws-da-coral-mic-pua --strict

## One-command runner (standard path)

This convenience script does dataset preparation → config generation → verification → training.

cd /Users/krobath/XcodeProjects/WordSuggestor/TTS_MODEL_da_DK
conda activate ws-tts
./scripts/run_da_coral_mic_train.sh

Important:
- It runs the ONNX vs CoreML parity check in **strict mode** by default (fails if mismatches exist).
- To bypass parity checking (not recommended):

WS_SKIP_G2P_PARITY=1 ./scripts/run_da_coral_mic_train.sh

## Import into the WordSuggestor app (developer workflow)

On your development Mac, copy the exported voice pack folder (from `TTS_MODEL_da_DK/work/tts-voices/...`) to your dev repo or any folder.

Then in the app:
- Settings → “Import voices” → choose the folder containing the pack (or the pack folder itself, depending on the UI path you implemented)
- Select the new voice in Settings → Lydindstillinger.

## Troubleshooting checklist (high impact)

If the voice sounds like gibberish:

1) Confirm G2P parity passes (ONNX vs CoreML), ideally on top‑2000.
   - If parity fails, fix parity first (do not train).

2) Confirm the app is actually using the intended G2P artifact.
   - Ensure the bundled `da_ctc.mlpackage` in the app matches the training repo’s.

3) Confirm the app’s WS‑PUA transformer is not producing “fixed-length per word” outputs.
   - A smoking-gun symptom is every word mapping to ~32 PUA tokens (max_word_len).

4) Confirm `tokens.txt` includes *all* PUA tokens for the chosen `puaBase` and token count.

5) Only after all above pass: increase training epochs / fine-tune hyperparameters.

## Rebuilding the Danish neural G2P model (CTC)

Rebuild this if:
- ONNX vs CoreML parity fails, or
- You suspect G2P artifacts are stale/corrupted, or
- You changed the pronunciation dataset and want to retrain.

Important constraint:
- PyTorch **CTCLoss is not implemented on MPS** (Apple GPU). Train the CTC G2P model on CPU (recommended), or use CPU fallback.

### Step A — Build the pronunciation dataset (CC0 sources)

cd /Users/krobath/XcodeProjects/WordSuggestor
./scripts/build_da_pronunciation_dataset_cc0.sh

This produces:
- /tmp/ws-pronunciation/da_pron_training.tsv

### Step B — Prepare the clean NST-only G2P dataset + split

cd /Users/krobath/XcodeProjects/WordSuggestor
./scripts/prepare_da_nst_g2p_dataset.sh
./scripts/split_da_nst_g2p_dataset.sh

Outputs:
- /tmp/ws-pronunciation/splits/da_nst_g2p/train.tsv
- /tmp/ws-pronunciation/splits/da_nst_g2p/dev.tsv
- /tmp/ws-pronunciation/splits/da_nst_g2p/test.tsv
- /tmp/ws-pronunciation/da_nst_g2p_gold.json

### Step C — Train the neural CTC G2P model

CPU (recommended):

cd /Users/krobath/XcodeProjects/WordSuggestor
FREQ_LEMMA=LanguageRessources/freq-lemma/freq-30k-in.txt DEVICE=cpu ./scripts/train_da_phonemizer_neural.sh

Optional CPU fallback if you force DEVICE=mps (not recommended):

cd /Users/krobath/XcodeProjects/WordSuggestor
PYTORCH_ENABLE_MPS_FALLBACK=1 FREQ_LEMMA=LanguageRessources/freq-lemma/freq-30k-in.txt DEVICE=mps ./scripts/train_da_phonemizer_neural.sh

Outputs (default):
- /tmp/ws-pronunciation/neural/da_ctc/model_state.pt
- /tmp/ws-pronunciation/neural/da_ctc/vocab.json
- /tmp/ws-pronunciation/neural/da_ctc/report.json

### Step D — Export to ONNX + CoreML

ONNX:

cd /Users/krobath/XcodeProjects/WordSuggestor
./scripts/export_g2p_ctc_to_onnx.py --in-dir /tmp/ws-pronunciation/neural/da_ctc --out /tmp/ws-pronunciation/neural/da_ctc/da_ctc.onnx

CoreML (.mlpackage):

cd /Users/krobath/XcodeProjects/WordSuggestor
./scripts/export_g2p_ctc_to_coreml.py --in-dir /tmp/ws-pronunciation/neural/da_ctc --out /tmp/ws-pronunciation/neural/da_ctc/da_ctc.mlpackage

### Step E — Install into the app repo’s `g2p-models/`

cd /Users/krobath/XcodeProjects/WordSuggestor
./scripts/install_g2p_model.sh --lang da-DK --in-dir /tmp/ws-pronunciation/neural/da_ctc

Rebuild WordSuggestor so the updated `Resources/NeuralG2P/da-DK/...` is bundled.

### Step F — Re-verify parity (mandatory)

Copy the updated `g2p-models/da-DK/` into the training repo and run strict parity:

rsync -a /Users/krobath/XcodeProjects/WordSuggestor/g2p-models/da-DK/ /Users/krobath/XcodeProjects/WordSuggestor/TTS_MODEL_da_DK/g2p-models/da-DK/
cd /Users/krobath/XcodeProjects/WordSuggestor/TTS_MODEL_da_DK
conda activate ws-tts
python scripts/compare_g2p_onnx_coreml.py --g2p-dir g2p-models/da-DK --pua-base 0xE000 --words-file wordlists/da_top2000_display.txt --limit 2000 --strict

Only when parity passes should you train WS‑PUA voices again.
