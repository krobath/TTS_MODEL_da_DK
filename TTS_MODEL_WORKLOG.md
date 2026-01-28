# WordSuggestor — Danish Neural TTS Model (TTS_MODEL_da_DK)

This document explains the **Danish neural Text‑to‑Speech (TTS)** work done so far for WordSuggestor.
It is written so you can:

- Explain the approach to other people (what we built, why, what we use).
- Reproduce training on another machine (e.g. Mac Studio).
- Give a new chat session enough context to continue this work.

Repository root for this work: `TTS_MODEL_da_DK/`

---

## 1) Background, Research, and Decisions

### 1.1 Problem statement

WordSuggestor already had “read aloud” support, but:

- macOS system voices (AVSpeechSynthesizer) can sound robotic for long text.
- Neural voices we tested improved quality, but **mispronunciations** (especially Danish) were still an issue.
- We need a path that is:
  - **commercially safe** (avoid GPL/CC BY‑SA)
  - **offline-capable** (no per‑request cloud costs)
  - portable later to **Windows / Chromebooks** (at least at inference time)

### 1.2 Why “phoneme-aware TTS” is required for “awesome quality”

If a voice model is purely “grapheme/character” based, pronunciation is learned indirectly.
That makes it hard to fix specific mispronunciations reliably (e.g. user reports “kampagnen” is spoken wrong).

To fix mispronunciations systematically we want a **phoneme frontend**:

- text → phonemes (or phoneme IDs)
- TTS model trained to read those phonemes
- user dictionary / overrides can adjust phonemes deterministically

### 1.3 Licensing constraints and why we avoided eSpeak/eSpeak‑ng

- We need to avoid **GPL/AGPL** and **CC BY‑SA** in shipped components.
- eSpeak / eSpeak‑ng are often used as phonemizers but are GPL‑licensed, so we avoided them.

### 1.4 Core decision: “PUA phoneme” frontend (WordSuggestor-only)

We implemented a WordSuggestor-specific trick to get phoneme control without shipping a GPL phonemizer:

1. Train (or bundle) our own phonemizer: **neural G2P** (grapheme‑to‑phoneme) model.
2. Represent phoneme tokens as **Unicode Private Use Area (PUA)** characters:
   - G2P outputs token IDs: `[t0, t1, t2, ...]`
   - We map each token to one PUA scalar: `chr(puaBase + tokenID)` (default `puaBase = 0xE000`)
   - The TTS model is trained as if these PUA symbols are just “characters”
3. At runtime:
   - WordSuggestor converts normal text → token IDs → PUA string
   - The TTS runtime reads the PUA string using a character-based frontend

Benefits:

- No GPL phonemizer is shipped.
- We get phoneme-level control (pronunciation overrides become deterministic).
- Training and inference still fit the standard VITS + “characters frontend” export path.

### 1.5 Data decision: use CC0 Danish voice datasets

We decided to train a new Danish voice pack primarily on:

- **CoRal‑TTS** (CC0): high-quality Danish recordings (2 speakers, ~17h each).

Optional future robustness stage:

- **NST‑da** (CC0): multi-speaker corpus that can help generalization and spellings.

### 1.6 Model/framework decisions

Training framework:

- **Coqui TTS** (`TTS==0.22.0`) with the **VITS** architecture.

Inference/runtime in app:

- **sherpa-onnx** offline TTS runtime (already integrated into WordSuggestor).

Hardware acceleration:

- On Apple Silicon: use **MPS** via **HuggingFace Accelerate** (`--use_accelerate true`).

### 1.7 Key operational lessons learned

1) **TorchAudio 2.9 defaults to TorchCodec/FFmpeg** for decoding.
   - This caused repeated failures during training (missing FFmpeg / torchcodec).
   - We implemented a patch that forces `torchaudio.load()` to use `soundfile`.

2) **MPS + Accelerate eval crashes can occur**:
   - Error: `RuntimeError: Placeholder storage has not been allocated on MPS device!`
   - We implemented a “disable evaluation” option so long runs don’t die mid‑training.

3) **Language embeddings are unnecessary for single‑language training**:
   - Coqui base configs often enable language embeddings even with a single language.
   - This path has caused instability on MPS/Accelerate in our experiments.
   - We explicitly disable language embeddings in the generated config.

---

## 2) What Data We Use (and How)

### 2.1 CoRal‑TTS (CC0)

Used for training a high-quality Danish voice.

Downloaded automatically by our dataset prep script (Hugging Face dataset):

- Dataset ID: `CoRal-project/coral-tts`

Speaker selection:

- We can train per speaker (e.g. `--speaker mic`), or later do multi-speaker.

### 2.2 Danish G2P model (neural phonemizer)

We use a prebuilt Danish G2P model in this repo:

- `g2p-models/da-DK/da_ctc.onnx`
- `g2p-models/da-DK/vocab.json`

It is used **only** to transform CoRal text into the PUA phoneme representation.

---

## 3) Architecture Overview (End-to-End)

### 3.1 Dataset pipeline

`CoRal audio + transcript` →
`neural G2P (CTC) produces token IDs` →
`token IDs → PUA characters` →
`LJSpeech-style dataset (wavs + metadata.csv)`

We chose LJSpeech format because Coqui has a built-in `ljspeech` dataset formatter.

### 3.2 Training pipeline

`LJSpeech dataset (PUA text)` →
`Coqui VITS training` →
`checkpoint (.pth)`

### 3.3 Export pipeline

`checkpoint (.pth) + config.json` →
`export to model.onnx + tokens.txt` →
`package as WordSuggestor voice pack` →
`bundle into the app (later)`

---

## 4) Code / Scripts (What They Do)

All paths below are relative to `TTS_MODEL_da_DK/`.

### 4.1 Diagnostics

- `scripts/ws_tts_diag.py`
  - Prints Python executable, architecture, Rosetta status, torch version, and `mps_available`.
  - Use this first to ensure you are not accidentally running x86_64 Python on Apple Silicon.

### 4.2 Dataset preparation (CoRal → WS PUA phonemes → LJSpeech format)

- `scripts/prepare_da_coral_ws_pua_dataset.py`
  - Downloads CoRal dataset, selects a speaker, reads audio using `soundfile`, optionally resamples.
  - Converts each transcript to PUA phoneme text using the Danish G2P ONNX model.
  - Output folder contains:
    - `wavs/*.wav`
    - `metadata.csv` rows: `id|original_text|pua_text`
    - `ws_voicepack.json` (frontend metadata)
    - `stats.json`

- `scripts/prepare_da_coral_ws_pua_dataset.sh`
  - Shell wrapper for the Python script.
  - Creates a local Hugging Face cache directory (avoid permission problems).

### 4.3 Training config generator (baseline → WS PUA training config)

- `scripts/generate_coqui_vits_config_ws_pua.py`
  - Reads baseline config: `baselines/vits-coqui-da-cv/config.json`
  - Reads dataset `metadata.csv` and builds the exact character inventory from PUA text.
  - Writes a new Coqui training `config.json`.
  - Important modifications we apply:
    - disables language embeddings (`use_language_embedding=false`)
    - can disable evaluation (`--disable-eval`)
    - sets loader workers default to 0 (macOS stability)

### 4.4 Training wrapper (offline-friendly, macOS-friendly)

- `scripts/train_da_coral_ws_pua_vits.sh`
  - Disables trainer telemetry (`TRAINER_TELEMETRY=0`)
  - Sets Numba/Matplotlib cache dirs to `/tmp` (permission stability)
  - Injects `scripts/py/sitecustomize.py` into `PYTHONPATH`
  - Optional environment variables:
    - `WS_TTS_DIAG=1` prints diagnostics before training
    - `WS_TTS_DEBUG=1` prints debug logs from patches
    - `WS_TORCH_NUM_THREADS` / `WS_TORCH_NUM_INTEROP_THREADS` tune CPU threads

- `scripts/py/sitecustomize.py`
  - Forces `torchaudio.load()` to use `soundfile` (avoids TorchCodec/FFmpeg requirement).
  - Sets torch multiprocessing sharing strategy to `file_system` (macOS shared memory issues).

### 4.5 One-command runner (Mac Studio friendly)

- `scripts/run_da_coral_mic_train.sh`
  - Runs diagnostics
  - Prepares dataset into `/tmp/ws-coral-mic-ws-pua`
  - Generates config into `work/tts-training/da/coral_mic_ws_pua/config.json`
  - Starts training (by default uses accelerate)
  - If `WS_DISABLE_EVAL=1` is set, it generates config with evaluation disabled.

### 4.6 Export to a sherpa-onnx compatible voice pack

- `scripts/export_coqui_vits_to_ws_voicepack.py`
  - Uses Coqui’s export method to produce `model.onnx`.
  - Writes `tokens.txt` and pack metadata (speaker/language ids).
  - Writes `ws_voicepack.json` so WordSuggestor knows to apply the WS PUA frontend.

---

## 5) Step-by-Step Commands (Data Prep → Training → “Evaluation”)

The most reliable test of quality is to export a voice pack and listen to it in a controlled way.
We currently “evaluate” mainly by **listening tests** and by tracking whether training is stable.

### 5.1 Machine setup (Mac Studio, no Homebrew)

Install Apple command line tools (gives `git`, `curl`, etc):

1. `xcode-select --install`

Install Miniforge (Conda + Python + pip):

2. `cd ~`
3. `curl -L -o Miniforge3-MacOSX-arm64.sh https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-MacOSX-arm64.sh`
4. `bash Miniforge3-MacOSX-arm64.sh -b -p "$HOME/miniforge3"`
5. `source "$HOME/miniforge3/bin/activate"`
6. `conda init zsh`
7. Close Terminal and reopen, then: `conda --version`

Clone this repo:

8. `cd ~`
9. `git clone YOUR_GITHUB_REPO_URL TTS_MODEL_da_DK`
10. `cd TTS_MODEL_da_DK`

Create environment:

11. `conda env create -f environment.yml`
12. `conda activate ws-tts`

Verify MPS:

13. `python scripts/ws_tts_diag.py`

### 5.2 Prepare dataset (CoRal → WS PUA dataset)

This downloads CoRal on first run (large).

1. `conda activate ws-tts`
2. `./scripts/prepare_da_coral_ws_pua_dataset.sh --out /tmp/ws-coral-mic-ws-pua --speaker mic --resample 22050`

Output you should see:

- `/tmp/ws-coral-mic-ws-pua/wavs/*.wav`
- `/tmp/ws-coral-mic-ws-pua/metadata.csv`
- `/tmp/ws-coral-mic-ws-pua/ws_voicepack.json`

### 5.3 Generate a training config

1. `python scripts/generate_coqui_vits_config_ws_pua.py --dataset /tmp/ws-coral-mic-ws-pua --out-dir work/tts-training/da/coral_mic_ws_pua --run-name coral_mic_ws_pua --epochs 500 --batch-size 16`

If you hit the MPS eval crash (`Placeholder storage ...`) disable eval:

2. `python scripts/generate_coqui_vits_config_ws_pua.py --dataset /tmp/ws-coral-mic-ws-pua --out-dir work/tts-training/da/coral_mic_ws_pua --run-name coral_mic_ws_pua --epochs 500 --batch-size 16 --disable-eval`

### 5.4 Run training (GPU/MPS via accelerate)

One-command runner:

1. `WS_DISABLE_EVAL=1 ./scripts/run_da_coral_mic_train.sh`

Or manual training:

2. `WS_TTS_DIAG=1 ./scripts/train_da_coral_ws_pua_vits.sh work/tts-training/da/coral_mic_ws_pua/config.json --use_accelerate true`

### 5.5 Track progress / “evaluate” training quality

We evaluate quality primarily by:

1) exporting checkpoints to a sherpa voice pack and listening to sample texts
2) comparing pronunciation correctness on known-problem words (via a curated test set)

**Export a voice pack** from a checkpoint:

1. Find a checkpoint in your run folder (Coqui creates a run directory under `work/tts-training/.../RUNNAME-<timestamp>/`).
2. Run export:

Prereqs (in your `ws-tts` env):

- `python -m pip install onnx onnxscript`

Export command:

`python scripts/export_coqui_vits_to_ws_voicepack.py --config work/tts-training/da/coral_mic_ws_pua/config.json --checkpoint PATH_TO_CHECKPOINT.pth --out-root work/tts-voices --pack-name vits-ws-da-coral-pua --language da --speaker-label coral_mic --ws-frontend ws_pua_phonemes_v1 --ws-pua-base 0xE000 --ws-g2p-language da-DK`

Notes:

- The export script uses `torch.onnx.export()` under the hood (via Coqui). Newer PyTorch versions require `onnxscript` to be installed, otherwise export fails.
- Newer PyTorch versions may also default to the **torch.export/dynamo-based** ONNX exporter, which often fails for VITS due to data-dependent control flow inside spline transforms. Our export script forces the legacy tracer-based export path (and runs export on CPU) for robustness.
- By default we do **not** copy the `.pth` checkpoint into the voice pack (it’s huge and not needed at runtime). If you want it included for reproducibility, pass `--include-checkpoint`.

This will create:

- `work/tts-voices/vits-ws-da-coral-pua/vits-ws-da-coral-pua/model.onnx`
- `work/tts-voices/vits-ws-da-coral-pua/vits-ws-da-coral-pua/tokens.txt`
- `work/tts-voices/vits-ws-da-coral-pua/vits-ws-da-coral-pua/ws_voicepack.json`

At that point you can copy the pack into WordSuggestor’s voice pack directory (or bundle it into the app resources) and do listening tests.

---

## 6) Known Issues and Workarounds

### 6.1 TorchAudio / TorchCodec / FFmpeg errors

Symptoms:

- `Could not load libtorchcodec ...`

Fix:

- We patch `torchaudio.load()` to `soundfile` via `scripts/py/sitecustomize.py`.
- Always run training through `scripts/train_da_coral_ws_pua_vits.sh` (not directly `python -m TTS.bin.train_tts`).

### 6.2 MPS/Accelerate eval crash (`Placeholder storage has not been allocated on MPS device!`)

Symptoms:

- Training works, but crashes during evaluation.

Fix:

- Disable evaluation for the run:
  - set `WS_DISABLE_EVAL=1` when running `scripts/run_da_coral_mic_train.sh`
  - or pass `--disable-eval` to `scripts/generate_coqui_vits_config_ws_pua.py`

### 6.3 VITS alignment crash (NaNs + `maximum_path` index error)

Symptoms (typical):

- `loss_kl: nan` and `loss_duration: nan`
- followed by a crash in `maximum_path` / `maximum_path_numpy` with an `IndexError`

Cause:

This usually happens when a batch includes **audio that is too short** compared to the
text length (here: the PUA phoneme string). The monotonic alignment search (MAS) then
cannot find a valid path and numerical issues appear.

Fix / mitigation:

- Filter at dataset prep time:
  - `scripts/prepare_da_coral_ws_pua_dataset.sh` now supports `--min-frames-per-char`
    (default `1.0`), using an approximation `frames ≈ len(audio)/hop_length` with hop 256.
  - It also supports:
    - `--min-audio-sec` (default 1.5)
    - `--max-audio-sec` (default 20.0)
    - `--max-text-len` (default 120)
- Filter at training time:
  - `scripts/generate_coqui_vits_config_ws_pua.py` now defaults to:
    - `--min-audio-sec 1.5`
    - `--max-text-len 120`
  - Both can be overridden if needed once training is stable.

### 6.4 Coqui DataLoader `IndexError: list index out of range` (MPS/Accelerate)

Symptoms:

- Crash during training DataLoader iteration
- Stack trace ends in `TTS/tts/models/vits.py` `__getitem__` repeatedly calling itself and then:
  `IndexError: list index out of range`

Likely cause:

- Coqui discards samples at DataLoader construction time based on `min_audio_len/max_audio_len`
  and `min_text_len/max_text_len`.
- On some MPS/Accelerate runs this appears to cause sampler/index mismatches after discarding.

Fix:

- We now **disable Coqui internal discarding by default** and rely on dataset-side filtering.
  - `scripts/generate_coqui_vits_config_ws_pua.py` sets very wide limits unless you pass `--enable-coqui-filters`.

If you still see NaNs after this filtering:

- reduce batch size
- reduce max audio length
- add grad clipping / lower LR (future work)

### 6.3 No MPS available

If `python scripts/ws_tts_diag.py` shows `mps_available=false`, training is CPU-only and will be much slower.
Usually this means:

- wrong python architecture (x86_64 under Rosetta)
- wrong torch build/channel

---

## 7) Current Status / Next Steps

### Status

- CoRal → PUA dataset pipeline works.
- Coqui VITS training works on MPS (GPU utilized) using `--use_accelerate true`.
- We have workarounds for common macOS failures (TorchCodec/FFmpeg and MPS eval crash).

### Next steps

1) Establish a repeatable “listening evaluation” suite:
   - fixed set of Danish sentences + known mispronunciation words
   - consistent export + playback tests

2) Improve text normalization before phonemization:
   - numbers, abbreviations, dates, URLs, etc.

3) Expand beyond a single speaker:
   - train separate packs per CoRal speaker (male/female), or multi-speaker with explicit speaker control.

4) Iterate on training settings to increase quality:
   - longer training, batch size changes, grad clipping, LR schedule tweaks, etc.
