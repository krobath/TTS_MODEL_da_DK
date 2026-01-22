# Danish Neural TTS (Phoneme-Based) — CoRal-TTS (CC0) Plan & Pipeline

Goal: produce a **high-quality Danish neural voice** that:

- Sounds noticeably better than the current character-based `vits-coqui-da-cv` pack.
- Uses a **phoneme frontend** so we can correct mispronunciations reliably.
- Stays compatible with WordSuggestor’s commercial licensing constraints (avoid GPL/CC BY‑SA).

This is a **big lift**, but we now have a strong CC0 foundation:

- Primary dataset: **CoRal-TTS (CC0)** — 2 professional Danish speakers (~17h each).
- Optional robustness/pretraining: **NST-da (CC0)** — large multi-speaker corpus.

We train a voice model, export it to ONNX, and package it as a WordSuggestor voice pack.

## 0) Why the current approach can’t get “awesome” pronunciation

Our current shipped/offline Danish voice pack is `vits-coqui-da-cv`:

- It is **character/grapheme-based**: `use_phonemes=false` in its `config.json`.
- WordSuggestor can *generate* phoneme tokens (via our phonemizer), but the voice pack’s `tokens.txt`
  does not contain those phoneme tokens, so `--vits-lexicon` cannot be used.

To get real pronunciation control we need a **phoneme-aware voice pack**.

## 1) Data strategy

### Primary: CoRal-TTS (CC0)

Pros:
- Professional Danish recordings (high baseline quality).
- Clear licensing (CC0).
- Enough hours for a real voice pack without months of recording.

Cons:
- Two speakers: we typically want one “default” voice per pack (we can pick one speaker or train multi-speaker later).

### Optional robustness: NST-da (CC0)

NST-da is large and multi-speaker. It’s great for:
- improving coverage of strange spellings
- robustness to names/dialects/noisy tokens
- pretraining representation before fine-tuning on CoRal-TTS

## 2) Phoneme frontend strategy (no-GPL)

We avoid shipping eSpeak/eSpeak‑ng due to GPL concerns.

Instead, we use our own phonemizer (neural G2P) and a WordSuggestor-only trick:

### “PUA phoneme” frontend (WordSuggestor-only)

We represent phonemes as single Unicode “characters” in the Private Use Area (PUA):
- G2P produces a list of phoneme token IDs (CTC vocabulary indices).
- Each token ID becomes one PUA codepoint: `PUA = puaBase + tokenID` (default base `0xE000`).
- We train a *character-based* VITS model on that PUA text.
- At runtime, WordSuggestor converts normal text -> PUA phoneme string before calling sherpa.

Benefits:
- No GPL phonemizer is shipped.
- We still get phoneme-level control and consistent pronunciation.
- We can apply pronunciation overrides deterministically.

## 3) Dataset preparation (CoRal → PUA)

Script:
- `scripts/prepare_da_coral_ws_pua_dataset.sh`
- `scripts/prepare_da_coral_ws_pua_dataset.py`

It produces an LJSpeech-style folder:

- `wavs/*.wav`
- `metadata.csv` with rows: `id|original_text|pua_text` (matches Coqui’s built-in `ljspeech` formatter)
- `ws_voicepack.json` (manifest copied into the voice pack)

Note: you must have the Danish G2P model present in `g2p-models/da-DK/` (used to phonemize the CoRal text).

## 4) Model training (Coqui VITS)

We train with Coqui TTS (Python) and export an ONNX model compatible with sherpa-onnx.

This is intentionally not fully automated yet (training is slow + GPU/CPU dependent),
but the pipeline is designed to be reproducible and scriptable.

Note (offline training): Coqui's `trainer` library pings a telemetry endpoint by default.
If you have restricted/no network access, set `TRAINER_TELEMETRY=0` or use
`scripts/train_da_coral_ws_pua_vits.sh`.

Note (TorchAudio 2.9+): Coqui uses `torchaudio.load()` which defaults to TorchCodec and needs FFmpeg.
Our wrapper `scripts/train_da_coral_ws_pua_vits.sh` injects a `sitecustomize.py` patch to make
`torchaudio.load()` use `soundfile` for WAV decoding (no FFmpeg needed).

GPU note (Apple Silicon): the default Trainer backend in this setup is CPU-only unless you enable
HF Accelerate. If you install `accelerate`, you can run training on the Apple GPU (MPS) by adding:
`--use_accelerate true` to the training command.

Stability note (Apple Silicon): we explicitly disable Coqui “language embeddings” in our generated
config because our model is single-language, and the language-embedding code path has caused
crashes when experimenting with MPS/Accelerate.

Evaluation note (Apple Silicon): some environments still show intermittent MPS/Accelerate crashes
during the eval loop (error: `Placeholder storage has not been allocated on MPS device!`).
If you hit this, disable evaluation for the run:
- when generating config: add `--disable-eval`
- when using the convenience runner: set `WS_DISABLE_EVAL=1`

Diagnostics: to see what PyTorch thinks is available (CPU vs MPS), run `scripts/ws_tts_diag.py`
inside your training environment, or set `WS_TTS_DIAG=1` when running the training wrapper.

## 5) Voice pack export (Coqui → sherpa pack)

Script:
- `scripts/export_coqui_vits_to_ws_voicepack.py`

It creates a voice pack folder under `tts-voices/<pack>/<pack>/` containing:
- `model.onnx`
- `tokens.txt`
- `config.json` + `model_file.pth` (for provenance/debugging)
- `speaker_ids.json`, `language_ids.json`
- `ws_voicepack.json` (tells the app to use the PUA phoneme frontend)

## 6) How this plugs into the app

The Xcode build includes a build phase that bundles:
- `tts-runtime/` into the app Resources `NeuralTTS/runtime/`
- `tts-voices/` into the app Resources `NeuralTTS/voices/`
- `g2p-models/` into the app Resources `NeuralG2P/`

Runtime selection:
- When a voice pack contains `ws_voicepack.json` with `frontend=ws_pua_phonemes_v1`,
  `SpeechHighlighter` will transform text using `NeuralTTSInputTransformer` before calling sherpa.

This means:
- the user can keep typing normal Danish text
- the app handles phonemization + pronunciation overrides
- the sherpa runtime only sees PUA "characters"

## 7) Future work

- Better text normalization (numbers, abbreviations, dates).
- Multi-speaker support (ship `coral_female` and `coral_male` packs).
- Fine-tuning on user opt-in recordings (if you later want “personal voice”).
