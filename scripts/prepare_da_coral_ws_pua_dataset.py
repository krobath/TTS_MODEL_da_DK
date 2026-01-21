#!/usr/bin/env python3
"""
Prepare CoRal-TTS (CC0) for training a WordSuggestor Danish voice pack.

We generate a *character-based* VITS training dataset, but the "characters" are
Private Use Area (PUA) codepoints that represent phoneme token IDs produced by our
neural G2P model (CTC).

Why this design:
- Coqui VITS exports + sherpa-onnx runtime work well with a "characters" frontend.
- Traditional phoneme-token frontends require a phonemizer (often eSpeak; GPL risk).
- With PUA encoding, the phonemizer is the app (WordSuggestor), not the TTS runtime.
  The app converts raw text -> phoneme IDs -> PUA string before calling sherpa.

Output format:
- LJSpeech-style folder with:
  - wavs/*.wav
  - metadata.csv  (id|original_text|pua_text)  (matches Coqui's `ljspeech` formatter)
  - ws_voicepack.json  (frontend metadata for the app)

Dependencies (install in your env):
  python3 -m pip install datasets soundfile numpy onnxruntime

Optional (for resampling to 22050Hz like the sherpa Coqui packs):
  python3 -m pip install torch torchaudio
"""

from __future__ import annotations

import argparse
import json
import os
import re
import unicodedata
import io
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import soundfile as sf

try:
    import onnxruntime as ort
except Exception as e:  # pragma: no cover
    raise SystemExit("Missing dependency: onnxruntime. Install with: python3 -m pip install onnxruntime") from e


PUA_DEFAULT_BASE = 0xE000


def nfc(s: str) -> str:
    return unicodedata.normalize("NFC", s)


def is_letter(ch: str) -> bool:
    # Keep it simple: treat Unicode letters as "word" characters.
    # Danish special letters are still `isalpha()==True` in Python.
    return ch.isalpha()


@dataclass(frozen=True)
class G2PVocab:
    chars: List[str]
    tokens: List[str]
    blank_id: int
    max_word_len: int


class G2PCTCOnnx:
    """
    Minimal ONNXRuntime wrapper for our CTC G2P model.
    - input: (1, max_word_len) int tensor
    - output: logits (1, T, V) float tensor
    """

    def __init__(self, model_path: Path, vocab: G2PVocab):
        self.vocab = vocab
        self.char_to_id: Dict[str, int] = {c: i for i, c in enumerate(vocab.chars)}

        # CPU only: deterministic and works everywhere (macOS/Linux/Windows).
        self.sess = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
        self.input_name = self.sess.get_inputs()[0].name
        self.output_name = self.sess.get_outputs()[0].name

        input_type = self.sess.get_inputs()[0].type
        # Common: tensor(int64) or tensor(int32)
        self.input_dtype = np.int64 if "int64" in input_type else np.int32

    def _encode_word(self, word: str) -> Optional[np.ndarray]:
        w = nfc(word).strip().lower()
        if not w:
            return None

        ids: List[int] = []
        for ch in w:
            if ch in self.char_to_id:
                ids.append(self.char_to_id[ch])
                continue

            # Conservative fold for letters outside the vocab.
            folded = unicodedata.normalize("NFKD", ch)
            for f in folded:
                f = f.lower()
                if f in self.char_to_id:
                    ids.append(self.char_to_id[f])

        if not ids:
            return None

        max_len = self.vocab.max_word_len
        ids = ids[:max_len]
        padded = ids + [0] * (max_len - len(ids))
        arr = np.array(padded, dtype=self.input_dtype).reshape(1, max_len)
        return arr

    def phonemize_token_ids(self, word: str) -> List[int]:
        inp = self._encode_word(word)
        if inp is None:
            return []

        logits = self.sess.run([self.output_name], {self.input_name: inp})[0]
        # logits: (1, T, V) or (T, V)
        if logits.ndim == 3:
            logits = logits[0]

        # Greedy CTC decode: argmax per time step, collapse repeats, drop blank.
        raw = logits.argmax(axis=-1).tolist()
        out: List[int] = []
        prev = None
        for idx in raw:
            if prev == idx:
                continue
            prev = idx
            if idx == self.vocab.blank_id:
                continue
            out.append(int(idx))
        return out


def pua_from_ids(ids: List[int], base: int) -> str:
    # Each token ID becomes a single PUA scalar.
    # This must match the mapping used by the app and the voice pack.
    chars: List[str] = []
    for i in ids:
        v = base + int(i)
        try:
            chars.append(chr(v))
        except ValueError:
            # Skip invalid codepoints (should never happen with sane base).
            continue
    return "".join(chars)


def split_into_word_and_nonword(text: str) -> List[Tuple[str, bool]]:
    """
    Split into runs of letters vs "everything else" while preserving separators.
    Returns list of (segment, is_word).
    """
    out: List[Tuple[str, bool]] = []
    buf: List[str] = []
    buf_is_word: Optional[bool] = None

    def flush():
        nonlocal buf, buf_is_word
        if buf:
            out.append(("".join(buf), bool(buf_is_word)))
        buf = []
        buf_is_word = None

    for ch in text:
        isw = is_letter(ch)
        if buf_is_word is None:
            buf_is_word = isw
            buf.append(ch)
            continue
        if isw == buf_is_word:
            buf.append(ch)
        else:
            flush()
            buf_is_word = isw
            buf.append(ch)
    flush()
    return out


def text_to_pua(text: str, g2p: G2PCTCOnnx, base: int, cache: Dict[str, str]) -> str:
    text = nfc(text)
    parts = split_into_word_and_nonword(text)
    out: List[str] = []
    for seg, is_word_seg in parts:
        if not is_word_seg:
            out.append(seg)
            continue

        key = seg.lower()
        cached = cache.get(key)
        if cached is not None:
            out.append(cached)
            continue

        ids = g2p.phonemize_token_ids(seg)
        pua = pua_from_ids(ids, base) if ids else seg
        cache[key] = pua
        out.append(pua)

    return "".join(out)


def maybe_resample(audio: np.ndarray, orig_sr: int, target_sr: int) -> Tuple[np.ndarray, int]:
    if orig_sr == target_sr:
        return audio, orig_sr

    try:
        import torch
        import torchaudio
    except Exception as e:  # pragma: no cover
        raise SystemExit(
            "Resampling requested but torch/torchaudio are missing. Install with: python3 -m pip install torch torchaudio"
        ) from e

    # torchaudio expects (channels, time)
    t = torch.tensor(audio, dtype=torch.float32)
    if t.ndim == 1:
        t = t.unsqueeze(0)
    res = torchaudio.functional.resample(t, orig_freq=orig_sr, new_freq=target_sr)
    out = res.squeeze(0).cpu().numpy()
    return out, target_sr


def load_audio_row(audio_obj: object) -> Tuple[np.ndarray, int]:
    """
    Load audio from a HF datasets `Audio` feature row.

    We intentionally avoid datasets' audio decoding (TorchCodec/FFmpeg dependency)
    by casting the dataset column with `Audio(decode=False)` and decoding ourselves
    via `soundfile` (WAV/FLAC/etc.).
    """
    if not isinstance(audio_obj, dict):
        raise RuntimeError(f"Unexpected audio object type: {type(audio_obj)}")

    # If the dataset was loaded with decode=True, keep backward compatibility.
    if "array" in audio_obj and "sampling_rate" in audio_obj:
        arr = np.array(audio_obj["array"], dtype=np.float32)
        sr = int(audio_obj["sampling_rate"])
        return arr, sr

    path = audio_obj.get("path")
    raw = audio_obj.get("bytes")

    # With `Audio(decode=False)` it's common for `path` to be a relative filename
    # (e.g. "mic_00001.wav") while the actual bytes are in `bytes`.
    # Prefer reading the file only when the path exists on disk.
    if path:
        p = Path(str(path))
        if p.is_absolute() and p.exists():
            data, sr = sf.read(str(p), dtype="float32", always_2d=False)
        elif p.exists():
            data, sr = sf.read(str(p.resolve()), dtype="float32", always_2d=False)
        elif raw:
            data, sr = sf.read(io.BytesIO(raw), dtype="float32", always_2d=False)
        else:
            raise RuntimeError(f"Audio path does not exist and no bytes provided: path={path!r}")
    elif raw:
        data, sr = sf.read(io.BytesIO(raw), dtype="float32", always_2d=False)
    else:
        raise RuntimeError(f"Audio row missing both path and bytes keys: keys={list(audio_obj.keys())}")

    arr = np.array(data, dtype=np.float32)
    if arr.ndim == 2:
        # Downmix to mono for TTS training consistency.
        arr = arr.mean(axis=1).astype(np.float32, copy=False)
    return arr, int(sr)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True, help="Output directory (LJSpeech-style)")
    ap.add_argument("--speaker", default=None, help="Optional speaker_id to filter (e.g. 'mic' or 'lea')")
    ap.add_argument("--limit", type=int, default=0, help="Optional max number of samples (0 = no limit)")
    ap.add_argument("--g2p-dir", default="g2p-models/da-DK", help="Directory containing da_ctc.onnx + vocab.json")
    ap.add_argument("--pua-base", type=lambda s: int(s, 0), default=PUA_DEFAULT_BASE, help="Base scalar (e.g. 0xE000)")
    ap.add_argument("--resample", type=int, default=0, help="Optional target sample rate (e.g. 22050). 0 = keep original.")
    ap.add_argument("--cache-dir", default=None, help="Optional HF datasets cache dir")
    args = ap.parse_args()

    out_root = Path(args.out).expanduser().resolve()
    wav_dir = out_root / "wavs"
    wav_dir.mkdir(parents=True, exist_ok=True)

    g2p_dir = Path(args.g2p_dir).expanduser().resolve()
    vocab_path = g2p_dir / "vocab.json"
    model_path = g2p_dir / "da_ctc.onnx"
    if not vocab_path.exists():
        raise SystemExit(f"Missing vocab.json at {vocab_path}")
    if not model_path.exists():
        raise SystemExit(f"Missing da_ctc.onnx at {model_path}")

    vocab_raw = json.loads(vocab_path.read_text(encoding="utf-8"))
    vocab = G2PVocab(
        chars=vocab_raw["chars"],
        tokens=vocab_raw["tokens"],
        blank_id=int(vocab_raw["blank_id"]),
        max_word_len=int(vocab_raw["max_word_len"]),
    )
    g2p = G2PCTCOnnx(model_path=model_path, vocab=vocab)

    # Lazy import: this can trigger a large download on first use.
    try:
        from datasets import load_dataset  # type: ignore
        from datasets import Audio  # type: ignore
    except Exception as e:  # pragma: no cover
        raise SystemExit("Missing dependency: datasets. Install with: python3 -m pip install datasets") from e

    ds_kwargs = {}
    if args.cache_dir:
        ds_kwargs["cache_dir"] = args.cache_dir
    ds = load_dataset("CoRal-project/coral-tts", **ds_kwargs)
    split = ds["train"]

    # Avoid TorchCodec/FFmpeg runtime dependency in datasets by disabling audio decoding.
    # We decode audio ourselves using `soundfile`.
    split = split.cast_column("audio", Audio(sampling_rate=None, decode=False))

    # Discover speakers (for user convenience).
    speakers = sorted(set(split["speaker_id"]))
    if args.speaker and args.speaker not in speakers:
        raise SystemExit(f"Unknown speaker '{args.speaker}'. Available: {', '.join(speakers)}")

    # Cache word->PUA for speed.
    word_cache: Dict[str, str] = {}

    meta_lines: List[str] = []
    seen = 0
    kept = 0

    for row in split:
        seen += 1
        spk = row["speaker_id"]
        if args.speaker and spk != args.speaker:
            continue

        arr, sr = load_audio_row(row["audio"])

        if args.resample:
            arr, sr = maybe_resample(arr, orig_sr=sr, target_sr=args.resample)

        text = str(row["text"])
        pua_text = text_to_pua(text, g2p=g2p, base=args.pua_base, cache=word_cache)

        # Use a stable ID/filename so we can re-run without producing duplicates.
        tid = int(row["transcription_id"])
        item_id = f"{spk}_{tid:05d}"
        fname = f"{item_id}.wav"
        wav_path = wav_dir / fname
        if not wav_path.exists():
            sf.write(str(wav_path), arr, sr)

        # Coqui's `ljspeech` formatter expects:
        #   wav_file = root/wavs/<col0>.wav
        #   text     = col2
        # So we write: id|original_text|pua_text
        safe_text = text.replace("|", " ")
        meta_lines.append(f"{item_id}|{safe_text}|{pua_text}")
        kept += 1

        if args.limit and kept >= args.limit:
            break

    # Write metadata.csv
    (out_root / "metadata.csv").write_text("\n".join(meta_lines) + "\n", encoding="utf-8")

    # Write a WordSuggestor pack manifest next to the dataset so export/install can copy it.
    ws_manifest = {
        "schema": 1,
        "languageCode": "da-DK",
        "frontend": "ws_pua_phonemes_v1",
        "puaBase": int(args.pua_base),
        "g2pLanguageCode": "da-DK",
        "notes": "This dataset uses PUA-encoded phonemes produced by the WordSuggestor neural G2P model.",
    }
    (out_root / "ws_voicepack.json").write_text(json.dumps(ws_manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    stats = {
        "dataset": "CoRal-project/coral-tts",
        "speakerFilter": args.speaker,
        "seen": seen,
        "kept": kept,
        "out": str(out_root),
        "sampleRate": args.resample if args.resample else "source",
        "g2p": {
            "model": str(model_path),
            "vocab": str(vocab_path),
            "tokens": len(vocab.tokens),
            "chars": len(vocab.chars),
            "blank_id": vocab.blank_id,
            "max_word_len": vocab.max_word_len,
        },
        "puaBase": int(args.pua_base),
    }
    (out_root / "stats.json").write_text(json.dumps(stats, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    print("== prepare_da_coral_ws_pua_dataset ==")
    print(f"out={out_root}")
    print(f"speakers={speakers}")
    print(f"kept={kept} (speaker={args.speaker or 'ALL'})")
    print(f"wavs={wav_dir} metadata={out_root / 'metadata.csv'}")
    print(f"ws_voicepack={out_root / 'ws_voicepack.json'}")


if __name__ == "__main__":
    main()
