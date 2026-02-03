#!/usr/bin/env python3
"""
Compare G2P outputs between:
  - ONNXRuntime inference of `da_ctc.onnx` (used by dataset preparation scripts)
  - CoreML inference of `da_ctc.mlpackage` (used by the WordSuggestor app)

If these two disagree, a WS-PUA voice pack trained using ONNX G2P will likely sound like
"gibberish" when run in the app (or vice-versa), because the voice model is conditioned on
the exact token-id sequences produced by the phonemizer.

This is a fast "fail early" check you can run before a multi-day training run.

Dependencies (in your training env):
  python -m pip install onnxruntime coremltools numpy
"""

from __future__ import annotations

import argparse
import json
import unicodedata
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np


def nfc(s: str) -> str:
    return unicodedata.normalize("NFC", s)


def load_vocab(vocab_path: Path) -> Dict:
    vocab = json.loads(vocab_path.read_text(encoding="utf-8"))
    for k in ("chars", "tokens", "blank_id", "max_word_len"):
        if k not in vocab:
            raise SystemExit(f"vocab.json missing '{k}': {vocab_path}")
    return vocab


def encode_word(word: str, chars: List[str], max_len: int) -> Optional[List[int]]:
    w = nfc(word).strip().lower()
    if not w:
        return None

    char_to_id: Dict[str, int] = {c: i for i, c in enumerate(chars)}
    ids: List[int] = []

    for ch in w:
        if ch in char_to_id:
            ids.append(char_to_id[ch])
            continue

        folded = unicodedata.normalize("NFKD", ch)
        for f in folded:
            f = f.lower()
            if f in char_to_id:
                ids.append(char_to_id[f])

    if not ids:
        return None
    ids = ids[:max_len]
    ids = ids + [0] * (max_len - len(ids))
    return ids


def ctc_greedy_decode(logits: np.ndarray, blank_id: int) -> List[int]:
    # logits: (T, V) or (1, T, V)
    if logits.ndim == 3:
        logits = logits[0]
    raw = logits.argmax(axis=-1).tolist()
    out: List[int] = []
    prev = None
    for idx in raw:
        if prev == idx:
            continue
        prev = idx
        if int(idx) == int(blank_id):
            continue
        out.append(int(idx))
    return out


def pua_from_ids(ids: List[int], pua_base: int) -> str:
    return "".join(chr(pua_base + i) for i in ids)


def run_onnx(onnx_path: Path, input_ids: np.ndarray) -> np.ndarray:
    try:
        import onnxruntime as ort
    except Exception as e:  # pragma: no cover
        raise SystemExit("Missing dependency: onnxruntime. Install with: python -m pip install onnxruntime") from e

    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    input_name = sess.get_inputs()[0].name
    output_name = sess.get_outputs()[0].name
    return sess.run([output_name], {input_name: input_ids})[0]


def run_coreml(mlpackage_path: Path, input_ids: np.ndarray) -> np.ndarray:
    try:
        import coremltools as ct
    except Exception as e:  # pragma: no cover
        raise SystemExit("Missing dependency: coremltools. Install with: python -m pip install coremltools") from e

    model = ct.models.MLModel(str(mlpackage_path))
    out = model.predict({"input_ids": input_ids})
    if not out:
        raise SystemExit(f"CoreML returned no outputs for: {mlpackage_path}")
    # Prefer "logits" if present; otherwise pick first.
    if "logits" in out:
        return out["logits"]
    return out[sorted(out.keys())[0]]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--g2p-dir", required=True, help="Directory containing vocab.json + da_ctc.onnx + da_ctc.mlpackage")
    ap.add_argument("--pua-base", type=lambda s: int(s, 0), default=0xE000)
    ap.add_argument(
        "--words",
        default="hej,ikke,kampagne,kampagnen,tømrer,chef,schæfer,løber",
        help="Comma-separated list (ignored if --words-file is provided)",
    )
    ap.add_argument(
        "--words-file",
        default="",
        help="Path to newline-separated wordlist (lines starting with # are ignored)",
    )
    ap.add_argument("--limit", type=int, default=0, help="Limit number of words read from --words-file")
    ap.add_argument("--strict", action="store_true", help="Exit non-zero if any mismatch is found")
    args = ap.parse_args()

    g2p_dir = Path(args.g2p_dir).expanduser().resolve()
    vocab_path = g2p_dir / "vocab.json"
    onnx_path = g2p_dir / "da_ctc.onnx"
    mlpackage_path = g2p_dir / "da_ctc.mlpackage"

    if not vocab_path.exists():
        raise SystemExit(f"Missing: {vocab_path}")
    if not onnx_path.exists():
        raise SystemExit(f"Missing: {onnx_path}")
    if not mlpackage_path.exists():
        raise SystemExit(f"Missing: {mlpackage_path}")

    vocab = load_vocab(vocab_path)
    chars: List[str] = vocab["chars"]
    blank_id = int(vocab["blank_id"])
    max_len = int(vocab["max_word_len"])

    if args.words_file:
        words_path = Path(args.words_file).expanduser().resolve()
        if not words_path.exists():
            raise SystemExit(f"--words-file not found: {words_path}")
        words: List[str] = []
        for line in words_path.read_text(encoding="utf-8").splitlines():
            s = line.strip()
            if not s or s.startswith("#"):
                continue
            words.append(s)
            if args.limit and len(words) >= int(args.limit):
                break
        if not words:
            raise SystemExit(f"--words-file produced no words: {words_path}")
    else:
        words = [w.strip() for w in str(args.words).split(",") if w.strip()]
        if not words:
            raise SystemExit("--words produced no words")

    print("== G2P Parity Check (ONNX vs CoreML) ==")
    print(f"g2p_dir={g2p_dir}")
    print(f"vocab={vocab_path} tokens={len(vocab['tokens'])} blank_id={blank_id} max_word_len={max_len}")
    print(f"onnx={onnx_path}")
    print(f"coreml={mlpackage_path}")
    print(f"puaBase=0x{int(args.pua_base):X}")
    if args.words_file:
        print(f"words_file={Path(args.words_file).expanduser().resolve()} limit={int(args.limit)}")
    print("")

    mismatches: List[Tuple[str, List[int], List[int]]] = []

    for w in words:
        ids = encode_word(w, chars=chars, max_len=max_len)
        if ids is None:
            print(f"- word={w!r}: skipped (empty after encoding)")
            continue

        x_onnx = np.array(ids, dtype=np.int64).reshape(1, max_len)
        x_coreml = np.array(ids, dtype=np.int32).reshape(1, max_len)

        onnx_logits = run_onnx(onnx_path, x_onnx)
        coreml_logits = run_coreml(mlpackage_path, x_coreml)

        onnx_out = ctc_greedy_decode(onnx_logits, blank_id=blank_id)
        coreml_out = ctc_greedy_decode(coreml_logits, blank_id=blank_id)

        ok = onnx_out == coreml_out
        if not ok:
            mismatches.append((w, onnx_out, coreml_out))

        # Print a compact line for each word (avoid huge dumps).
        onnx_len = len(onnx_out)
        coreml_len = len(coreml_out)
        status = "OK" if ok else "DIFF"
        print(f"- {status} word={w!r} len_onnx={onnx_len} len_coreml={coreml_len}")

        if not ok:
            # Show first 40 IDs and the PUA hex preview (helps spot systematic shifts).
            onnx_preview = " ".join(map(str, onnx_out[:40]))
            coreml_preview = " ".join(map(str, coreml_out[:40]))
            onnx_pua = " ".join(f"{ord(ch):04X}" for ch in pua_from_ids(onnx_out[:40], int(args.pua_base)))
            coreml_pua = " ".join(f"{ord(ch):04X}" for ch in pua_from_ids(coreml_out[:40], int(args.pua_base)))
            print(f"  onnx_ids : {onnx_preview}")
            print(f"  coreml_ids: {coreml_preview}")
            print(f"  onnx_pua : {onnx_pua}")
            print(f"  coreml_pua: {coreml_pua}")

    print("")
    if mismatches:
        print(f"⚠️  MISMATCHES: {len(mismatches)}/{len(words)} words differ (this will break WS-PUA voice conditioning)")
        if args.strict:
            raise SystemExit(2)
    else:
        print("✅ OK: ONNX and CoreML G2P outputs match for all tested words")


if __name__ == "__main__":
    main()
