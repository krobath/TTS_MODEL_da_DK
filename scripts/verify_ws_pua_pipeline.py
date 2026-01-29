#!/usr/bin/env python3
"""
Verify that the WS-PUA pipeline artifacts are compatible with a WordSuggestor G2P vocab.

Checks (when provided):
- Dataset metadata.csv: does it contain PUA scalars for each G2P token ID?
- Coqui config.json: does characters.characters contain required PUA scalars for each token ID?
- Exported voice pack tokens.txt: does it contain each required PUA scalar token?

Exit code:
  0 = OK
  2 = missing required tokens / invalid inputs
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Dict, Iterable, List, Set, Tuple


def sha256_hex(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def load_vocab(vocab_path: Path) -> Dict:
    vocab = json.loads(vocab_path.read_text(encoding="utf-8"))
    if "tokens" not in vocab:
        raise SystemExit(f"vocab.json missing 'tokens': {vocab_path}")
    return vocab


def required_pua_chars(tokens_len: int, pua_base: int) -> Set[str]:
    return {chr(pua_base + i) for i in range(tokens_len)}


def scan_metadata_pua_counts(metadata_csv: Path, tokens_len: int, pua_base: int) -> List[int]:
    counts = [0] * tokens_len
    for line in metadata_csv.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        parts = line.split("|")
        if len(parts) < 3:
            continue
        pua_text = parts[2]
        for ch in pua_text:
            o = ord(ch)
            if pua_base <= o < (pua_base + tokens_len):
                counts[o - pua_base] += 1
    return counts


def find_voicepack_tokens_file(voicepack_dir: Path) -> Path:
    if voicepack_dir.is_file() and voicepack_dir.name == "tokens.txt":
        return voicepack_dir
    direct = voicepack_dir / "tokens.txt"
    if direct.exists():
        return direct
    nested = voicepack_dir / voicepack_dir.name / "tokens.txt"
    if nested.exists():
        return nested
    raise SystemExit(f"Could not find tokens.txt under: {voicepack_dir}")


def parse_tokens_txt(tokens_path: Path) -> Set[str]:
    tokens: Set[str] = set()
    for line in tokens_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        parts = line.split()
        if not parts:
            continue
        tokens.add(parts[0])
    return tokens


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--g2p-dir", required=True, help="Directory containing vocab.json (and optionally da_ctc.onnx)")
    ap.add_argument("--pua-base", type=lambda s: int(s, 0), default=0xE000)
    ap.add_argument("--dataset", default="", help="Dataset dir containing metadata.csv (optional)")
    ap.add_argument("--config", default="", help="Coqui config.json (optional)")
    ap.add_argument("--voicepack", default="", help="Voice pack dir (or tokens.txt) (optional)")
    ap.add_argument("--strict", action="store_true", help="Exit non-zero if any required tokens are missing")
    args = ap.parse_args()

    g2p_dir = Path(args.g2p_dir).expanduser().resolve()
    vocab_path = g2p_dir / "vocab.json"
    onnx_path = g2p_dir / "da_ctc.onnx"
    if not vocab_path.exists():
        raise SystemExit(f"Missing vocab.json at: {vocab_path}")

    vocab = load_vocab(vocab_path)
    tokens = vocab["tokens"]
    tokens_len = len(tokens)
    pua_base = int(args.pua_base)

    required = required_pua_chars(tokens_len=tokens_len, pua_base=pua_base)

    print("== WS-PUA Verification ==")
    print(f"g2p_vocab={vocab_path} sha256={sha256_hex(vocab_path)} tokens={tokens_len} blank_id={vocab.get('blank_id')} max_word_len={vocab.get('max_word_len')}")
    if onnx_path.exists():
        print(f"g2p_onnx={onnx_path} sha256={sha256_hex(onnx_path)}")
    print(f"puaBase=0x{pua_base:X} requiredPUA={len(required)} (U+{pua_base:04X}..U+{pua_base + tokens_len - 1:04X})")

    failures = 0

    if args.dataset:
        dataset_dir = Path(args.dataset).expanduser().resolve()
        meta = dataset_dir / "metadata.csv"
        if not meta.exists():
            raise SystemExit(f"--dataset missing metadata.csv: {meta}")
        counts = scan_metadata_pua_counts(meta, tokens_len=tokens_len, pua_base=pua_base)
        covered = sum(1 for c in counts if c > 0)
        missing_ids = [i for i, c in enumerate(counts) if c == 0]
        print(f"dataset={dataset_dir} coveredTokens={covered}/{tokens_len} coverage={covered/tokens_len:.2%}")
        if missing_ids:
            failures += 1
            preview = ", ".join(f"{i}:{tokens[i]}" for i in missing_ids[:20])
            print(f"dataset_missing_token_ids(count={len(missing_ids)}) first20={preview}")

    if args.config:
        config_path = Path(args.config).expanduser().resolve()
        if not config_path.exists():
            raise SystemExit(f"--config missing: {config_path}")
        cfg = json.loads(config_path.read_text(encoding="utf-8"))
        chars = cfg.get("characters", {}).get("characters", "")
        if not isinstance(chars, str):
            raise SystemExit("config.json: characters.characters must be a string")
        present = set(chars)
        missing = sorted(required - present, key=lambda c: ord(c))
        print(f"config={config_path} presentPUA={sum(1 for c in present if pua_base <= ord(c) < pua_base + tokens_len)}/{tokens_len}")
        if missing:
            failures += 1
            preview = ", ".join(f"U+{ord(c):04X}" for c in missing[:20])
            print(f"config_missing_pua_tokens(count={len(missing)}) first20={preview}")

    if args.voicepack:
        vp = Path(args.voicepack).expanduser().resolve()
        tokens_path = find_voicepack_tokens_file(vp)
        vp_tokens = parse_tokens_txt(tokens_path)
        missing = sorted(required - vp_tokens, key=lambda c: ord(c))
        print(f"voicepack={vp} tokens_txt={tokens_path} tokens={len(vp_tokens)}")
        if missing:
            failures += 1
            missing_ids = [ord(c) - pua_base for c in missing]
            preview = ", ".join(f"{i}:{tokens[i]}" for i in missing_ids[:20])
            print(f"voicepack_missing_pua_tokens(count={len(missing_ids)}) first20={preview}")

    if failures and args.strict:
        raise SystemExit(2)
    if failures:
        print(f"⚠️  Verification found issues: failures={failures} (re-run with --strict to fail)")
    else:
        print("✅ Verification OK")


if __name__ == "__main__":
    main()
