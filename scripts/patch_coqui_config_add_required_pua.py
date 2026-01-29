#!/usr/bin/env python3
"""
Patch a Coqui VITS config.json (created by generate_coqui_vits_config_ws_pua.py) so its
character set includes *all* required PUA scalars for the current WordSuggestor G2P vocab.

This prevents the training tokenizer (and thus exported tokens.txt) from missing phoneme tokens
that simply didn't appear in the training split.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, help="Path to Coqui config.json to patch (in-place)")
    ap.add_argument("--g2p-dir", required=True, help="Directory containing vocab.json")
    ap.add_argument("--pua-base", type=lambda s: int(s, 0), default=0xE000)
    args = ap.parse_args()

    config_path = Path(args.config).expanduser().resolve()
    g2p_dir = Path(args.g2p_dir).expanduser().resolve()
    vocab_path = g2p_dir / "vocab.json"

    if not config_path.exists():
        raise SystemExit(f"Missing config: {config_path}")
    if not vocab_path.exists():
        raise SystemExit(f"Missing vocab.json: {vocab_path}")

    cfg = json.loads(config_path.read_text(encoding="utf-8"))
    vocab = json.loads(vocab_path.read_text(encoding="utf-8"))

    tokens = vocab["tokens"]
    required = {chr(int(args.pua_base) + int(i)) for i in range(len(tokens))}

    current_chars = cfg.get("characters", {}).get("characters", "")
    if not isinstance(current_chars, str):
        raise SystemExit("config.json: characters.characters must be a string")

    current_set = {c for c in current_chars if c not in ("\n", "\r")}
    merged = sorted(current_set | required, key=lambda c: ord(c))
    merged_str = "".join(merged)

    cfg.setdefault("characters", {})
    cfg["characters"]["characters"] = merged_str
    cfg["characters"]["is_unique"] = True
    cfg["characters"]["is_sorted"] = True

    cfg.setdefault("model_args", {})
    cfg["model_args"]["num_chars"] = 4 + len(merged_str)

    missing_after = sorted(required - set(merged_str), key=lambda c: ord(c))
    if missing_after:
        missing_hex = ", ".join(f"U+{ord(c):04X}" for c in missing_after[:20])
        raise SystemExit(f"BUG: still missing required PUA characters after patch: {missing_hex}")

    config_path.write_text(json.dumps(cfg, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    added = len(required - current_set)
    print("✅ Patched Coqui config characters")
    print(f"config={config_path}")
    print(f"g2p_vocab={vocab_path} tokens={len(tokens)} puaBase=0x{int(args.pua_base):X}")
    print(f"chars_before={len(current_set)} chars_after={len(merged_str)} added_required={added}")


if __name__ == "__main__":
    main()

