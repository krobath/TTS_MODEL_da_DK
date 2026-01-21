#!/usr/bin/env python3
"""
Fix/normalize WS PUA dataset metadata into Coqui's LJSpeech format:

  id|original_text|pua_text

Coqui's LJSpeech dataset formatter expects:
- metadata.csv under dataset root
- wavs/<id>.wav for each row

This script is useful if metadata.csv was produced in an older layout (or contains `.wav` suffixes).
"""

from __future__ import annotations

import argparse
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("dataset_dir", help="Dataset folder containing metadata.csv and wavs/")
    args = ap.parse_args()

    ds = Path(args.dataset_dir).expanduser().resolve()
    meta = ds / "metadata.csv"
    if not meta.exists():
        raise SystemExit(f"Missing metadata.csv: {meta}")

    lines = meta.read_text(encoding="utf-8").splitlines()
    out_lines: list[str] = []
    for line in lines:
        if not line.strip():
            continue
        parts = line.split("|")
        if len(parts) < 3:
            continue

        # Most common broken layouts:
        # 1) wav|pua|original
        # 2) id|original|pua  (already correct)
        a, b, c = parts[0], parts[1], parts[2]

        # Strip `.wav` suffix if present.
        if a.lower().endswith(".wav"):
            a = a[:-4]

        # Heuristic: PUA text will contain private-use chars (>= 0xE000) frequently.
        def looks_like_pua(s: str) -> bool:
            return any(ord(ch) >= 0xE000 for ch in s)

        if looks_like_pua(b) and not looks_like_pua(c):
            # wav|pua|original  -> id|original|pua
            out_lines.append(f"{a}|{c}|{b}")
        else:
            # Assume already id|original|pua
            out_lines.append(f"{a}|{b}|{c}")

    backup = meta.with_suffix(".csv.bak")
    backup.write_text(meta.read_text(encoding="utf-8"), encoding="utf-8")
    meta.write_text("\n".join(out_lines) + "\n", encoding="utf-8")

    print(f"✅ Converted metadata.csv (backup at {backup}) rows={len(out_lines)}")


if __name__ == "__main__":
    main()

