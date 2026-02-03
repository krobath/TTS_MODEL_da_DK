#!/usr/bin/env python3
"""
Extract top-N Danish words by frequency from the WordSuggestor lexicon SQLite.

This is intended to build a reproducible wordlist for G2P parity checks (ONNX vs CoreML),
so we can fail early before training a WS-PUA voice for days.

Input DB schema (expected):
  terms(term_norm TEXT PRIMARY KEY, display TEXT, ..., freq INTEGER, ...)

Examples:
  python TTS_MODEL_da_DK/scripts/extract_top_words_from_lexicon_sqlite.py --n 2000
  python TTS_MODEL_da_DK/scripts/extract_top_words_from_lexicon_sqlite.py --db WordSuggestor/da_lexicon.sqlite --n 2000 --out-dir TTS_MODEL_da_DK/wordlists
"""

from __future__ import annotations

import argparse
import sqlite3
from pathlib import Path
from typing import Iterable, List, Tuple


def resolve_default_db() -> Path:
    candidates = [
        Path("WordSuggestor/da_lexicon.sqlite"),
        Path("WordSuggestor/Ressources/da_lexicon.sqlite"),
        Path("WordSuggestorCore/Ressources/da_lexicon.sqlite"),
        Path("../WordSuggestor/da_lexicon.sqlite"),
        Path("../WordSuggestorCore/Ressources/da_lexicon.sqlite"),
    ]
    for c in candidates:
        if c.exists():
            return c.resolve()
    raise SystemExit(
        "Could not find da_lexicon.sqlite. Pass --db, e.g. --db WordSuggestor/da_lexicon.sqlite"
    )


def _is_word_only(term: str) -> bool:
    # Accept only Unicode letters (no spaces, punctuation, hyphens, apostrophes, digits).
    return bool(term) and all(ch.isalpha() for ch in term)


def fetch_top_terms(db_path: Path, limit: int) -> List[Tuple[str, str, int]]:
    con = sqlite3.connect(str(db_path))
    try:
        cur = con.cursor()
        cur.execute(
            """
            SELECT term_norm, display, freq
            FROM terms
            ORDER BY freq DESC, term_norm ASC
            LIMIT ?
            """,
            (int(limit),),
        )
        rows = cur.fetchall()
        out: List[Tuple[str, str, int]] = []
        for term_norm, display, freq in rows:
            out.append((str(term_norm), str(display), int(freq)))
        return out
    finally:
        con.close()


def write_files(rows: Iterable[Tuple[str, str, int]], out_dir: Path, n: int) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    txt_path = out_dir / f"da_top{n}_display.txt"
    tsv_path = out_dir / f"da_top{n}_terms.tsv"

    rows_list = list(rows)

    # For parity checks, we want the exact surface characters (incl. æ/ø/å) but
    # case-insensitive comparisons. The G2P encoders lower-case anyway.
    txt_path.write_text(
        "\n".join(display.casefold() for _, display, _ in rows_list) + "\n",
        encoding="utf-8",
    )
    tsv_path.write_text(
        "rank\tterm_norm\tdisplay\tfreq\n"
        + "\n".join(
            f"{i}\t{term}\t{display}\t{freq}"
            for i, (term, display, freq) in enumerate(rows_list, start=1)
        )
        + "\n",
        encoding="utf-8",
    )

    print("== Top words extracted ==")
    print(f"rows={len(rows_list)}")
    print(f"txt={txt_path}")
    print(f"tsv={tsv_path}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default="", help="Path to da_lexicon.sqlite (defaults to repo-known locations)")
    ap.add_argument("--n", type=int, default=2000, help="Number of terms to export")
    ap.add_argument(
        "--word-only",
        action="store_true",
        help="Filter to terms consisting of letters only (recommended for G2P tests)",
    )
    ap.add_argument("--raw", action="store_true", help="Also write an unfiltered (raw) top-N list")
    ap.add_argument(
        "--out-dir",
        default="TTS_MODEL_da_DK/wordlists",
        help="Output directory (txt + tsv will be created here)",
    )
    args = ap.parse_args()

    db_path = Path(args.db).expanduser().resolve() if args.db else resolve_default_db()
    if not db_path.exists():
        raise SystemExit(f"DB not found: {db_path}")

    out_dir = Path(args.out_dir).expanduser().resolve()
    target_n = int(args.n)

    # Fetch more than N, then filter down to exactly N.
    fetch_limit = max(target_n, 1000)
    rows = fetch_top_terms(db_path, limit=fetch_limit)

    if args.word_only:
        filtered: List[Tuple[str, str, int]] = []
        seen = set()
        while True:
            for term, display, freq in rows:
                if term in seen:
                    continue
                seen.add(term)
                # Filter on the actual surface form we will use for tests.
                if not _is_word_only(display):
                    continue
                filtered.append((term, display, freq))
                if len(filtered) >= target_n:
                    break
            if len(filtered) >= target_n:
                rows = filtered[:target_n]
                break

            # Not enough: fetch more.
            fetch_limit *= 2
            more = fetch_top_terms(db_path, limit=fetch_limit)
            if len(more) == len(rows):
                # DB exhausted (shouldn't happen for Danish).
                rows = filtered
                break
            rows = more

    write_files(rows, out_dir=out_dir, n=target_n)

    if args.raw:
        raw_rows = fetch_top_terms(db_path, limit=target_n)
        raw_out_dir = out_dir / "raw"
        write_files(raw_rows, out_dir=raw_out_dir, n=target_n)


if __name__ == "__main__":
    main()
