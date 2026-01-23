#!/usr/bin/env python3
"""
Generate a Coqui TTS VITS config.json for training a WordSuggestor "WS PUA phonemes" voice pack.

We train a *character-based* VITS model where the "characters" are Unicode Private Use Area (PUA)
symbols representing phoneme token IDs.

The dataset metadata is LJSpeech-style:
  metadata.csv rows: id|original_text|pua_text
  wavs/<id>.wav

Important stability detail:
- We disable "language embeddings" because this model is single-language and the language-embedding
  code path has caused crashes on Apple Silicon when experimenting with MPS/Accelerate.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def collect_unique_chars_from_metadata(metadata_csv: Path) -> str:
    chars: set[str] = set()
    for line in metadata_csv.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        parts = line.split("|")
        if len(parts) < 3:
            continue
        pua_text = parts[2]
        for ch in pua_text:
            if ch in ("\n", "\r"):
                continue
            chars.add(ch)
    return "".join(sorted(chars, key=lambda c: ord(c)))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, help="LJSpeech-style folder (metadata.csv + wavs/)")
    ap.add_argument("--out-dir", required=True, help="Output directory (writes config.json here)")
    ap.add_argument("--run-name", required=True, help="Coqui run_name")
    ap.add_argument(
        "--base-config",
        default="baselines/vits-coqui-da-cv/config.json",
        help="Baseline config.json to copy hyperparameters from",
    )
    ap.add_argument("--language", default="da", help="Language code for dataset entry (e.g. da)")
    ap.add_argument("--sample-rate", type=int, default=22050)
    ap.add_argument(
        "--max-audio-sec",
        type=float,
        default=20.0,
        help="Max audio duration (seconds) to keep. Larger keeps longer utterances but may reduce stability.",
    )
    ap.add_argument(
        "--min-audio-sec",
        type=float,
        default=1.5,
        help="Min audio duration (seconds) to keep. Prevents too-short audio that can break VITS alignment.",
    )
    ap.add_argument(
        "--max-text-len",
        type=int,
        default=120,
        help="Max text length (characters) to keep. For PUA text, this bounds alignment complexity and prevents NaNs.",
    )
    ap.add_argument(
        "--min-text-len",
        type=int,
        default=1,
        help="Min text length (characters) to keep.",
    )
    ap.add_argument("--epochs", type=int, default=500)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--eval-batch-size", type=int, default=4)
    ap.add_argument(
        "--disable-eval",
        action="store_true",
        help="Disable Coqui evaluation during training (workaround for occasional MPS/Accelerate eval crashes).",
    )
    ap.add_argument("--mixed-precision", action="store_true")
    ap.add_argument("--num-loader-workers", type=int, default=0)
    ap.add_argument("--num-eval-loader-workers", type=int, default=0)
    args = ap.parse_args()

    dataset = Path(args.dataset).expanduser().resolve()
    out_dir = Path(args.out_dir).expanduser().resolve()
    base_cfg_path = Path(args.base_config).expanduser().resolve()

    metadata_csv = dataset / "metadata.csv"
    wavs_dir = dataset / "wavs"
    if not metadata_csv.exists():
        raise SystemExit(f"Missing metadata.csv: {metadata_csv}")
    if not wavs_dir.exists():
        raise SystemExit(f"Missing wavs/: {wavs_dir}")
    if not base_cfg_path.exists():
        raise SystemExit(f"Missing base config: {base_cfg_path}")

    chars_sorted = collect_unique_chars_from_metadata(metadata_csv)
    if not chars_sorted:
        raise SystemExit(f"No characters found in {metadata_csv}")

    cfg = json.loads(base_cfg_path.read_text(encoding="utf-8"))

    cfg["output_path"] = str(out_dir)
    cfg["run_name"] = args.run_name
    cfg["project_name"] = "wordsuggestor-tts"
    cfg["epochs"] = int(args.epochs)
    cfg["batch_size"] = int(args.batch_size)
    cfg["eval_batch_size"] = int(args.eval_batch_size)
    cfg["num_loader_workers"] = int(args.num_loader_workers)
    cfg["num_eval_loader_workers"] = int(args.num_eval_loader_workers)
    cfg["mixed_precision"] = bool(args.mixed_precision)

    # Disable language embedding plumbing (single-language model; avoids MPS/Accelerate crashes).
    cfg["use_language_embedding"] = False
    cfg["model_args"]["use_language_embedding"] = False
    cfg["language_ids_file"] = ""

    # Evaluation: On Apple Silicon with MPS + HF Accelerate we have seen intermittent failures
    # during the eval loop (runtime: "Placeholder storage has not been allocated on MPS device!").
    # Disabling eval keeps long training runs alive; we can still export checkpoints and listen
    # to samples periodically.
    if bool(args.disable_eval):
        cfg["run_eval"] = False
        cfg["print_eval"] = False

    cfg["audio"]["sample_rate"] = int(args.sample_rate)
    cfg["audio"]["resample"] = False

    cfg["min_audio_len"] = int(max(0.0, float(args.min_audio_sec)) * args.sample_rate)
    cfg["max_audio_len"] = int(max(1.0, float(args.max_audio_sec)) * args.sample_rate)
    cfg["min_text_len"] = int(args.min_text_len)
    cfg["max_text_len"] = int(args.max_text_len)

    cfg["sort_by_audio_len"] = True
    cfg["batch_group_size"] = 16

    cfg["datasets"] = [
        {
            "formatter": "ljspeech",
            "dataset_name": args.run_name,
            "path": str(dataset),
            "meta_file_train": "metadata.csv",
            "ignored_speakers": [],
            "language": str(args.language),
            "meta_file_val": "",
            "meta_file_attn_mask": "",
        }
    ]

    cfg["use_phonemes"] = False
    cfg["phonemizer"] = None
    cfg["phoneme_language"] = None
    cfg["text_cleaner"] = "no_cleaners"
    cfg["enable_eos_bos_chars"] = False

    cfg["characters"]["characters_class"] = "TTS.tts.utils.text.characters.Graphemes"
    cfg["characters"]["characters"] = chars_sorted
    cfg["characters"]["punctuations"] = ""
    cfg["characters"]["phonemes"] = None
    cfg["characters"]["is_unique"] = True
    cfg["characters"]["is_sorted"] = True

    cfg["model_args"]["num_chars"] = 4 + len(chars_sorted)

    # Coqui VITS training expects a discriminator to be present for GAN losses.
    cfg["model_args"]["init_discriminator"] = True

    # Single-speaker pack by default.
    cfg["model_args"]["use_speaker_embedding"] = False
    cfg["model_args"]["num_speakers"] = 0
    cfg["model_args"]["speakers_file"] = ""

    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "config.json"
    out_path.write_text(json.dumps(cfg, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    print("✅ Wrote Coqui VITS config")
    print(f"config={out_path}")
    print(f"dataset={dataset}")
    print(f"unique_chars={len(chars_sorted)} num_chars={cfg['model_args']['num_chars']}")


if __name__ == "__main__":
    main()
