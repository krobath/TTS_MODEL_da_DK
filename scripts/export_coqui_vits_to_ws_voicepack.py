#!/usr/bin/env python3
"""
Export a trained Coqui VITS checkpoint into a WordSuggestor voice pack folder compatible with
the sherpa-onnx offline TTS runtime.

Dependencies:
  python -m pip install TTS onnx onnxscript

Note:
  Newer PyTorch versions require `onnxscript` during `torch.onnx.export()`. If it's missing,
  Coqui's `vits.export_onnx()` will fail even if `onnx` itself is installed.
"""

from __future__ import annotations

import argparse
import collections
import json
import os
import inspect
from pathlib import Path
from typing import Any, Dict

import onnx
import torch
from TTS.tts.configs.vits_config import VitsConfig
from TTS.tts.models.vits import Vits


LANG_MAP = {
    "da": "Danish",
    "sv": "Swedish",
    "de": "German",
    "fr": "French",
    "es": "Spanish",
    "it": "Italian",
    "nb": "Norwegian",
    "nn": "Norwegian",
}


def add_meta_data(filename: Path, meta_data: Dict[str, Any]) -> None:
    model = onnx.load(str(filename))
    for key, value in meta_data.items():
        meta = model.metadata_props.add()
        meta.key = key
        meta.value = str(value)
    onnx.save(model, str(filename))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, help="Coqui VITS config.json")
    ap.add_argument("--checkpoint", required=True, help="Checkpoint .pth file")
    ap.add_argument("--out-root", required=True, help="Destination root (e.g. tts-voices/)")
    ap.add_argument("--pack-name", required=True, help="Voice pack directory name")
    ap.add_argument("--language", required=True, help="Language code for metadata (e.g. da)")
    ap.add_argument("--speaker-label", default="speaker0", help="Speaker label for speaker_ids.json")
    ap.add_argument("--ws-frontend", default="", help="Optional WordSuggestor frontend id (e.g. ws_pua_phonemes_v1)")
    ap.add_argument("--ws-pua-base", default="", help="Optional puaBase (int or hex, e.g. 0xE000)")
    ap.add_argument("--ws-g2p-language", default="", help="Optional g2pLanguageCode (e.g. da-DK)")
    ap.add_argument(
        "--include-checkpoint",
        action="store_true",
        help="Also copy the training checkpoint into the voice pack (big; not needed for runtime).",
    )
    args = ap.parse_args()

    config_path = Path(args.config).expanduser().resolve()
    ckpt_path = Path(args.checkpoint).expanduser().resolve()
    out_root = Path(args.out_root).expanduser().resolve()

    if not config_path.exists():
        raise SystemExit(f"Missing config: {config_path}")
    if not ckpt_path.exists():
        raise SystemExit(f"Missing checkpoint: {ckpt_path}")

    try:
        import onnxscript  # noqa: F401
    except ModuleNotFoundError as exc:
        raise SystemExit(
            "Missing Python dependency: onnxscript\n"
            "Install it in your active environment and retry:\n"
            "  python -m pip install onnxscript\n"
        ) from exc

    pack_outer = out_root / args.pack_name
    pack_inner = pack_outer / args.pack_name
    pack_inner.mkdir(parents=True, exist_ok=True)

    (pack_inner / "config.json").write_bytes(config_path.read_bytes())
    if args.include_checkpoint:
        (pack_inner / "model_file.pth").write_bytes(ckpt_path.read_bytes())

    cfg = VitsConfig()
    cfg.load_json(str(config_path))
    vits = Vits.init_from_config(cfg)
    vits.load_checkpoint(cfg, str(ckpt_path))
    onnx_path = pack_inner / "model.onnx"

    # Coqui calls into `torch.onnx.export()`. Newer PyTorch defaults to the dynamo/torch.export-based
    # ONNX exporter, which often fails on VITS because it contains data-dependent control flow
    # (e.g. `if torch.min(inputs) < left ...` in rational_quadratic_spline).
    #
    # We prefer the legacy tracer-based exporter here, which is sufficient for our runtime use.
    os.environ.setdefault("TORCH_ONNX_USE_EXPERIMENTAL_EXPORTER", "0")
    os.environ.setdefault("TORCH_ONNX_USE_DYNAMO_EXPORT", "0")

    vits.eval()
    vits.to("cpu")

    orig_export = torch.onnx.export
    try:
        try:
            sig = inspect.signature(torch.onnx.export)
            supports_dynamo_flag = "dynamo" in sig.parameters
        except Exception:
            supports_dynamo_flag = False

        if supports_dynamo_flag:
            # Force legacy exporter by default for any internal calls.
            def export_no_dynamo(*args: Any, **kwargs: Any) -> Any:
                kwargs.setdefault("dynamo", False)
                return orig_export(*args, **kwargs)

            torch.onnx.export = export_no_dynamo  # type: ignore[assignment]

        vits.export_onnx(output_path=str(onnx_path), verbose=False)
    finally:
        torch.onnx.export = orig_export  # type: ignore[assignment]

    language = LANG_MAP.get(args.language, args.language)
    meta_data: Dict[str, Any] = {
        "model_type": "vits",
        "comment": "coqui",
        "language": language,
        "frontend": "characters",
        "add_blank": int(vits.config.add_blank),
        "blank_id": vits.tokenizer.characters.blank_id,
        "n_speakers": vits.config.model_args.num_speakers,
        "use_eos_bos": int(vits.tokenizer.use_eos_bos),
        "bos_id": vits.tokenizer.characters.bos_id,
        "eos_id": vits.tokenizer.characters.eos_id,
        "pad_id": vits.tokenizer.characters.pad_id,
        "sample_rate": int(vits.ap.sample_rate),
    }
    if args.ws_frontend:
        meta_data["ws_frontend"] = args.ws_frontend
    add_meta_data(onnx_path, meta_data)

    chars = vits.tokenizer.characters._char_to_id
    all_upper_tokens = [i.upper() for i in chars.keys()]
    duplicate = set([item for item, count in collections.Counter(all_upper_tokens).items() if count > 1])
    with (pack_inner / "tokens.txt").open("w", encoding="utf-8") as f:
        for token, idx in chars.items():
            f.write(f"{token} {idx}\n")
            if (
                token not in ("<PAD>", "<EOS>", "<BOS>", "<BLNK>")
                and token.lower() != token.upper()
                and len(token.upper()) == 1
                and token.upper() not in duplicate
            ):
                f.write(f"{token.upper()} {idx}\n")

    (pack_inner / "speaker_ids.json").write_text(json.dumps({args.speaker_label: 0}, indent=2) + "\n", encoding="utf-8")
    (pack_inner / "language_ids.json").write_text(json.dumps({args.language: 0}, indent=2) + "\n", encoding="utf-8")

    if args.ws_frontend:
        ws: Dict[str, Any] = {"schema": 1, "frontend": args.ws_frontend}
        if args.ws_pua_base:
            ws["puaBase"] = int(args.ws_pua_base, 0)
        if args.ws_g2p_language:
            ws["g2pLanguageCode"] = args.ws_g2p_language
        (pack_inner / "ws_voicepack.json").write_text(json.dumps(ws, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    print("✅ Exported voice pack:")
    print(f"pack={args.pack_name}")
    print(f"dir={pack_inner}")
    print(f"onnx={onnx_path}")
    print(f"tokens={pack_inner / 'tokens.txt'}")


if __name__ == "__main__":
    main()
