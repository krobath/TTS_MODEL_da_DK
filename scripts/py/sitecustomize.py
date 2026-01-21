"""
TTS_MODEL training helpers.

This `sitecustomize.py` is imported automatically by Python when present on `sys.path`.
We use it to patch `torchaudio.load()` during Coqui TTS training so we don't depend on
TorchCodec/FFmpeg (TorchAudio 2.9+ defaults to TorchCodec for decoding).

Why:
- In offline/restricted environments, installing FFmpeg into every training env is painful.
- Our prepared datasets use plain WAV files; `soundfile` can read them reliably.

This patch is only intended to be activated by our training wrapper script, which sets:
  PYTHONPATH=$REPO_ROOT/scripts/py:$PYTHONPATH
"""

from __future__ import annotations

import io
import os
from typing import BinaryIO, Optional, Tuple, Union


def _maybe_log(msg: str) -> None:
    if os.environ.get("WS_TTS_DEBUG") == "1":
        print(f"[TTS_MODEL][WS_TTS] {msg}")


def _patch_torchaudio_load() -> None:
    try:
        import torch
        import torchaudio
    except Exception:
        return

    # Avoid double-patching.
    if getattr(torchaudio.load, "__name__", "") == "_ws_soundfile_load":
        return

    try:
        import numpy as np
        import soundfile as sf
    except Exception:
        # If soundfile isn't installed, do nothing and let torchaudio fail normally.
        return

    _maybe_log("Patching torchaudio.load -> soundfile backend (avoids torchcodec/ffmpeg).")

    def _ws_soundfile_load(
        uri: Union[BinaryIO, str, os.PathLike],
        frame_offset: int = 0,
        num_frames: int = -1,
        normalize: bool = True,  # kept for signature compatibility
        channels_first: bool = True,
        format: Optional[str] = None,  # unused by soundfile unless reading bytes; keep for API compatibility
        buffer_size: int = 4096,  # unused; API compatibility
        backend: Optional[str] = None,  # unused; API compatibility
    ) -> Tuple["torch.Tensor", int]:
        # Coqui training passes file paths. Support file-like for completeness.
        if isinstance(uri, (str, os.PathLike)):
            data, sr = sf.read(str(uri), dtype="float32", always_2d=True)
        else:
            raw = uri.read()
            data, sr = sf.read(io.BytesIO(raw), dtype="float32", always_2d=True)

        # data: (time, channels)
        if frame_offset:
            data = data[int(frame_offset) :]
        if num_frames is not None and int(num_frames) > 0:
            data = data[: int(num_frames)]

        # TorchAudio returns (channels, time) by default.
        if channels_first:
            data = np.transpose(data, (1, 0))

        tensor = torch.from_numpy(data)
        return tensor, int(sr)

    torchaudio.load = _ws_soundfile_load  # type: ignore[assignment]


_patch_torchaudio_load()


def _set_torch_sharing_strategy() -> None:
    """
    Prefer `file_system` tensor sharing to avoid `torch_shm_manager ... Operation not permitted`
    when DataLoader uses multiple worker processes on some macOS setups.
    """

    try:
        import torch.multiprocessing as mp  # type: ignore
    except Exception:
        return

    try:
        mp.set_sharing_strategy("file_system")
        _maybe_log("Set torch.multiprocessing sharing strategy=file_system")
    except Exception as e:
        _maybe_log(f"Failed to set torch sharing strategy: {e}")


_set_torch_sharing_strategy()


def _configure_torch_threads() -> None:
    """
    Allow the shell wrapper to tune PyTorch CPU thread pools.

    Usage:
      WS_TORCH_NUM_THREADS=16 WS_TORCH_NUM_INTEROP_THREADS=4 ./scripts/train_da_coral_ws_pua_vits.sh ...
    """

    try:
        import torch  # type: ignore
    except Exception:
        return

    num_threads = os.environ.get("WS_TORCH_NUM_THREADS")
    if num_threads:
        try:
            torch.set_num_threads(int(num_threads))
            _maybe_log(f"torch.set_num_threads({num_threads})")
        except Exception as e:
            _maybe_log(f"Failed to set torch num threads: {e}")

    num_interop = os.environ.get("WS_TORCH_NUM_INTEROP_THREADS")
    if num_interop:
        try:
            torch.set_num_interop_threads(int(num_interop))
            _maybe_log(f"torch.set_num_interop_threads({num_interop})")
        except Exception as e:
            _maybe_log(f"Failed to set torch interop threads: {e}")


_configure_torch_threads()

