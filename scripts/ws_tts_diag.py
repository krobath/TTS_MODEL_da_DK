#!/usr/bin/env python3
"""
TTS_MODEL training environment diagnostics.

Run this inside your training environment (e.g. `conda activate ws-tts`) to see whether PyTorch
can use Apple GPU acceleration (MPS) and what thread settings are active.
"""

from __future__ import annotations

import platform
import subprocess
import sys


def _sysctl_bool(name: str) -> bool | None:
    try:
        out = subprocess.check_output(["/usr/sbin/sysctl", "-n", name], text=True).strip()
        return out == "1"
    except Exception:
        return None


def main() -> None:
    print("== TTS_MODEL Diagnostics ==")
    print(f"python_exe={sys.executable}")
    print(f"python_ver={sys.version.splitlines()[0]}")
    print(f"platform={platform.platform()}")
    print(f"machine={platform.machine()}")
    translated = _sysctl_bool("sysctl.proc_translated")
    if translated is not None:
        print(f"rosetta_translated={translated}")

    try:
        import torch  # type: ignore

        print(f"torch={torch.__version__}")
        try:
            print(f"torch_num_threads={torch.get_num_threads()}")
            print(f"torch_num_interop_threads={torch.get_num_interop_threads()}")
        except Exception:
            pass

        try:
            mps_built = bool(torch.backends.mps.is_built())
            mps_avail = bool(torch.backends.mps.is_available())
            print(f"mps_built={mps_built} mps_available={mps_avail}")
        except Exception:
            pass

        try:
            print(f"cuda_available={bool(torch.cuda.is_available())}")
        except Exception:
            pass
    except Exception as e:
        print(f"torch_import_error={type(e).__name__}: {e}")

    try:
        import torchaudio  # type: ignore

        print(f"torchaudio={torchaudio.__version__}")
    except Exception as e:
        print(f"torchaudio_import_error={type(e).__name__}: {e}")


if __name__ == "__main__":
    main()

