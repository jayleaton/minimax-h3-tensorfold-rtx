r"""Where things are, from the environment (no machine-specific defaults).

COMFYUI_PORTABLE   the ComfyUI Windows portable folder (holds python_embeded\ and ComfyUI\)
MINIMAX_H3_DIT     the MiniMax H3 DiT checkpoint (default: <ComfyUI>\models\diffusion_models\minimax_h3_fl2va_pruned_int8_convrot.safetensors)
COMFYUI_PYTHON     the Python that runs ComfyUI (default: <COMFYUI_PORTABLE>\python_embeded\python.exe); for a git checkout
                   of ComfyUI with its own venv, point COMFYUI_PORTABLE at the folder holding the ComfyUI checkout
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DIT_NAME = "minimax_h3_fl2va_pruned_int8_convrot.safetensors"


def portable() -> Path:
    p = os.environ.get("COMFYUI_PORTABLE")
    if not p:
        sys.exit("set COMFYUI_PORTABLE to your ComfyUI portable folder (the one holding python_embeded and ComfyUI)")
    return Path(p)


def comfy() -> Path:
    return portable() / "ComfyUI"


def python() -> Path:
    return Path(os.environ.get("COMFYUI_PYTHON") or portable() / "python_embeded" / "python.exe")


def dit() -> str:
    return os.environ.get("MINIMAX_H3_DIT") or str(comfy() / "models" / "diffusion_models" / DIT_NAME)
