"""Attention backends for the full (non-causal) packed sequence: q, k, v [1, H, S, D] bf16 (strided views are fine)
-> [1, H, S, D]. At 40k tokens attention is most of an H3 step, so the backend is the main speed knob.

sdpa / cudnn   bf16 tensor cores (PyTorch flash, or cuDNN)
int8           comfy-kitchen's INT8 SageAttention-style kernel (Q/K per block, V per channel)
fp8            tfvideo.fp4attn: Q/K FP8 per row (K smoothed), P and V FP8
fp4            tfvideo.fp4attn: Q/K NVFP4 (block-scaled FP4 tensor cores), P and V FP8
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel


def sdpa_flash(q, k, v):
    with sdpa_kernel([SDPBackend.FLASH_ATTENTION, SDPBackend.CUDNN_ATTENTION, SDPBackend.EFFICIENT_ATTENTION],
                     set_priority=True):
        return F.scaled_dot_product_attention(q, k, v)


def sdpa_cudnn(q, k, v):
    with sdpa_kernel([SDPBackend.CUDNN_ATTENTION, SDPBackend.FLASH_ATTENTION, SDPBackend.EFFICIENT_ATTENTION],
                     set_priority=True):
        return F.scaled_dot_product_attention(q, k, v)


def int8(q, k, v):
    from comfy_kitchen.sage_attention import int8_attention
    return int8_attention(q, k, v)


def fp8(q, k, v):
    from . import fp4attn
    return fp4attn.attention(q, k, v, qk="fp8", pv="fp8")


def fp4(q, k, v):
    from . import fp4attn
    return fp4attn.attention(q, k, v, qk="fp4", pv="fp8")


BACKENDS = {"sdpa": sdpa_flash, "cudnn": sdpa_cudnn, "int8": int8, "fp8": fp8, "fp4": fp4}
BENCH: dict = {}                 # extra kernels for tools/attn_bench.py
DEFAULT = "int8"


def get(name: str):
    if name == "auto":
        name = DEFAULT
    if name not in BACKENDS:
        raise ValueError(f"attention backend {name!r}: one of {', '.join(['auto', *BACKENDS])}")
    return BACKENDS[name]
