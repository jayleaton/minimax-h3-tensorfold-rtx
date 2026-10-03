"""Fused elementwise kernels for a MiniMax H3 block (Triton): each reads its input once and writes its output once.

H3 modulates per token: a row's shift / scale / gate come from one of a few modulation rows (timestep class x
modality tag), so every kernel takes ``idx`` (int32 [S], the row's modulation row) and the block's modulation table
``mod`` (fp32 [R, 6, D]: shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp). fp32 inside, bf16 out.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _norm_mod_kernel(x_ptr, w_ptr, mod_ptr, idx_ptr, o_ptr, D: tl.constexpr, shift_off, scale_off, mod_row, eps,
                     BLOCK: tl.constexpr):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < D
    x = tl.load(x_ptr + row.to(tl.int64) * D + cols, mask=mask, other=0.0).to(tl.float32)
    rr = tl.rsqrt(tl.sum(x * x, axis=0) / D + eps)
    w = tl.load(w_ptr + cols, mask=mask, other=0.0)
    r = tl.load(idx_ptr + row).to(tl.int64) * mod_row
    shift = tl.load(mod_ptr + r + shift_off + cols, mask=mask, other=0.0)
    scale = tl.load(mod_ptr + r + scale_off + cols, mask=mask, other=0.0)
    y = (x * rr * w) * (1.0 + scale) + shift
    tl.store(o_ptr + row.to(tl.int64) * D + cols, y.to(tl.bfloat16), mask=mask)


def norm_mod(x: torch.Tensor, w: torch.Tensor, mod: torch.Tensor, part: int, idx: torch.Tensor, eps: float,
             out: torch.Tensor | None = None) -> torch.Tensor:
    """RMSNorm(x) * w * (1 + scale) + shift for x [S, D] bf16; ``part`` 0 = attention (chunks 0, 1), 1 = MLP (3, 4)."""

    S, D = x.shape
    out = torch.empty_like(x) if out is None else out
    base = 3 * part
    _norm_mod_kernel[(S,)](x, w, mod, idx, out, D, base * D, (base + 1) * D, 6 * D, eps,
                           BLOCK=triton.next_power_of_2(D), num_warps=8)
    return out


@triton.jit
def _gate_add_kernel(x_ptr, y_ptr, mod_ptr, idx_ptr, D: tl.constexpr, gate_off, mod_row, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    cols = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    mask = cols < D
    r = tl.load(idx_ptr + row).to(tl.int64) * mod_row
    g = tl.load(mod_ptr + r + gate_off + cols, mask=mask, other=0.0)
    off = row.to(tl.int64) * D + cols
    x = tl.load(x_ptr + off, mask=mask, other=0.0).to(tl.float32)
    y = tl.load(y_ptr + off, mask=mask, other=0.0).to(tl.float32)
    tl.store(x_ptr + off, (x + y * g).to(tl.bfloat16), mask=mask)


def gate_add(x: torch.Tensor, y: torch.Tensor, mod: torch.Tensor, part: int, idx: torch.Tensor) -> torch.Tensor:
    """x += y * gate in place (x, y [S, D] bf16); ``part`` 0 = gate_msa (chunk 2), 1 = gate_mlp (chunk 5)."""

    S, D = x.shape
    BLOCK = 2048
    _gate_add_kernel[(S, triton.cdiv(D, BLOCK))](x, y, mod, idx, D, (3 * part + 2) * D, 6 * D, BLOCK=BLOCK,
                                                  num_warps=4)
    return x


@triton.jit
def _swiglu_kernel(gu_ptr, o_ptr, F: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    cols = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    mask = cols < F
    g = tl.load(gu_ptr + row.to(tl.int64) * 2 * F + cols, mask=mask, other=0.0).to(tl.float32)
    u = tl.load(gu_ptr + row.to(tl.int64) * 2 * F + F + cols, mask=mask, other=0.0).to(tl.float32)
    tl.store(o_ptr + row.to(tl.int64) * F + cols, (g / (1.0 + tl.exp(-g)) * u).to(tl.bfloat16), mask=mask)


def swiglu(gu: torch.Tensor, out: torch.Tensor | None = None) -> torch.Tensor:
    """[M, 2F] = [gate | up] bf16 -> silu(gate) * up, [M, F] bf16."""

    M, F2 = gu.shape
    F = F2 // 2
    out = torch.empty((M, F), dtype=torch.bfloat16, device=gu.device) if out is None else out
    BLOCK = 2048
    _swiglu_kernel[(M, triton.cdiv(F, BLOCK))](gu, out, F, BLOCK=BLOCK, num_warps=8)
    return out
