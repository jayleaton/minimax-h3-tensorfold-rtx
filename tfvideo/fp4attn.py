"""Low-precision flash attention for sm_120 in Triton: QK^T on NVFP4 block-scaled tensor cores, PV on FP8 or NVFP4.

Full non-causal attention over the packed H3 sequence is ~2.3 PFLOP a forward at 40k tokens; bf16 tensor cores (fp32
accumulate) run ~85 TF/s on an RTX 5070 Ti, FP8 ~2x that and NVFP4 (``mma.sync ... kind::mxf4nvf4.block_scale``) ~4x
more again. The recipe follows SageAttention 2/3:

* prepare (one pass over q, k, v per call): K is smoothed (its per-channel mean over the sequence removed, which no
  softmax row can see); Q (with the softmax scale and log2 e folded in) and K are quantized to NVFP4 along the head
  dim: e2m1 codes, one e4m3 scale per 16 channels, and one fp32 scale per row so the e4m3 scales use their range. V is
  quantized per channel: FP8 e4m3 (``pv="fp8"``), or NVFP4 along the sequence (``pv="fp4"``, scales per 16 keys).
* attend: per 128-query tile, loop over 64-key tiles: S = dot_scaled(Q, K) * row scales, online softmax in fp32,
  P quantized in registers (FP8: P * 448; FP4: per 16-key group e4m3 scale over P * 2688), O += P @ V.
"""

from __future__ import annotations

import math

import torch
import triton
import triton.language as tl

LOG2E = 1.4426950408889634
P8 = 448.0              # FP8 P scale: P in [0, 1] -> [0, 448]
P4 = 448.0 * 6.0        # FP4 P first-level scale: a group's e4m3 scale can reach 448


# ---------------------------------------------------------------------------------------------------- e2m1 helpers
@triton.jit
def _e2m1x2(lo, hi):
    """Two fp32 tensors (already divided by their block scale) -> uint8 codes, ``lo`` in the low nibble (RN, satfinite)."""

    return tl.inline_asm_elementwise(
        asm="""
        {
        .reg .b8 b0, b1, b2, b3;
        cvt.rn.satfinite.e2m1x2.f32 b0, $5, $1;
        cvt.rn.satfinite.e2m1x2.f32 b1, $6, $2;
        cvt.rn.satfinite.e2m1x2.f32 b2, $7, $3;
        cvt.rn.satfinite.e2m1x2.f32 b3, $8, $4;
        mov.b32 $0, {b0, b1, b2, b3};
        }
        """,
        constraints="=r,r,r,r,r,r,r,r,r",
        args=[lo, hi],
        dtype=tl.uint8,
        is_pure=True,
        pack=4,
    )


@triton.jit
def _e2m1(v):
    """fp32 (already divided by its block scale) -> e2m1 code (uint8, low 4 bits): nearest, ties to even, saturating.

    Arithmetic instead of ``cvt.rn.satfinite.e2m1x2``: Triton 3.6 miscompiles some reshape/split layouts feeding a
    packed inline asm (and broadcasts of a reduction over a reshaped tile), so tiles are loaded as [rows, groups, 8]
    even / odd halves and never reshaped."""

    a = tl.abs(v)
    m = ((a > 0.25).to(tl.uint8) + (a >= 0.75).to(tl.uint8) + (a > 1.25).to(tl.uint8) + (a >= 1.75).to(tl.uint8)
         + (a > 2.5).to(tl.uint8) + (a >= 3.5).to(tl.uint8) + (a > 5.0).to(tl.uint8))
    return m | ((v < 0).to(tl.uint8) << 3)


@triton.jit
def _quant_pairs(xe, xo, G: tl.constexpr):
    """Even / odd halves [R, G, 8] fp32 of rows of 16*G values -> (codes [R, G, 8] uint8, e4m3 scales [R, G], row
    scale [R]). Codes hold x / row_scale / group_scale; row_scale = rowmax|x| / (6 * 448), so group scales are <= 448."""

    amax = tl.maximum(tl.max(tl.max(tl.abs(xe), axis=2), axis=1), tl.max(tl.max(tl.abs(xo), axis=2), axis=1))
    rs = tl.maximum(amax, 1e-30) / (6.0 * 448.0)
    inv_r = (1.0 / rs)[:, None, None]
    xe = xe * inv_r
    xo = xo * inv_r
    gs = (tl.maximum(tl.max(tl.abs(xe), axis=2), tl.max(tl.abs(xo), axis=2)) / 6.0).to(tl.float8e4nv)
    inv = (1.0 / tl.maximum(gs.to(tl.float32), 1e-30))[:, :, None]
    return _e2m1(xe * inv) | (_e2m1(xo * inv) << 4), gs, rs


# ---------------------------------------------------------------------------------------------------- prepare
@triton.jit
def _stats_kernel(k_ptr, v_ptr, kmean_ptr, vmax_ptr, S, sk_s, sk_h, sv_s, sv_h, D: tl.constexpr, BS: tl.constexpr):
    # per (token block, head): partial sums of k and max |v| per channel, accumulated atomically
    pid, h = tl.program_id(0), tl.program_id(1)
    rows = pid * BS + tl.arange(0, BS)
    cols = tl.arange(0, D)
    m = rows < S
    k = tl.load(k_ptr + h * sk_h + rows[:, None].to(tl.int64) * sk_s + cols[None, :], mask=m[:, None], other=0.0)
    v = tl.load(v_ptr + h * sv_h + rows[:, None].to(tl.int64) * sv_s + cols[None, :], mask=m[:, None], other=0.0)
    tl.atomic_add(kmean_ptr + h * D + cols, tl.sum(k.to(tl.float32), axis=0) / S)
    tl.atomic_max(vmax_ptr + h * D + cols, tl.max(tl.abs(v.to(tl.float32)), axis=0))


@triton.jit
def _quant_qk_kernel(x_ptr, mean_ptr, codes_ptr, scales_ptr, rs_ptr, S, SP, sx_s, sx_h, mult,
                     D: tl.constexpr, BS: tl.constexpr, SMOOTH: tl.constexpr):
    pid, h = tl.program_id(0), tl.program_id(1)
    G: tl.constexpr = D // 16
    rows = pid * BS + tl.arange(0, BS)
    r3 = rows[:, None, None]
    ch = tl.arange(0, G)[None, :, None] * 16 + 2 * tl.arange(0, 8)[None, None, :]    # even channels [1, G, 8]
    m = r3 < S
    src = x_ptr + h * sx_h + r3.to(tl.int64) * sx_s + ch
    xe = tl.load(src, mask=m, other=0.0).to(tl.float32)
    xo = tl.load(src + 1, mask=m, other=0.0).to(tl.float32)
    if SMOOTH:
        xe = tl.where(m, xe - tl.load(mean_ptr + h * D + ch), 0.0)
        xo = tl.where(m, xo - tl.load(mean_ptr + h * D + ch + 1), 0.0)
    codes, gs, rs = _quant_pairs(xe * mult, xo * mult, G)
    out = h.to(tl.int64) * SP + rows
    tl.store(codes_ptr + out[:, None, None] * (D // 2) + tl.arange(0, G)[None, :, None] * 8
             + tl.arange(0, 8)[None, None, :], codes)
    tl.store(scales_ptr + out[:, None] * G + tl.arange(0, G)[None, :], gs)
    tl.store(rs_ptr + out, tl.where(rows < S, rs, 0.0))


@triton.jit
def _quant_qk8_kernel(x_ptr, mean_ptr, out_ptr, rs_ptr, S, SP, sx_s, sx_h, mult, qmax, D: tl.constexpr,
                      BS: tl.constexpr, SMOOTH: tl.constexpr):
    # FP8 e4m3 rows with one fp32 scale a row (rowmax / qmax: 448, or 16 when QK^T accumulates in fp16)
    pid, h = tl.program_id(0), tl.program_id(1)
    rows = pid * BS + tl.arange(0, BS)
    cols = tl.arange(0, D)
    m = rows < S
    x = tl.load(x_ptr + h * sx_h + rows[:, None].to(tl.int64) * sx_s + cols[None, :], mask=m[:, None],
                other=0.0).to(tl.float32)
    if SMOOTH:
        x = tl.where(m[:, None], x - tl.load(mean_ptr + h * D + cols)[None, :], 0.0)
    x = x * mult
    rs = tl.maximum(tl.max(tl.abs(x), axis=1), 1e-30) / qmax
    out = h.to(tl.int64) * SP + rows
    tl.store(out_ptr + out[:, None] * D + cols[None, :], (x * (1.0 / rs)[:, None]).to(tl.float8e4nv))
    tl.store(rs_ptr + out, tl.where(m, rs, 0.0))


@triton.jit
def _quant_v8_kernel(v_ptr, vmax_ptr, out_ptr, S, SP, sv_s, sv_h, vq_max, D: tl.constexpr, BS: tl.constexpr):
    pid, h = tl.program_id(0), tl.program_id(1)
    rows = pid * BS + tl.arange(0, BS)
    cols = tl.arange(0, D)
    m = rows < S
    v = tl.load(v_ptr + h * sv_h + rows[:, None].to(tl.int64) * sv_s + cols[None, :], mask=m[:, None],
                other=0.0).to(tl.float32)
    sc = vq_max / tl.maximum(tl.load(vmax_ptr + h * D + cols), 1e-30)
    q = (v * sc[None, :]).to(tl.float8e4nv)
    tl.store(out_ptr + (h.to(tl.int64) * SP + rows[:, None]) * D + cols[None, :], q)


@triton.jit
def _quant_v4_kernel(v_ptr, codes_ptr, scales_ptr, rs_ptr, S, SP, sv_s, sv_h, D: tl.constexpr, BS: tl.constexpr):
    # V^T per head: rows = channels, quantized along the sequence (the PV contraction) in blocks of BS tokens:
    # codes [D, SP/2], e4m3 scales [D, SP/16], one fp32 scale per (channel, token block)
    pid, h = tl.program_id(0), tl.program_id(1)
    G: tl.constexpr = BS // 16
    ch = tl.arange(0, D)[:, None, None]
    tok = pid * BS + tl.arange(0, G)[None, :, None] * 16 + 2 * tl.arange(0, 8)[None, None, :]   # even tokens
    src = v_ptr + h * sv_h + tok.to(tl.int64) * sv_s + ch
    ve = tl.load(src, mask=tok < S, other=0.0).to(tl.float32)
    vo = tl.load(src + sv_s, mask=tok + 1 < S, other=0.0).to(tl.float32)
    codes, gs, rs = _quant_pairs(ve, vo, G)
    row = h.to(tl.int64) * D + tl.arange(0, D)
    tl.store(codes_ptr + row[:, None, None] * (SP // 2) + pid * (BS // 2) + tl.arange(0, G)[None, :, None] * 8
             + tl.arange(0, 8)[None, None, :], codes)
    tl.store(scales_ptr + row[:, None] * (SP // 16) + pid * G + tl.arange(0, G)[None, :], gs)
    tl.store(rs_ptr + row * (SP // BS) + pid, rs)


# ---------------------------------------------------------------------------------------------------- attend
@triton.jit
def _attn_kernel(qc_ptr, qs_ptr, qr_ptr, kc_ptr, ks_ptr, kr_ptr, v_ptr, vs_ptr, vr_ptr, vmax_ptr, o_ptr,
                 S, SP, so_s, so_h, D: tl.constexpr, BM: tl.constexpr, BN: tl.constexpr, QK: tl.constexpr,
                 PV: tl.constexpr, PV_F16: tl.constexpr, VB: tl.constexpr, VQ: tl.constexpr):
    pid, h = tl.program_id(0), tl.program_id(1)
    rows = pid * BM + tl.arange(0, BM)
    hq = h.to(tl.int64) * SP
    dq = tl.arange(0, D // 2)
    ds = tl.arange(0, D // 16)
    if QK == "fp4":
        q = tl.load(qc_ptr + (hq + rows)[:, None] * (D // 2) + dq[None, :])
        qs = tl.load(qs_ptr + (hq + rows)[:, None] * (D // 16) + ds[None, :])
    else:
        q = tl.load(qc_ptr + (hq + rows)[:, None] * D + tl.arange(0, D)[None, :])
    qr = tl.load(qr_ptr + hq + rows)
    m_i = tl.full((BM,), -float("inf"), tl.float32)
    l_i = tl.zeros((BM,), tl.float32)
    acc = tl.zeros((BM, D), tl.float32)
    cols = tl.arange(0, D)
    for n0 in range(0, SP, BN):
        keys = n0 + tl.arange(0, BN)
        kr = tl.load(kr_ptr + hq + keys)
        if QK == "fp4":
            k = tl.load(kc_ptr + (hq + keys)[:, None] * (D // 2) + dq[None, :])
            ks = tl.load(ks_ptr + (hq + keys)[:, None] * (D // 16) + ds[None, :])
            s = tl.dot_scaled(q, qs, "e2m1", tl.trans(k), ks, "e2m1")
        else:
            k = tl.load(kc_ptr + (hq + keys)[:, None] * D + tl.arange(0, D)[None, :])
            if QK == "fp8h":
                s = tl.dot(q, tl.trans(k), out_dtype=tl.float16).to(tl.float32)   # |s| <= 128 * 16 * 16
            else:
                s = tl.dot(q, tl.trans(k))
        s = s * kr[None, :] * qr[:, None]
        if n0 + BN > S:
            s = tl.where((keys < S)[None, :], s, -float("inf"))
        m_new = tl.maximum(m_i, tl.max(s, axis=1))
        alpha = tl.math.exp2(m_i - m_new)
        p = tl.math.exp2(s - m_new[:, None])
        l_i = l_i * alpha + tl.sum(p, axis=1)
        acc = acc * alpha[:, None]
        if PV == "fp8":
            v = tl.load(v_ptr + (hq + keys)[:, None] * D + cols[None, :])
            p8 = (p * 448.0).to(tl.float8e4nv)
            if PV_F16:
                acc += tl.dot(p8, v, out_dtype=tl.float16).to(tl.float32)
            else:
                acc = tl.dot(p8, v, acc)
        else:
            # P [BM, BN] -> NVFP4 per 16 keys over P * 2688; V^T codes [D, BN/2]
            pg = tl.reshape(p * 2688.0, (BM, BN // 16, 16))
            gs = (tl.max(pg, axis=2) / 6.0).to(tl.float8e4nv)
            pq = pg * (1.0 / tl.maximum(gs.to(tl.float32), 1e-30))[:, :, None]
            lo, hi = tl.split(tl.reshape(pq, (BM, BN // 2, 2)))
            pc = _e2m1x2(lo, hi)
            hv = h.to(tl.int64) * D + cols
            vc = tl.load(v_ptr + hv[:, None] * (SP // 2) + n0 // 2 + tl.arange(0, BN // 2)[None, :])
            vsc = tl.load(vs_ptr + hv[:, None] * (SP // 16) + n0 // 16 + tl.arange(0, BN // 16)[None, :])
            vr = tl.load(vr_ptr + hv * (SP // VB) + n0 // VB)
            pv = tl.dot_scaled(pc, gs, "e2m1", tl.trans(vc), vsc, "e2m1")
            acc += pv * vr[None, :]
        m_i = m_new
    if PV == "fp8":
        vmax = tl.load(vmax_ptr + h * D + cols)
        o = acc * (vmax / (448.0 * VQ))[None, :] / l_i[:, None]
    else:
        o = acc / (2688.0 * l_i[:, None])
    msk = rows < S
    tl.store(o_ptr + h * so_h + rows[:, None].to(tl.int64) * so_s + cols[None, :], o.to(tl.bfloat16),
             mask=msk[:, None])


# ---------------------------------------------------------------------------------------------------- entry
BM, BN, BS = 128, 64, 128


def attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, qk: str = "fp4", pv: str = "fp8",
              pv_f16: bool = False, smooth: bool = True, out: torch.Tensor | None = None) -> torch.Tensor:
    """q, k, v [1, H, S, 128] bf16 (strided views ok, last dim contiguous) -> [1, H, S, 128] bf16."""

    _, H, S, D = q.shape
    dev = q.device
    SP = triton.cdiv(S, BS) * BS
    kmean = torch.zeros((H, D), dtype=torch.float32, device=dev)
    vmax = torch.zeros((H, D), dtype=torch.float32, device=dev)
    grid = (triton.cdiv(S, BS), H)
    _stats_kernel[(triton.cdiv(S, 64), H)](k, v, kmean, vmax, S, k.stride(2), k.stride(1), v.stride(2), v.stride(1),
                                           D, 64, num_warps=4)
    qr = torch.empty((H, SP), dtype=torch.float32, device=dev)
    kr = torch.empty_like(qr)
    if qk == "fp4":
        qc = torch.empty((H, SP, D // 2), dtype=torch.uint8, device=dev)
        qs = torch.empty((H, SP, D // 16), dtype=torch.float8_e4m3fn, device=dev)
        kc, ks = torch.empty_like(qc), torch.empty_like(qs)
        _quant_qk_kernel[grid](q, kmean, qc, qs, qr, S, SP, q.stride(2), q.stride(1), LOG2E / math.sqrt(D), D, BS,
                               False, num_warps=8)
        _quant_qk_kernel[grid](k, kmean, kc, ks, kr, S, SP, k.stride(2), k.stride(1), 1.0, D, BS, smooth,
                               num_warps=8)
    else:
        qc = torch.empty((H, SP, D), dtype=torch.float8_e4m3fn, device=dev)
        kc = torch.empty_like(qc)
        qs = ks = qc
        qmax = 16.0 if qk == "fp8h" else 448.0
        _quant_qk8_kernel[grid](q, kmean, qc, qr, S, SP, q.stride(2), q.stride(1), LOG2E / math.sqrt(D), qmax, D, BS,
                                False, num_warps=8)
        _quant_qk8_kernel[grid](k, kmean, kc, kr, S, SP, k.stride(2), k.stride(1), 1.0, qmax, D, BS, smooth,
                                num_warps=8)
    if pv == "fp8":
        vq = torch.empty((H, SP, D), dtype=torch.float8_e4m3fn, device=dev)
        # fp16 accumulation over a 64-key tile: 64 * 448 (P) * 2 (V) stays below 65504
        _quant_v8_kernel[grid](v, vmax, vq, S, SP, v.stride(2), v.stride(1), 2.0 if pv_f16 else 448.0, D, BS,
                               num_warps=8)
        vs = vr = vq
    else:
        vq = torch.empty((H, D, SP // 2), dtype=torch.uint8, device=dev)
        vs = torch.empty((H, D, SP // 16), dtype=torch.float8_e4m3fn, device=dev)
        vr = torch.empty((H, D, SP // BS), dtype=torch.float32, device=dev)
        _quant_v4_kernel[grid](v, vq, vs, vr, S, SP, v.stride(2), v.stride(1), D, BS, num_warps=8)
    o = torch.empty((1, H, S, D), dtype=torch.bfloat16, device=dev) if out is None else out
    _attn_kernel[(SP // BM, H)](qc, qs, qr, kc, ks, kr, vq, vs, vr, vmax, o, S, SP, o.stride(2), o.stride(1), D, BM,
                                BN, qk, pv, pv_f16, BS, 2.0 if pv_f16 else 448.0, num_warps=8, num_stages=3)
    return o
