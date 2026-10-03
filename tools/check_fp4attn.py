"""fp4attn: e2m1 packing against a reference quantizer, then attention variants vs fp32 SDPA (accuracy and speed).
Usage: tools/env.cmd python tools/check_fp4attn.py [--seq 4096] [--qkv runs/ref/qkv.pt]"""
import argparse
import time

import torch
import torch.nn.functional as F
import triton
import triton.language as tl

from tfvideo import fp4attn as FA

E2M1 = torch.tensor([0, .5, 1, 1.5, 2, 3, 4, 6, -0., -.5, -1, -1.5, -2, -3, -4, -6])


@triton.jit
def _pack_kernel(x_ptr, o_ptr, N: tl.constexpr):
    i = tl.arange(0, N)
    x = tl.load(x_ptr + i)
    lo, hi = tl.split(tl.reshape(x, (N // 2, 2)))
    tl.store(o_ptr + tl.arange(0, N // 2), FA._e2m1x2(lo, hi))


def check_pack():
    x = (torch.rand(256, device="cuda") * 14 - 7).float()
    out = torch.empty(128, dtype=torch.uint8, device="cuda")
    _pack_kernel[(1,)](x, out, 256)
    codes = torch.stack([out & 15, out >> 4], 1).flatten().long().cpu()
    deq = E2M1[codes]
    # reference: nearest e2m1 value, saturating at +-6 (ties may go either way at exact midpoints)
    ref = E2M1[(x.cpu()[:, None] - E2M1[None, :]).abs().argmin(1)]
    bad = (deq - ref).abs() > 1e-6
    tie = ((x.cpu()[:, None] - E2M1[None, :]).abs().sort(1).values[:, :2].diff(1).abs().squeeze(1) < 1e-6)
    print(f"e2m1x2 pack: {int((bad & ~tie).sum())} mismatches of 256 (low nibble = even element)")


def timeit(fn, reps=5):
    fn()
    torch.cuda.synchronize()
    t = time.perf_counter()
    for _ in range(reps):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t) / reps


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seq", type=int, default=4096)
    ap.add_argument("--heads", type=int, default=56)
    ap.add_argument("--qkv", default="")
    a = ap.parse_args()
    check_pack()
    torch.manual_seed(0)
    if a.qkv:
        d = torch.load(a.qkv)
        q, k, v = (d[n].cuda() for n in ("q", "k", "v"))
    else:
        q, k, v = (torch.randn(1, a.heads, a.seq, 128, device="cuda", dtype=torch.bfloat16) for _ in range(3))
        k = k + 2.0 * torch.randn(1, a.heads, 1, 128, device="cuda", dtype=torch.bfloat16)    # channel bias (smoothing)
    S, H = q.shape[2], q.shape[1]
    rows = torch.randperm(S, device="cuda")[:512].sort().values
    ref = F.scaled_dot_product_attention(q[:, :, rows].float(), k.float(), v.float())
    flops = 4 * S * S * 128 * H
    variants = {"sdpa_bf16": lambda: F.scaled_dot_product_attention(q, k, v),
                "fp4qk_fp8pv": lambda: FA.attention(q, k, v, pv="fp8"),
                "fp4qk_fp8pv_nosmooth": lambda: FA.attention(q, k, v, pv="fp8", smooth=False),
                "fp4qk_fp4pv": lambda: FA.attention(q, k, v, pv="fp4"),
                "fp8qk_fp8pv": lambda: FA.attention(q, k, v, qk="fp8", pv="fp8"),
                "fp4qk_fp8pv_f16acc": lambda: FA.attention(q, k, v, pv="fp8", pv_f16=True)}
    for name, fn in variants.items():
        try:
            o = fn()[:, :, rows].float()
            err = ((o - ref).norm() / ref.norm()).item()
            cos = F.cosine_similarity(o.flatten(), ref.flatten(), dim=0).item()
            t = timeit(fn)
            print(f"{name:22s} {t * 1e3:8.2f} ms {flops / t / 1e12:7.1f} TF/s  rel_err {err:.4f} cos {cos:.6f}", flush=True)
        except Exception as e:  # noqa: BLE001
            print(f"{name:22s} failed: {type(e).__name__}: {str(e)[:600]}", flush=True)


if __name__ == "__main__":
    main()
