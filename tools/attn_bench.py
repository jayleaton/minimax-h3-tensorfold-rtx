"""Attention kernels at MiniMax H3's shape: one forward's attention is 50 x [S x S] over 56 heads x 128.

Times every candidate on the same bf16 q/k/v ([1, H, S, D]) and reports dense-equivalent TF/s and the error against
an fp32 reference on a slice of query rows. ``--qkv file.pt`` uses real q/k/v dumped from a block (bench/dump_qkv.py);
otherwise unit-RMS Gaussian rows (q/k RMSNorm output), which flatter sparse methods less than real video attention.
Usage: tools\\env.cmd python tools\\attn_bench.py --seq 40000 [--only sdpa_flash,ck_int8]
"""
from __future__ import annotations

import argparse
import math
import time

import torch
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel


def timeit(fn, warm=2, reps=5):
    for _ in range(warm):
        fn()
    torch.cuda.synchronize()
    t = time.perf_counter()
    for _ in range(reps):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t) / reps


def sdpa(backend):
    def run(q, k, v):
        with sdpa_kernel([backend, SDPBackend.EFFICIENT_ATTENTION], set_priority=True):
            return F.scaled_dot_product_attention(q, k, v)
    return run


def ck_int8(q, k, v):
    from comfy_kitchen.sage_attention import int8_attention
    return int8_attention(q, k, v)


def ck_sol(tau):
    def run(q, k, v):
        import comfy_kitchen as ck
        return ck.sol_attn(q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), tau=tau).transpose(1, 2)
    return run


CANDIDATES = {
    "sdpa_flash": sdpa(SDPBackend.FLASH_ATTENTION),
    "sdpa_cudnn": sdpa(SDPBackend.CUDNN_ATTENTION),
    "ck_int8": ck_int8,
    "ck_sol_1.0": ck_sol(1.0),
}


def extra_candidates():
    try:
        from tfvideo import attention as A
    except Exception as e:  # noqa: BLE001
        print("tfvideo.attention unavailable:", e)
        return {}
    from tfvideo import fp4attn as FA
    return {"tf_fp8": A.fp8, "tf_fp4": A.fp4,
            "fp4_pv16": lambda q, k, v: FA.attention(q, k, v, qk="fp4", pv="fp8", pv_f16=True),
            "fp8_pv16": lambda q, k, v: FA.attention(q, k, v, qk="fp8", pv="fp8", pv_f16=True),
            "fp8h_pv16": lambda q, k, v: FA.attention(q, k, v, qk="fp8h", pv="fp8", pv_f16=True)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seq", type=int, default=40000)
    ap.add_argument("--heads", type=int, default=56)
    ap.add_argument("--qkv", default="")
    ap.add_argument("--only", default="")
    ap.add_argument("--check-rows", type=int, default=1024)
    a = ap.parse_args()
    torch.manual_seed(0)
    if a.qkv:
        d = torch.load(a.qkv)
        q, k, v = (d[n].cuda() for n in ("q", "k", "v"))
    else:
        shape = (1, a.heads, a.seq, 128)
        q, k, v = (torch.randn(shape, device="cuda", dtype=torch.bfloat16) for _ in range(3))
    B, H, S, D = q.shape
    flops = 4 * S * S * D * H
    rows = torch.randperm(S, device="cuda")[:a.check_rows].sort().values
    ref = F.scaled_dot_product_attention(q[:, :, rows].float(), k.float(), v.float())
    cands = {**CANDIDATES, **extra_candidates()}
    if a.only:
        cands = {n: f for n, f in cands.items() if n in a.only.split(",")}
    print(f"S={S} H={H} D={D}: {flops / 1e12:.1f} TFLOP per call, x50 blocks = {50 * flops / 1e15:.2f} PFLOP a forward")
    for name, fn in cands.items():
        try:
            out = fn(q, k, v)
            o = out[:, :, rows].float()
            err = ((o - ref).norm() / ref.norm()).item()
            cos = F.cosine_similarity(o.flatten(), ref.flatten(), dim=0).item()
            del out
            t = timeit(lambda: fn(q, k, v))
            print(f"{name:14s} {t * 1e3:9.1f} ms  {flops / t / 1e12:7.1f} TF/s  x50 = {50 * t:6.2f} s   "
                  f"rel_err {err:.4f}  cos {cos:.6f}", flush=True)
        except Exception as e:  # noqa: BLE001
            print(f"{name:14s} failed: {type(e).__name__}: {str(e)[:200]}", flush=True)
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
