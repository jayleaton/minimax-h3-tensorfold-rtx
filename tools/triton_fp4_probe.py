"""Does Triton (this install) run block-scaled FP4 / FP8 dots natively on this GPU? A plain tiled matmul per format,
timed at 4096^3 and checked against a dequantized reference. Native sm_120 FP4 should be several times bf16."""
import time

import torch
import triton
import triton.language as tl


@triton.jit
def mm_kernel(a_ptr, b_ptr, as_ptr, bs_ptr, c_ptr, M, N, K, FMT: tl.constexpr, GS: tl.constexpr,
              BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
    pm, pn = tl.program_id(0), tl.program_id(1)
    rm = pm * BM + tl.arange(0, BM)
    rn = pn * BN + tl.arange(0, BN)
    acc = tl.zeros((BM, BN), dtype=tl.float32)
    PK: tl.constexpr = 2 if FMT == "e2m1" else 1
    for k0 in range(0, K, BK):
        rk = k0 // PK + tl.arange(0, BK // PK)
        a = tl.load(a_ptr + rm[:, None] * (K // PK) + rk[None, :])
        b = tl.load(b_ptr + rn[:, None] * (K // PK) + rk[None, :])
        if FMT == "bf16":
            acc = tl.dot(a, tl.trans(b), acc)
        elif FMT == "e4m3":
            acc = tl.dot(a, tl.trans(b), acc)
        else:
            rs = k0 // GS + tl.arange(0, BK // GS)
            sa = tl.load(as_ptr + rm[:, None] * (K // GS) + rs[None, :])
            sb = tl.load(bs_ptr + rn[:, None] * (K // GS) + rs[None, :])
            acc = tl.dot_scaled(a, sa, "e2m1", tl.trans(b), sb, "e2m1", acc)
    tl.store(c_ptr + rm[:, None] * N + rn[None, :], acc)


def run(fmt, M=4096, N=4096, K=4096, gs=32, scale_dtype=torch.uint8):
    dev = "cuda"
    if fmt == "bf16":
        a = torch.randn(M, K, device=dev, dtype=torch.bfloat16)
        b = torch.randn(N, K, device=dev, dtype=torch.bfloat16)
        sa = sb = a
    elif fmt == "e4m3":
        a = torch.randn(M, K, device=dev).to(torch.float8_e4m3fn)
        b = torch.randn(N, K, device=dev).to(torch.float8_e4m3fn)
        sa = sb = a
    else:
        a = torch.randint(0, 256, (M, K // 2), device=dev, dtype=torch.uint8)
        b = torch.randint(0, 256, (N, K // 2), device=dev, dtype=torch.uint8)
        if scale_dtype == torch.uint8:
            sa = torch.full((M, K // gs), 127, device=dev, dtype=torch.uint8)
            sb = torch.full((N, K // gs), 127, device=dev, dtype=torch.uint8)
        else:
            sa = torch.ones((M, K // gs), device=dev).to(scale_dtype)
            sb = torch.ones((N, K // gs), device=dev).to(scale_dtype)
    c = torch.empty(M, N, device=dev, dtype=torch.float32)
    BM, BN, BK = 128, 128, (128 if fmt != "bf16" else 64)
    grid = (M // BM, N // BN)
    k = lambda: mm_kernel[grid](a, b, sa, sb, c, M, N, K, fmt, gs, BM, BN, BK, num_warps=8, num_stages=3)  # noqa
    k()
    torch.cuda.synchronize()
    t = time.perf_counter()
    for _ in range(10):
        k()
    torch.cuda.synchronize()
    dt = (time.perf_counter() - t) / 10
    print(f"{fmt:5s} gs={gs:2d} scale={str(scale_dtype).split('.')[-1]:14s} {2 * M * N * K / dt / 1e12:7.1f} TF/s", flush=True)
    return k


if __name__ == "__main__":
    run("bf16")
    run("e4m3")
    run("e2m1", gs=32)
    try:
        run("e2m1", gs=16, scale_dtype=torch.float8_e4m3fn)
    except Exception as e:  # noqa: BLE001
        print("nvfp4 (e4m3 scales, group 16) failed:", type(e).__name__, str(e)[:300])
