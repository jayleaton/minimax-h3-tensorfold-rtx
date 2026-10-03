"""Bitwise run-to-run determinism of each engine piece on identical inputs (same seed must give the same video).

Usage: tools\\env.cmd python tools\\check_determinism.py [--seq 6000]
"""
import argparse
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "bench"))
import paths  # noqa: E402

from tfvideo import attention as A  # noqa: E402
from tfvideo import ext, store  # noqa: E402


def same(name, fn, reps=4):
    outs = [fn().float().clone() for _ in range(reps)]
    diffs = [(o - outs[0]).abs().max().item() for o in outs[1:]]
    print(f"{name:28s} {'deterministic' if max(diffs) == 0 else 'DIFFERS: max |d| ' + str(max(diffs))}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seq", type=int, default=6000)
    a = ap.parse_args()
    ext.use_prebuilt()
    e = store.load(store.cache_dir() / (store.key(paths.dit(), "nvfp4") + ".safetensors"), "cuda")
    S, D, H = a.seq, e.cfg.dim, e.cfg.heads
    torch.manual_seed(0)
    x = torch.randn(S, D, device="cuda", dtype=torch.bfloat16)
    q, k, v = (torch.randn(1, H, S, 128, device="cuda", dtype=torch.bfloat16) for _ in range(3))
    e.plan_memory(S)
    b = e.blocks[10]
    e.streamer.acquire(10)
    for name in ("qkv", "out", "fc1"):
        lin = b.lins[name]
        xin = torch.randn(S, lin.k, device="cuda", dtype=torch.bfloat16)
        same(f"nvfp4 gemm {name}", lambda: lin(xin))
    for name in ("int8", "cudnn", "sdpa"):
        same(f"attention {name}", lambda: A.get(name)(q, k, v))
    import comfy_kitchen as ck
    same("sol_attn tau 1.3", lambda: ck.sol_attn(q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), tau=1.3,
                                                 token_aug=256))
    t_emb = torch.randn(3, 8, device="cuda")
    segs = [(0, S, 0)]
    half = torch.randn(S, 48, device="cuda")
    c, s = torch.cos(half), torch.sin(half)
    rope = torch.stack([c, -s, s, c], -1).reshape(1, S, 1, 48, 2, 2).to(torch.bfloat16)
    same("whole block (int8 attention)", lambda: e.block(10, x.clone(), t_emb, segs, rope))
    e.streamer.release(10)

    def forward(resident):
        e.release_vram()
        for j in range(resident):
            for kk, (holder, attr) in e.streamer.refs(j):
                setattr(holder, attr, e.blocks[j].host[kk].cuda())
            e.blocks[j].resident = True
        e.streamer.plan()
        e._plan_seq = S
        h = x.clone()
        for j in range(len(e.blocks)):
            h = e.block(j, h, t_emb0, segs, rope)
            if j % 10 == 9 and not bool(torch.isfinite(h).all()):
                raise SystemExit(f"non-finite activations after block {j}: synthetic inputs too far from real ones")
        return h
    t_emb0 = torch.zeros_like(t_emb)        # modulation = the adaln biases: realistic magnitudes over 50 blocks
    e.delayed_scales = False
    for r in (50, 0, 25, 49):
        same(f"50 blocks, {r} resident", lambda: forward(r), reps=3)
    ref = forward(50).float()
    for r in (0, 25, 49):
        d = (forward(r).float() - ref).abs().max().item()
        print(f"50 blocks, {r} resident vs all resident: {'identical' if d == 0 else 'max |d| ' + str(d)}", flush=True)


if __name__ == "__main__":
    main()
