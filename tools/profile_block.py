"""Where an engine block's time goes at a given sequence length: CUDA-event timings of each stage of one block
(converted engine from the cache, block weights resident), synthetic activations of the real shapes.

Usage: tools\\env.cmd python tools\\profile_block.py [--seq 37790] [--attn int8] [--spec nvfp4]
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
from tfvideo import kernels as K  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seq", type=int, default=37790)
    ap.add_argument("--attn", default="int8")
    ap.add_argument("--spec", default="nvfp4")
    ap.add_argument("--block", type=int, default=25)
    ap.add_argument("--forward", action="store_true", help="time all blocks in order (streaming as planned)")
    ap.add_argument("--resident", type=int, default=-1, help="force this many resident blocks (with --forward)")
    a = ap.parse_args()
    ext.use_prebuilt()
    path = store.cache_dir() / (store.key(paths.dit(), a.spec) + ".safetensors")
    if not path.is_file():
        sys.exit(f"no converted engine at {path}: run the loader once (or tools/check_forward.py) first")
    e = store.load(path, "cuda")
    e.attn = A.get(a.attn)
    S, D, cfg = a.seq, e.cfg.dim, e.cfg
    torch.manual_seed(0)
    x = torch.randn(S, D, device="cuda", dtype=torch.bfloat16)
    t_emb = torch.randn(3, 8, device="cuda")
    segs = [(0, S, 0)]
    half = torch.randn(S, 48, device="cuda")
    import comfy_kitchen  # noqa: F401  (rms_rope)
    ang = torch.cat([half, half], -1)[:, :48]
    c, s = torch.cos(ang), torch.sin(ang)
    rope = torch.stack([c, -s, s, c], -1).reshape(1, S, 1, 48, 2, 2).to(torch.bfloat16)
    e.plan_memory(S)
    if a.forward:
        if a.resident >= 0:
            e.release_vram()
            for j in range(a.resident):
                for k, (holder, attr) in e.streamer.refs(j):
                    setattr(holder, attr, e.blocks[j].host[k].cuda())
                e.blocks[j].resident = True
            e.streamer.plan()
            e._plan_seq = S
        for rep in range(3):
            e.begin_forward(S, {})                  # delayed scales: the first forward syncs per linear
            torch.cuda.synchronize()
            st, en = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            st.record()
            for j in range(len(e.blocks)):
                e.block(j, x, t_emb, segs, rope)
            en.record()
            torch.cuda.synchronize()
            n_res = sum(b.resident for b in e.blocks)
            mode = "delayed scales" if any(getattr(l, "next_act", None) for _, l in e.linears()) else "per-call scales"
            print(f"forward of {len(e.blocks)} blocks at S={S}, {n_res} resident, {mode}: "
                  f"{st.elapsed_time(en) / 1000:.2f} s")
        return
    i = a.block
    b = e.blocks[i]

    timings = {}

    def stage(name, fn):
        st, en = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        st.record()
        out = fn()
        en.record()
        torch.cuda.synchronize()
        timings[name] = timings.get(name, 0.0) + st.elapsed_time(en)
        return out

    for rep in range(3):
        if rep == 1:
            timings.clear()
        e.streamer.acquire(i)
        idx = e.row_index(segs, S)
        mod = stage("modulation", lambda: e.modulation(b, t_emb))
        h = stage("norm_mod", lambda: K.norm_mod(x, b.norm1, mod, 0, idx, cfg.eps))
        qkv = stage("qkv gemm", lambda: b.lins["qkv"](h))
        H, Dh = cfg.heads, cfg.head_dim
        q = qkv[:, :H * Dh].view(1, S, H, Dh)
        k = qkv[:, H * Dh:2 * H * Dh].view(1, S, H, Dh)
        v = qkv[:, 2 * H * Dh:].view(S, H, Dh)
        import comfy_kitchen as ck
        stage("qk norm+rope", lambda: ck.rms_rope_split_half_(q, k, rope, b.q_norm, b.k_norm, epsilon=cfg.qk_eps,
                                                               rot_dim=96))
        o = stage("attention", lambda: e.attn(q[0].transpose(0, 1).unsqueeze(0), k[0].transpose(0, 1).unsqueeze(0),
                                              v.transpose(0, 1).unsqueeze(0)))
        o2 = stage("attn transpose", lambda: o[0].transpose(0, 1).reshape(S, H * Dh))
        stage("out gemm", lambda: b.lins["out"](o2, h))
        stage("gate_add", lambda: K.gate_add(x, h, mod, 0, idx))
        stage("norm_mod", lambda: K.norm_mod(x, b.norm2, mod, 1, idx, cfg.eps, out=h))
        for r0 in range(0, S, e.MLP_CHUNK):
            r1 = min(S, r0 + e.MLP_CHUNK)
            gu = stage("fc1 gemm", lambda: b.lins["fc1"](h[r0:r1]))
            act = stage("swiglu", lambda: K.swiglu(gu))
            stage("fc2 gemm", lambda: b.lins["fc2"](act, h[r0:r1]))
        stage("gate_add", lambda: K.gate_add(x, h, mod, 1, idx))
        e.streamer.release(i)
        del qkv, q, k, v, o, o2
    total = sum(timings.values()) / 2
    print(f"one block at S={S} ({a.spec}, attention {a.attn}): {total:.1f} ms  ->  x50 = {total * 50 / 1000:.2f} s/step")
    for name, ms in sorted(timings.items(), key=lambda kv: -kv[1]):
        print(f"  {name:16s} {ms / 2:8.2f} ms  {100 * ms / 2 / total:5.1f}%")


if __name__ == "__main__":
    main()
