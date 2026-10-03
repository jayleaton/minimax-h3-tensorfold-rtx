r"""tfvideo vs ComfyUI's MiniMax H3 on the dumped calls (runs/ref/forward_<tag>.pt), per precision. Portable python:

    python_embeded\python.exe -B tools\check_forward.py small nvfp4 [fp8 bf16]

Builds the ComfyUI shell model with engine blocks exactly as the loader node does, replays each captured call
(the inputs of MiniMaxH3Model._forward: packed AV latent, timestep, refined text, payload) and compares the video and audio velocities with ComfyUI's.
Engines are converted in memory (bf16 needs ~39 GB of RAM) and not cached.
"""
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "bench"))
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "vendor" / "TensorFold" / "src"))
import paths  # noqa: E402

COMFY = paths.comfy()
sys.path.insert(0, str(COMFY))
os.chdir(COMFY)
tag, kinds = sys.argv[1], sys.argv[2:] or ["nvfp4"]
attn = os.environ.get("TFVIDEO_ATTN", "auto")
sys.argv = ["comfy", "--disable-all-custom-nodes", "--database-url", "sqlite:///:memory:"]
import comfy.options  # noqa: E402

comfy.options.enable_args_parsing()
import torch  # noqa: E402
import comfy.model_management as mm  # noqa: E402

from tfvideo import attention as A  # noqa: E402
from tfvideo import comfy_nodes, ext, store  # noqa: E402

torch.inference_mode().__enter__()
ref = torch.load(ROOT / "runs" / "ref" / f"forward_{tag}.pt", weights_only=False)


def cmp(a, b):
    a, b = a.float().flatten(), b.float().flatten()
    return f"relL2 {((a - b).norm() / b.norm()).item():.4f} cos {torch.nn.functional.cosine_similarity(a, b, dim=0).item():.5f}"


def move(v, dev):
    if torch.is_tensor(v):
        return v.to(dev)
    if isinstance(v, dict):
        return {k: move(x, dev) for k, x in v.items()}
    if isinstance(v, list):
        return [move(x, dev) for x in v]
    return v


ext.use_prebuilt()
dit = paths.dit()
for kind in kinds:
    t0 = time.perf_counter()
    engine = store.convert(dit, kind, device="cuda", log=print)
    engine.attn, engine.attn_name = A.get(attn), attn
    print(f"[{kind}] converted in {time.perf_counter() - t0:.0f} s, {engine.nbytes() / 2**30:.2f} GiB of block weights")
    patcher = comfy_nodes.build_model(dit, engine)
    mm.load_models_gpu([patcher])
    dm = patcher.model.diffusion_model
    dm.to("cuda")
    for c in ref["calls"]:
        x = [t.cuda() for t in c["x"]]
        kw = move(c["kwargs"], "cuda")
        to = move(dict(c["transformer_options"]), "cuda")
        torch.cuda.synchronize()
        t1 = time.perf_counter()
        out = dm._forward(x, c["timestep"].cuda(), c["context"].cuda(), transformer_options=to, **kw)
        torch.cuda.synchronize()
        dt = time.perf_counter() - t1
        print(f"[{kind}] step {c['step']:2d} sigma {float(c['timestep'].flatten()[0]) / 1000:.3f}  "
              f"video {cmp(out[0].cpu(), c['out'][0])}  audio {cmp(out[1].cpu(), c['out'][1])}  {dt:.2f} s", flush=True)
    engine.release_vram()
    del patcher, dm, engine, out
    mm.unload_all_models()
    import gc
    gc.collect()
    mm.soft_empty_cache()
