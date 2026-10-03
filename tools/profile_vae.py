r"""Where MiniMax H3's video VAE decode spends its time: one temporal chunk x one 256 px tile through ComfyUI's decoder,
torch.profiler grouped by kernel (portable python, -B). Scale by the tile/chunk counts the log prints.

Usage: python_embeded\python.exe -B tools\profile_vae.py [--width 1344 --height 768 --seconds 5]
"""
import argparse
import asyncio
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "bench"))
import paths  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--width", type=int, default=1344)
ap.add_argument("--height", type=int, default=768)
ap.add_argument("--seconds", type=float, default=5.0)
a = ap.parse_args()
COMFY = paths.comfy()
sys.path.insert(0, str(COMFY))
os.chdir(COMFY)
sys.argv = ["comfy", "--disable-all-custom-nodes", "--database-url", "sqlite:///:memory:"]
import comfy.options  # noqa: E402

comfy.options.enable_args_parsing()
import torch  # noqa: E402
import nodes  # noqa: E402
import comfy.model_management as mm  # noqa: E402

asyncio.run(nodes.init_extra_nodes(init_custom_nodes=False, init_api_nodes=False))
torch.inference_mode().__enter__()
vae = nodes.NODE_CLASS_MAPPINGS["VAELoader"]().load_vae(vae_name="minimax_h3_video_vae_fp16.safetensors")[0]
mm.load_models_gpu([vae.patcher])
m = vae.first_stage_model
print("decoder:", type(m).__name__, "tile", m.tile_size, "overlap", m.tile_overlap_min, "clip_length", m.clip_length,
      "vae_ratio", m.vae_ratio, "vae_ratio_t", m.vae_ratio_t, "tokens_chunk", m.tokens_chunk_size, "dtype", vae.vae_dtype)
from comfy_extras.nodes_minimax_h3 import video_latent_t, align_frame_count  # noqa: E402

T = video_latent_t(align_frame_count(round(a.seconds * 24)))
ys, xs = m.split_tiles(a.height)[0], m.split_tiles(a.width)[0]
chunks, pad = m._decode_temporal_chunks(T)[1], None
print(f"latent T {T}: {len(ys)} x {len(xs)} tiles per chunk, {chunks} temporal chunks")
tl = m.tile_size // m.vae_ratio
z = torch.randn(1, 24, m.tokens_chunk_size + m.token_overlap, tl, tl, device="cuda", dtype=vae.vae_dtype)
dec = lambda: m._decode_pixels(z)  # noqa: E731
for _ in range(2):
    dec()
torch.cuda.synchronize()
import time  # noqa: E402

t = time.perf_counter()
for _ in range(3):
    dec()
torch.cuda.synchronize()
per = (time.perf_counter() - t) / 3
print(f"one tile chunk {tuple(z.shape)}: {per * 1000:.0f} ms -> x {len(ys) * len(xs) * chunks} = "
      f"{per * len(ys) * len(xs) * chunks:.1f} s (before batching/overlap effects)")
from torch.profiler import ProfilerActivity, profile  # noqa: E402

with profile(activities=[ProfilerActivity.CUDA]) as prof:
    dec()
    torch.cuda.synchronize()
print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=15, max_name_column_width=70))
