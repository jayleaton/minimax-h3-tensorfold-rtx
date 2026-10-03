r"""Ground truth for tfvideo: ComfyUI's own MiniMax H3 (the int8 ConvRot checkpoint) on real inputs (portable python, -B).

Encodes a prompt with the MiniMax text encoder, starts a real sampling run at a small size and captures the
diffusion-model calls at the chosen steps (inputs as ComfyUI hands them over: the packed AV latent, timestep,
refined text states, payload with the packed layout), plus ComfyUI's outputs. Saves runs/ref/forward_<tag>.pt.
Usage: python_embeded\python.exe -B bench\comfy_dump_forward.py [--width 768 --height 448 --seconds 2] [--first-frame img]
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
ap.add_argument("--width", type=int, default=768)
ap.add_argument("--height", type=int, default=448)
ap.add_argument("--seconds", type=float, default=2.0)
ap.add_argument("--steps", type=int, default=20)
ap.add_argument("--capture", default="0,6,13,19", help="steps whose model calls are kept")
ap.add_argument("--first-frame", default="")
ap.add_argument("--tag", default="small")
a = ap.parse_args()

COMFY = paths.comfy()
sys.path.insert(0, str(COMFY))
os.chdir(COMFY)
out_dir = ROOT / "runs" / "ref"
out_dir.mkdir(parents=True, exist_ok=True)
sys.argv = ["comfy", "--disable-all-custom-nodes", "--database-url", "sqlite:///:memory:",
            "--temp-directory", str(out_dir / "temp")]
import comfy.options  # noqa: E402

comfy.options.enable_args_parsing()
import torch  # noqa: E402
import nodes  # noqa: E402
import comfy.patcher_extension  # noqa: E402
import comfy.sample  # noqa: E402

asyncio.run(nodes.init_extra_nodes(init_custom_nodes=False, init_api_nodes=False))
N = nodes.NODE_CLASS_MAPPINGS
from comfy_h3 import PROMPT, frames_for  # noqa: E402

torch.inference_mode().__enter__()
clip = N["CLIPLoader"]().load_clip(clip_name="qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors", type="minimax",
                                    device="default")[0]
vae = N["VAELoader"]().load_vae(vae_name="minimax_h3_video_vae_fp16.safetensors")[0]
first = None
if a.first_frame:
    import numpy as np
    from PIL import Image
    first = torch.from_numpy(np.asarray(Image.open(a.first_frame).convert("RGB"))).float().div(255)[None]
res = N["MiniMaxH3ImageToVideo"].execute(clip=clip, vae=vae, prompt=PROMPT, width=a.width, height=a.height,
                                         length=frames_for(a.seconds), first_frame=first)
positive, latent = res.result[0], res.result[1]
del clip

model = N["UNETLoader"]().load_unet(unet_name=paths.DIT_NAME, weight_dtype="default")[0]
capture = {int(s) for s in a.capture.split(",")}
calls = []


def grab(executor, x, timestep, context, transformer_options={}, **kwargs):
    out = executor(x, timestep, context, transformer_options, **kwargs)
    step = len(calls)
    if step in capture:
        keep = {k: transformer_options[k] for k in ("sample_sigmas", "minimax_h3_sigma_shift_video",
                                                     "minimax_h3_sigma_shift_audio") if k in transformer_options}
        calls.append({"step": step, "x": [t.cpu() for t in x], "timestep": timestep.cpu(), "context": context.cpu(),
                      "kwargs": {k: v for k, v in kwargs.items()}, "transformer_options": keep,
                      "out": [t.float().cpu() for t in out]})
    else:
        calls.append({"step": step})
    print(f"step {step} sigma {float(timestep.flatten()[0]) / 1000:.4f}", flush=True)
    return out


model = model.clone()
model.add_wrapper(comfy.patcher_extension.WrappersMP.DIFFUSION_MODEL, grab)
guider = N["BasicGuider"].execute(model=model, conditioning=positive).result[0]
sigmas = N["BasicScheduler"].execute(model=model, scheduler="simple", steps=a.steps, denoise=1.0).result[0]
sampler = N["KSamplerSelect"].execute(sampler_name="res_multistep").result[0]
noise = N["RandomNoise"].execute(noise_seed=757358688076805).result[0]
out = N["SamplerCustomAdvanced"].execute(noise=noise, guider=guider, sampler=sampler, sigmas=sigmas,
                                         latent_image=latent).result[0]
keep = [c for c in calls if "x" in c]
torch.save({"calls": keep, "sigmas": sigmas.cpu(), "final": [t.cpu() for t in out["samples"].unbind()],
            "width": a.width, "height": a.height, "frames": frames_for(a.seconds)}, out_dir / f"forward_{a.tag}.pt")
print("saved", [(c["step"], [tuple(t.shape) for t in c["x"]]) for c in keep], "context", tuple(keep[0]["context"].shape))
