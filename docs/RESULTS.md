# Results: baseline, profile, levers, milestone 1

Measured 2 Oct 2026 on one desktop: RTX 5070 Ti 16 GB (Blackwell, sm_120, 70 SMs), NVIDIA driver 617.14, Windows 11,
Ryzen 9 9950X, 64 GB RAM, models on an external drive, through ComfyUI 0.37.0 portable (torch 2.10.0+cu130,
comfy-kitchen 0.2.35, dynamic VRAM on) with the harness in `bench/`. Settings throughout are the "Image to Video
(MiniMax H3)" blueprint's: text-to-video+audio, 24 fps, res_multistep, simple schedule, BasicGuider (no CFG), Qwen3-VL
32B NVFP4 text encoder, fp16 video VAE, fp32 audio VAE; the same prompt (bench/comfy_h3.py) and seeds.

## 1. The model and where a step goes

MiniMax H3's DiT: 50 single-stream blocks, hidden 5376, 56 heads x 128, SwiGLU 14336, per-token modulation from a
small time-embedding curve; ~19.3B parameters in the block linears (the ComfyUI checkpoint stores them int8 with a
256-wide Hadamard rotation, "int8_convrot", 21 GB). The sequence packs [text | keyframe rows | audio | video]: a
1344x768, 124-frame video is 37 latent frames x 1,008 patches, ~37,800 tokens with text and audio.

One forward is then ~1.5 PFLOP of linears and **~2.2 PFLOP of full (non-causal) attention**: unlike the image model,
attention, not the weights, is most of the work. Measured kernels at that shape (`tools/attn_bench.py`, 56 x 128,
S = 39,936):

| Attention kernel | ms per call | TF/s | per forward (x50) | cos vs fp32 |
| --- | ---: | ---: | ---: | ---: |
| cuDNN bf16 (what stock ComfyUI runs; PyTorch's flash kernel is not built for sm_120 on Windows) | 478 | 96 | 23.9 s | 0.999997 |
| comfy-kitchen INT8 (SageAttention-style; `--use-ck-attention`) | 173 | 265 | 8.6 s | 0.99986 |
| this repo, Triton: FP8 QK, FP8 PV | 389 | 118 | 19.5 s | 0.9986 |
| this repo, Triton: NVFP4 QK (block-scaled MMA), FP8 PV | 339 | 135 | 16.9 s | 0.990 |
| ... with PV accumulated in fp16 | 306 | 149 | 15.3 s | 0.990 |

Raw Triton MMA ceilings on this GPU (4096^3, `tools/triton_fp4_probe.py`): bf16 94, FP8 179 (fp32 accumulate) / 269
(fp16 accumulate), NVFP4 292-305 TF/s. The Triton attention kernels are bound by the softmax and pipelining, not
the MMA; comfy-kitchen's INT8 kernel is the fastest available, and is the engine's default.

## 2. Baselines (stock ComfyUI)

| 1344x768, 5 s (124 frames), 20 steps | s / step | warm, new seed | cold (first run) |
| --- | ---: | ---: | ---: |
| stock ComfyUI | 34.6-34.9 cold, 27.8-28.8 warm | **623 s** | 14:14 |
| stock + `--use-ck-attention` | 14.4 | **335 s** | 7:41 |

| 768x448, 2 s (56 frames), 20 steps | s / step | warm |
| --- | ---: | ---: |
| stock ComfyUI | 1.76 | 60.6 s |
| stock + `--use-ck-attention` | 1.40 | 52.3 s |

Published numbers on similar 16 GB cards agree (an RTX 5080 at 1344x768 reportedly ~36 s/step).

## 3. Profile of one engine block (S = 37,790, NVFP4, INT8 attention; `tools/profile_block.py`)

| Stage | ms | share |
| --- | ---: | ---: |
| attention (comfy-kitchen INT8) | 148.4 | 65.8% |
| fc1 GEMM (NVFP4, 5376 -> 28672) | 25.5 | 11.3% |
| qkv GEMM (5376 -> 21504) | 16.7 | 7.4% |
| fc2 GEMM (14336 -> 5376) | 14.3 | 6.3% |
| out GEMM (7168 -> 5376) | 7.1 | 3.1% |
| SwiGLU, gated residuals, q/k RMSNorm + RoPE, norm + modulation, transpose | 13.7 | 6.1% |
| **block** | **225.7** | x50 = 11.3 s |

## 4. Levers (measured)

| Lever | Effect |
| --- | --- |
| NVFP4 W4A4 linears on TensorFold's prompt GEMMs | 63.5 ms of GEMMs a block (3.2 s a step); weights 10.1 GiB instead of 21 GB int8 |
| comfy-kitchen INT8 attention instead of cuDNN bf16 | 23.9 -> 8.6 s of attention a step |
| Delayed activation scales (previous step's absmax x2, adaptive per linear; one batched read a forward instead of ~600 host syncs) | full forward 12.98 -> 11.45 s |
| Engine blocks outside ComfyUI's malloc-graph recording | in-ComfyUI step 14.4 -> 11.7 s (with the line above) |
| Weights streamed from pinned RAM, two slots deep | free at this length: all 50 streamed 12.56 s vs 38 resident 12.66 s |
| No 20 GB DiT swapped in and out around the VAE decode | warm "other" time ~35 s vs ~57 s stock |
| Turbo LoRA (lightx2v, 8 steps) through the stock LoRA node, merged into the NVFP4 weights once (95 s, cached) | 20 -> 8 steps |
| Fused Triton norm + modulation, gated residual, SwiGLU | ~14 ms a block of elementwise work |

Tried and not kept:
- TensorFold's fused gate/up GEMM with a SwiGLU epilogue writing FP4 rows for fc2 (the next step's fc2 scale estimated
  from the written e4m3 block scales): 11.45 -> 11.37-11.44 s a forward, within noise; not worth the extra path.
- SageAttention3 (FP4 attention) from a community Windows wheel built for cp313 / torch 2.10 / cu130 / sm_120a:
  "CUDA error: misaligned address" at every shape and head count on this machine.
- Keeping the checkpoint's Hadamard rotation for NVFP4 (rotating activations per 256): no accuracy
gain (NVFP4's 16-wide block scales already absorb the outliers int8 needed rotation for); Triton FP4 attention
(above); FP8 for one projection type at a time (each removes only 10-20% of the NVFP4 error).

## 5. Milestone 1: the loader node end to end

| 1344x768, 5 s, warm (new seed) | s / step | end to end | vs stock | vs stock + ck-attention |
| --- | ---: | ---: | ---: | ---: |
| stock ComfyUI, 20 steps | 28.3 | 623 s | | |
| stock + `--use-ck-attention`, 20 steps | 14.4 | 335 s | 1.86x | |
| **engine, NVFP4, 20 steps** | **11.7** | **273 s** | **2.28x** | **1.23x** |
| **engine, NVFP4 + Turbo LoRA, 8 steps** | **11.65** | **133 s** | **4.7x** | |

## 6. Correctness

`tools/check_forward.py` replays four real denoising calls captured from ComfyUI (`bench/comfy_dump_forward.py`,
768x448x56, steps 0 / 6 / 13 / 19) through ComfyUI's model shell with engine blocks:

| Engine precision | video cos (min-max) | audio cos | video relL2 |
| --- | --- | --- | --- |
| bf16 (dequantized int8 weights) | 0.9983-0.9998 | 0.9997-0.9999 | 0.017-0.057 |
| fp8 (per-tensor W8A8) | 0.9906-0.9991 | 0.9977-0.9995 | 0.042-0.136 |
| nvfp4 | 0.9561-0.9924 | 0.9812-0.9966 | 0.12-0.29 |

The bf16 engine matches ComfyUI to the noise of ComfyUI's own int8 activation quantization: the plumbing (packing,
modulation rows, RoPE, gates, PDD heads) is exact; NVFP4's difference is 4-bit weights and activations.

## 7. After milestone 1: memory, sparse attention, Turbo, the VAE

| 1344x768, 5 s, warm (new seed) | sampling | VAE decode | end to end |
| --- | ---: | ---: | ---: |
| engine, 20 steps | 234 s | 24 s | 273 s |
| engine + sparse attention (ComfyUI's node, sol-attn tau 1.3, dense for the first 20%) | ~150 s | 24 s | 192 s |
| engine + Turbo LoRA, 8 steps | 93 s | 24 s | 133 s |
| engine + Turbo + sparse | 67 s | 24 s | 96 s |
| engine + Turbo + sparse + Comfy-Org int8 video VAE | 67 s | 10 s | **81 s** |
| stock + ck-attention + Turbo (for reference) | 120 s | 41 s | 167 s |

- Memory: under ComfyUI's dynamic VRAM, its text encoder stays mapped (11.4 GiB) until something unloads it; the
  engine now asks ComfyUI to unload everything but the sampling model before a memory plan (0.6 -> 11.6 GiB free),
  so ~27 of 50 blocks stay resident at 37.8k tokens (all 50 at short videos). Streaming the rest costs nothing at
  this length (all 50 streamed: 12.56 s vs 38 resident: 12.66 s a forward).
- The video VAE decoder is a 36-layer ViT over 256 px tiles (4 x 7 tiles x 7 temporal chunks = 196 calls of ~123 ms
  for 1344x768x124, `tools/profile_vae.py`); 79% of it is fp16 GEMMs, which the int8 ConvRot VAE runs on int8 tensor
  cores (and INT8 attention): 23.9 -> 10.3 s, a stock option.
- Determinism: engine forwards are bitwise reproducible whatever the resident / streamed split; whole videos are
  identical across runs and ComfyUI restarts for dense and LoRA workflows. ComfyUI's sparse attention node is not
  reproducible across restarts, stock ComfyUI included (`det_bs1` vs `det_bs2`: max pixel difference 108).
- Text encoding a new prompt: ~50 s the first time (the 32B encoder streams in), seconds once warm; at 768x448 a
  new-prompt video is 28.5 s with the engine vs 62 s stock.
