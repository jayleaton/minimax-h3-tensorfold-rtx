# AGENTS.md: setting this up on a Windows PC with an RTX 50-series GPU

Instructions for AI coding agents (and people) bringing up the TensorFold MiniMax H3 loader in an existing ComfyUI.
Follow the steps in order; run each check and stop on a failure. Ask the user before anything that changes state
outside this repo (installing Visual Studio, linking into their ComfyUI, deleting files).
Never modify the user's ComfyUI install: the benchmarks run it headless with every directory redirected into `runs/`
and an in-memory database.

## 1. Check the machine

| Check | Command | Expect |
| --- | --- | --- |
| GPU | `nvidia-smi --query-gpu=name,compute_cap,driver_version --format=csv,noheader` | compute capability `12.0` (RTX 50-series); 617.x is known good |
| ComfyUI portable | `"%COMFYUI_PORTABLE%\python_embeded\python.exe" -c "import torch, comfy_kitchen;print(torch.__version__, comfy_kitchen.__version__)"` | `2.10.0+cu130 0.2.35` (the prebuilt kernels must match torch and Python 3.13) |
| ComfyUI | `type "%COMFYUI_PORTABLE%\ComfyUI\comfyui_version.py"` | 0.37.0 or later (MiniMax H3 support) |
| Models | `diffusion_models\minimax_h3_fl2va_pruned_int8_convrot.safetensors`, `text_encoders\qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors`, `vae\minimax_h3_video_vae_fp16.safetensors`, `vae\minimax_h3_audio_vae_fp32.safetensors` | present |
| RAM | 64 GB recommended | the engine keeps its 10 GB of NVFP4 weights in pinned memory beside ComfyUI's text encoder and VAEs |
| Visual Studio | `"%ProgramFiles(x86)%\Microsoft Visual Studio\Installer\vswhere.exe" -latest -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath` | a path (VS 2022 or 2026 with the C++ x64 tools) |
| uv, git, ffmpeg | `uv --version`, `git --version`, `ffmpeg -version` | present (ffmpeg only for `bench\quality.py`) |
| Disk | free space on the drive holding `%LOCALAPPDATA%` (or `TFVIDEO_CACHE`) | >= 12 GB per converted engine, preferably internal NVMe |

`COMFYUI_PORTABLE` is the folder holding `python_embeded\` and `ComfyUI\`. If the DiT is elsewhere, set
`MINIMAX_H3_DIT` to its full path.

## 2. Build

```bat
git submodule update --init
scripts\setup.cmd
```

Expect `tensorfold.cuda.nvfp4.checkpoint._ext OK` and `tensorfold.cuda.kernels.qmm._ext OK`, then `setup done`.
If nvcc fails with `cudafe++ died with status 0xC0000005`, the CUDA 13.0 compiler is being used with VS 2026: the
setup pins 13.4; check `.venv\Lib\site-packages\nvidia\cu13\bin\nvcc --version`.

## 3. Check the engine against ComfyUI (optional, recommended)

```bat
"%COMFYUI_PORTABLE%\python_embeded\python.exe" -B bench\comfy_dump_forward.py --tag small   :: ComfyUI's own forward on a real prompt
"%COMFYUI_PORTABLE%\python_embeded\python.exe" -B tools\check_forward.py small nvfp4         :: engine vs ComfyUI, per captured step
```

Expect video cosine >= 0.95 at every captured step for `nvfp4` (>= 0.998 for `bf16`, which needs ~40 GB of RAM).
The first engine load converts the DiT (~2 minutes; reading 21 GB) and caches it in `%LOCALAPPDATA%\tfvideo`.

## 4. Measure against the user's ComfyUI (optional, recommended)

```bat
tools\env.cmd python bench\comfy_h3.py --tag base --repeat 2                    :: stock ComfyUI, blueprint settings
tools\env.cmd python bench\comfy_h3.py --tag tf --repeat 2 --tf nvfp4            :: the TensorFold loader
tools\env.cmd python bench\quality.py base tf                                     :: frames (LPIPS / DINO / PSNR) and audio
```

The `--tf` run fails on purpose if ComfyUI's log shows no `[tfvideo]` engine. Look at the contact sheet
`runs\quality_tf.jpg` with the user.

## 5. Install the node (ask first)

Either a junction (ComfyUI then loads it like any custom node):

```bat
mklink /J "%COMFYUI_PORTABLE%\ComfyUI\custom_nodes\ComfyUI-TensorFold-Video" "%CD%\comfyui\ComfyUI-TensorFold-Video"
```

or, when ComfyUI is on an exFAT/FAT drive (`mklink /J` fails with "Local NTFS volumes are required"), an
`extra_model_paths.yaml` in the ComfyUI folder with `custom_nodes:` pointing at this repo's `comfyui\` folder:

```yaml
# <ComfyUI>\extra_model_paths.yaml
tensorfold_video:
  base_path: C:/path/to/minimax-h3-tensorfold-rtx
  custom_nodes: comfyui
```

Restart ComfyUI and load `workflows\Text to Video (MiniMax H3, TensorFold).json` (fastest settings; it needs the
int8 video VAE and the Turbo LoRA listed in the README), or in an existing MiniMax H3 workflow replace `Load Diffusion Model` (UNETLoader) with **TensorFold MiniMax H3 Loader**,
same file, precision `nvfp4`. LoRA nodes after it keep working (the engine merges them; first use of a LoRA set
converts once). The node finds this repo through its own location (or `TFVIDEO_REPO`).

Works with the engine: LoRA nodes, ComfyUI's Model Sparse Attention node. Refused (an error, by design): scheduled
hook weight patches.
