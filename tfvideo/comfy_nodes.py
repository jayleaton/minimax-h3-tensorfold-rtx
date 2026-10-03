"""ComfyUI nodes: a MiniMax H3 loader whose MODEL runs the 50 DiT blocks on tfvideo (TensorFold kernels).

The MODEL is ComfyUI's own MiniMaxH3 (model sampling, packed layout, conditioning, final PDD heads, VAE latents) built
with no blocks and the checkpoint's small tensors; its ``blocks`` list is replaced by engine blocks. Text encoding,
sampling, guides and the VAEs stay ComfyUI's, so a workflow swaps one loader node and keeps everything else.

The engine stays resident between prompts (its weights in pinned host memory, as many blocks in VRAM as the sequence
leaves room for); when a sampling run ends (ComfyUI's ON_CLEANUP) it hands its VRAM back so the VAE decode has room.

LoRAs go through ComfyUI's own LoRA nodes: for an engine MODEL, ``load_lora_for_models`` hands the block-linear
patches to the engine (merged into the source weights with ComfyUI's ``calculate_weight`` and re-quantized, cached on
disk per LoRA set) and the rest (token refiner) to ComfyUI's patcher as usual.
"""

from __future__ import annotations

import hashlib
import logging

import torch

PRECISIONS = ["nvfp4", "int8", "nvfp4:edge=2", "fp8", "bf16-check"]
ATTENTION = ["auto", "fp4", "fp8", "int8", "sdpa", "cudnn"]
_ENGINE: dict = {}                      # one resident engine: {"key": (...), "engine": MiniMaxH3Engine}


def _engine(path: str, spec: str):
    from . import ext, store

    if _ENGINE.get("key") == (path, spec):
        return _ENGINE["engine"]
    _ENGINE.clear()
    torch.cuda.empty_cache()
    ext.use_prebuilt()
    engine = store.load_or_convert(path, spec, device="cuda", log=logging.info)
    engine.on_forward = _on_forward
    _ENGINE.update(key=(path, spec), engine=engine)
    return engine


# ------------------------------------------------------------------------------------------------ LoRA
LORA_KEY = "tfvideo_loras"          # transformer_options: [(fingerprint, {block key: adapter}, strength), ...]


def _fingerprint(part: dict) -> str:
    h = hashlib.sha1()
    for k in sorted(part):
        h.update(k.encode())
        for t in getattr(part[k], "weights", ()):
            if torch.is_tensor(t):
                h.update(str(tuple(t.shape)).encode())
                h.update(t.flatten()[:64].float().cpu().numpy().tobytes())
    return h.hexdigest()[:16]


def _on_forward(engine, S, transformer_options):
    """Block 0 of each forward: LoRA set in sync; before a new memory plan, let ComfyUI unload what it must (VAE,
    text encoder) so the blocks fit instead of streaming."""

    import comfy.model_management as mm

    _sync_loras(engine, transformer_options)
    if engine._plan_seq != S:
        # everything but the H3 model that is sampling (text encoder, VAEs) leaves VRAM; they reload from RAM when
        # their nodes run. Without this, dynamic-VRAM models stay mapped and the engine would stream every block.
        shell = getattr(engine, "comfy_model", None)
        keep = [lm for lm in mm.current_loaded_models if shell is not None and getattr(lm.model, "model", None) is shell]
        before = torch.cuda.mem_get_info(engine.device)[0]
        # ComfyUI matches loaded models by device equality: "cuda" != "cuda:0", so pass its own device object
        mm.free_memory(1 << 60, mm.get_torch_device(), keep_loaded=keep)
        mm.soft_empty_cache()
        logging.info(f"[tfvideo] made room: {before / 2**30:.1f} -> "
                     f"{torch.cuda.mem_get_info(engine.device)[0] / 2**30:.1f} GiB free")
        _log_vram(mm.get_torch_device())


def _log_vram(device):
    """Who holds VRAM (diagnostics for the memory plan under ComfyUI's dynamic VRAM)."""

    import comfy.model_management as mm

    gib = 2 ** 30
    models = ", ".join(f"{lm.model.model.__class__.__name__}={lm.model.loaded_size() / gib:.2f}"
                       for lm in mm.current_loaded_models if lm.model is not None)
    msg = (f"[tfvideo] vram: torch reserved {torch.cuda.memory_reserved(device) / gib:.2f} GiB, allocated "
           f"{torch.cuda.memory_allocated(device) / gib:.2f} GiB; loaded models: {models or 'none'}")
    try:
        import comfy_aimdo.model_vbar as mv
        msg += f"; aimdo: {mv.vbars_analyze(device.index if device.index is not None else 0)}"
    except Exception as e:  # noqa: BLE001
        msg += f"; aimdo: n/a ({type(e).__name__})"
    logging.info(msg[:2000])


def _sync_loras(engine, transformer_options):
    """Block 0 of each forward: if the MODEL's LoRA set differs from the engine's weights, re-merge (or reload)."""

    import comfy.lora

    from . import store

    loras = (transformer_options or {}).get(LORA_KEY, [])
    sig = "|".join(f"{fp}x{strength:.4f}" for fp, _, strength in loras)
    if sig == engine.lora_sig:
        return
    patches: dict[str, list] = {}
    for _, part, strength in loras:
        for key, adapter in part.items():
            name = key[len("diffusion_model."):-len(".weight")]
            patches.setdefault(name, []).append((strength, adapter, 1.0, None, None))

    def merge(name, w):
        p = patches.get(name)
        return comfy.lora.calculate_weight(p, w, name) if p else None

    what = "re-merging" if loras else "restoring base weights"
    logging.info(f"[tfvideo] LoRA set changed ({len(loras)} LoRA(s)): {what}")
    engine.release_vram()
    engine.blocks = []                      # free the old pinned weights before building the new ones
    new = store.load_or_convert(engine.source, engine.kind, device="cuda", log=logging.info,
                                merge=merge if loras else None, merge_sig=sig)
    engine.replace_blocks(new.blocks, sig)


def _install_lora_hook():
    import comfy.sd

    original = comfy.sd.load_lora_for_models
    if getattr(original, "_tfvideo", False):
        return

    def load_lora_for_models(model, clip, lora, strength_model, strength_clip, lora_metadata=None):
        engine = getattr(getattr(getattr(model, "model", None), "diffusion_model", None), "tfvideo_engine", None)
        if engine is None or strength_model == 0:
            return original(model, clip, lora, strength_model, strength_clip, lora_metadata)
        import comfy.lora
        import comfy.lora_convert

        from .minimax_h3 import SOURCE

        key_map = comfy.lora.model_lora_keys_unet(model.model, {})
        block_keys = set()
        for i in range(len(engine.blocks)):
            for src in SOURCE.values():
                k = f"diffusion_model.blocks.{i}.{src}.weight"
                key_map[k[:-len(".weight")]] = k
                key_map["lora_unet_blocks_{}_{}".format(i, src.replace(".", "_"))] = k
                block_keys.add(k)
        loaded = comfy.lora.load_lora(comfy.lora_convert.convert_lora(lora), key_map)
        part = {k: v for k, v in loaded.items() if k in block_keys}
        rest = {k: v for k, v in loaded.items() if k not in block_keys}
        new_model = model.clone()
        new_model.add_patches(rest, strength_model)
        to = dict(new_model.model_options.get("transformer_options", {}))
        to[LORA_KEY] = list(to.get(LORA_KEY, [])) + [(_fingerprint(part), part, float(strength_model))]
        new_model.model_options["transformer_options"] = to
        if lora_metadata:
            new_model.set_attachments("lora_metadata", lora_metadata)
        logging.info(f"[tfvideo] LoRA: {len(part)} block linears to the engine, {len(rest)} other weights to ComfyUI")
        new_clip = None
        if clip is not None:
            _, new_clip = original(None, clip, lora, 0.0, strength_clip, lora_metadata)
        return new_model, new_clip

    load_lora_for_models._tfvideo = True
    comfy.sd.load_lora_for_models = load_lora_for_models


def _probe_state_dict(path: str):
    """Every key of the checkpoint with its shape (meta tensors), for ComfyUI's model detection."""

    from .source import SafeTensors

    f = SafeTensors(path)
    return {k: torch.empty(f.shape(k), device="meta") for k in f.keys()}


def build_model(path: str, engine):
    """ComfyUI's MiniMaxH3 MODEL from the checkpoint's non-block tensors, with the engine's blocks."""

    import comfy.model_detection
    import comfy.model_management as mm
    import comfy.model_patcher
    import comfy.patcher_extension

    from .source import SafeTensors

    from .minimax_h3 import EngineBlock

    probe = _probe_state_dict(path)
    prefix = comfy.model_detection.unet_prefix_from_state_dict(probe)
    if not any(k.startswith(prefix) for k in probe):     # ComfyUI's loader falls back to bare keys the same way
        prefix = ""
    unet_config = comfy.model_detection.detect_unet_config(probe, prefix)
    if unet_config is None or unet_config.get("image_model") != "minimax_h3":
        raise RuntimeError(f"TensorFold MiniMax H3: {path} is not a MiniMax H3 DiT")
    unet_config["num_layers"] = 0
    f = SafeTensors(path)                   # only the small tensors: the blocks are the engine's
    sd = {k[len(prefix):]: f.get(k) for k in f.keys()
          if k.startswith(prefix) and not k[len(prefix):].startswith("blocks.")}
    model_config = comfy.model_detection.model_config_from_unet_config(unet_config, sd)
    load_device = mm.get_torch_device()
    model_config.set_inference_dtype(torch.bfloat16, torch.bfloat16, device=load_device)
    model = model_config.get_model(sd, "")
    model.diffusion_model.blocks = torch.nn.ModuleList(EngineBlock(engine, i) for i in range(len(engine.blocks)))
    model.diffusion_model.tfvideo_engine = engine
    engine.comfy_model = model
    patcher = comfy.model_patcher.CoreModelPatcher(model, load_device=load_device,
                                                   offload_device=mm.unet_offload_device())
    model.load_model_weights(sd, "", assign=patcher.is_dynamic())
    patcher.add_wrapper(comfy.patcher_extension.WrappersMP.DIFFUSION_MODEL, _wrapper(model, engine))
    patcher.add_callback_with_key(comfy.patcher_extension.CallbacksMP.ON_CLEANUP, "tfvideo", _cleanup(engine))
    return patcher


def _wrapper(model, engine):
    """Around each diffusion-model call: refuse what the engine cannot honour."""

    def wrapper(executor, x, timestep, context, transformer_options={}, **kwargs):
        patcher = getattr(model, "current_patcher", None)
        if patcher is not None and getattr(patcher, "hook_patches", None):
            raise RuntimeError("TensorFold MiniMax H3: hook (scheduled) weight patches are not supported")
        return executor(x, timestep, context, transformer_options, **kwargs)
    return wrapper


def _cleanup(engine):
    """End of a sampling run (ComfyUI's ON_CLEANUP): hand the VRAM back for the VAE decode, reset the scales."""

    def cleanup(patcher):
        engine.release_vram()
        engine.end_run()
    return cleanup


def _model_files() -> list[str]:
    import folder_paths

    return [n for n in folder_paths.get_filename_list("diffusion_models") if n.endswith(".safetensors")]


class TFMiniMaxH3Loader:
    """Load a MiniMax H3 DiT into the TensorFold video engine (converted once, cached on internal disk)."""

    @classmethod
    def INPUT_TYPES(cls):
        files = _model_files()
        default = next((f for f in files if "minimax_h3" in f), files[0] if files else "")
        return {"required": {"unet_name": (files, {"default": default}),
                             "precision": (PRECISIONS, {"default": "nvfp4"}),
                             "attention": (ATTENTION, {"default": "auto"})}}

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "load"
    CATEGORY = "TensorFold"

    def load(self, unet_name: str, precision: str, attention: str = "auto"):
        import folder_paths

        from . import attention as A

        _install_lora_hook()
        path = folder_paths.get_full_path_or_raise("diffusion_models", unet_name)
        spec = "bf16" if precision == "bf16-check" else precision
        engine = _engine(path, spec)
        engine.attn, engine.attn_name = A.get(attention), attention
        logging.info(f"[tfvideo] MiniMax H3 engine ready: {spec}, attention {attention}, "
                     f"{engine.nbytes() / 2**30:.2f} GiB of block weights")
        return (build_model(path, engine),)


NODE_CLASS_MAPPINGS = {"TFMiniMaxH3Loader": TFMiniMaxH3Loader}
NODE_DISPLAY_NAME_MAPPINGS = {"TFMiniMaxH3Loader": "TensorFold MiniMax H3 Loader"}
