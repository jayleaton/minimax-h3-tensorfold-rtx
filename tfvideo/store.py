"""Converting a MiniMax H3 checkpoint to engine linears, and the converted engines on disk.

Converting reads every block linear (19.3B int8 weights, 21 GB; slow from an external drive), rotates it back to the
model's basis, merges LoRAs if any (``merge``: ComfyUI's own weight patching), and quantizes. The result (~10.8 GB for
NVFP4) goes to a cache keyed by the source file's path, size and mtime, the precision spec and the LoRA set's
signature; loading it back takes seconds from NVMe.
Every block's big tensors are kept in pinned host memory: the engine decides per sequence length which blocks also
stay in VRAM and streams the rest.
"""

from __future__ import annotations

import hashlib
import json
import os
import time
from pathlib import Path

import torch
from safetensors.torch import save_file

from . import linear as L
from .minimax_h3 import LIN_NAMES, SOURCE, Block, Config, MiniMaxH3Engine, _lin_tensors
from .source import Checkpoint, SafeTensors

FORMAT = 1


def cache_dir() -> Path:
    base = os.environ.get("TFVIDEO_CACHE") or os.path.join(os.environ.get("LOCALAPPDATA", str(Path.home())), "tfvideo")
    path = Path(base)
    path.mkdir(parents=True, exist_ok=True)
    return path


def _file_id(path: str | Path) -> str:
    st = os.stat(path)
    return f"{Path(path).resolve()}|{st.st_size}|{st.st_mtime_ns}"


def key(source: str | Path, spec: str, merge_sig: str = "") -> str:
    raw = f"{_file_id(source)}|{spec}|{FORMAT}|{merge_sig}"
    tag = spec.replace(":", "_").replace(",", "_").replace("=", "-") + ("-lora" if merge_sig else "")
    return f"{Path(source).stem}-{tag}-{hashlib.sha1(raw.encode()).hexdigest()[:10]}"


def policy(spec: str, layers: int):
    """``"nvfp4"`` or ``"nvfp4:fc2=fp8,edge=2"``: a base kind, per-projection overrides, and ``edge`` blocks at each
    end kept at ``edge_kind`` (default fp8). Returns (block, projection) -> kind."""

    base, _, rest = spec.partition(":")
    over = dict(kv.split("=") for kv in rest.split(",") if kv)
    edge = int(over.pop("edge", 0))
    edge_kind = over.pop("edge_kind", "fp8")

    def pick(i: int, name: str) -> str:
        if i < edge or i >= layers - edge:
            return edge_kind
        return over.get(name, base)
    return pick


def _to_host(block: Block, pin: bool = True) -> None:
    """Move a block's big tensors to (pinned) host memory (the engine re-places them per sequence length)."""

    for name in LIN_NAMES:
        for t, (holder, attr) in _lin_tensors(block.lins[name]).items():
            host = getattr(holder, attr).cpu()
            block.host[f"{name}.{t}"] = host.pin_memory() if pin else host
            setattr(holder, attr, None)
    block.resident = False


def _small(src: Checkpoint, i: int, device) -> dict[str, torch.Tensor]:
    p = f"blocks.{i}."
    return {"norm1": src.get(p + "norm1.weight").to(device, torch.float32),
            "norm2": src.get(p + "norm2.weight").to(device, torch.float32),
            "q_norm": src.get(p + "attn.q_norm.weight").to(device),
            "k_norm": src.get(p + "attn.k_norm.weight").to(device),
            "ada_w": src.get(p + "adaln_proj.linear.weight").to(device, torch.float32),
            "ada_b": src.get(p + "adaln_proj.linear.bias").to(device, torch.float32)}


def convert(source: str | Path, spec: str = "nvfp4", merge=None, device="cuda", log=print) -> MiniMaxH3Engine:
    """``merge(name, w)``: the patched fp32 weight of block linear ``name`` (e.g. ``blocks.3.mlp.fc1``), or None."""

    src = Checkpoint(source)
    pick = policy(spec, src.layers)
    blocks, merged, t0 = [], 0, time.perf_counter()
    for i in range(src.layers):
        lins = {}
        for name in LIN_NAMES:
            lin_name = f"blocks.{i}.{SOURCE[name]}"
            kind = pick(i, name)
            if kind == "int8" and merge is None and src.quant_conf(lin_name):
                lins[name] = L.Int8Linear(src.get(lin_name + ".weight").to(device),
                                          src.get(lin_name + ".weight_scale").to(device))
                continue
            w = src.linear(lin_name, device)
            if merge is not None:
                patched = merge(lin_name, w)
                if patched is not None:
                    w = patched
                    merged += 1
            lins[name] = L.make(kind, w if kind == "int8" else w.to(torch.bfloat16))
            del w
        b = Block(lins=lins, **_small(src, i, device))
        _to_host(b, pin=spec.split(":")[0] != "bf16")       # bf16 (39 GB) is a reference: not page-locked
        blocks.append(b)
        if i % 10 == 9:
            log(f"[tfvideo] converted {i + 1}/{src.layers} blocks ({time.perf_counter() - t0:.0f} s)")
    if merge is not None:
        log(f"[tfvideo] merged LoRA patches into {merged} block linears")
    return MiniMaxH3Engine(blocks, spec, Config(), device)


def _lin_meta(lin: L.Linear) -> dict:
    if isinstance(lin, L.Nvfp4Linear):
        f = lin.lin
        return {"kind": "nvfp4", "scale": f.scale, "n": f.n, "k": f.k, "tile": lin.tile}
    if isinstance(lin, L.Fp8Linear):
        return {"kind": "fp8", "n": lin.n, "k": lin.k, "ws": float(lin.ws)}
    if isinstance(lin, L.Int8Linear):
        return {"kind": "int8", "n": lin.n, "k": lin.k, "group": lin.group}
    return {"kind": "bf16", "n": lin.n, "k": lin.k}


def save(engine: MiniMaxH3Engine, path: Path) -> None:
    tensors, meta = {}, {"format": FORMAT, "kind": engine.kind, "linears": {}}
    for i, b in enumerate(engine.blocks):
        for small in ("norm1", "norm2", "q_norm", "k_norm", "ada_w", "ada_b"):
            tensors[f"{i}.{small}"] = getattr(b, small)
        for k, t in b.host.items():
            tensors[f"{i}.{k}"] = t.view(torch.uint8) if t.dtype == torch.float8_e4m3fn else t
        for name in LIN_NAMES:
            meta["linears"][f"{i}.{name}"] = _lin_meta(b.lins[name])
    tmp = path.with_suffix(".tmp")
    save_file({k: v.contiguous().cpu() for k, v in tensors.items()}, str(tmp), metadata={"tfvideo": json.dumps(meta)})
    os.replace(tmp, path)


def _lin_load(m: dict) -> L.Linear:
    """A linear with no device tensors yet (the engine places them)."""

    if m["kind"] == "nvfp4":
        from tensorfold.cuda.nvfp4.linear import Fp4Linear

        lin = L.Nvfp4Linear.__new__(L.Nvfp4Linear)
        lin.lin = Fp4Linear(None, None, m["scale"], m["n"], m["k"], act=1.0)
        lin.n, lin.k, lin.dynamic, lin.tile, lin.amax = m["n"], m["k"], True, m.get("tile", 12), 0.0
        return lin
    if m["kind"] == "fp8":
        lin = L.Fp8Linear.__new__(L.Fp8Linear)
        lin.w8, lin.ws = None, torch.tensor(m["ws"], dtype=torch.float32)
        lin.n, lin.k, lin.act, lin._a = m["n"], m["k"], None, None
        return lin
    if m["kind"] == "int8":
        lin = L.Int8Linear.__new__(L.Int8Linear)
        lin.q, lin.scale, lin.n, lin.k, lin.group = None, None, m["n"], m["k"], m["group"]
        return lin
    lin = L.Bf16Linear.__new__(L.Bf16Linear)
    lin.w, lin.n, lin.k = None, m["n"], m["k"]
    return lin


def load(path: Path, device="cuda", log=print) -> MiniMaxH3Engine:
    f = SafeTensors(path)
    meta = json.loads(f.metadata["tfvideo"])
    cfg = Config()
    blocks = []
    for i in range(cfg.layers):
        lins = {name: _lin_load(meta["linears"][f"{i}.{name}"]) for name in LIN_NAMES}
        b = Block(lins=lins, **{s: f.get(f"{i}.{s}").to(device) for s in
                                ("norm1", "norm2", "q_norm", "k_norm", "ada_w", "ada_b")})
        for name in LIN_NAMES:
            if isinstance(lins[name], L.Fp8Linear):
                lins[name].ws = lins[name].ws.to(device)
            for t in _lin_tensors(lins[name]):
                host = f.get(f"{i}.{name}.{t}")
                if t == "w8":
                    host = host.view(torch.float8_e4m3fn)
                b.host[f"{name}.{t}"] = host.pin_memory()
        b.resident = False
        blocks.append(b)
    return MiniMaxH3Engine(blocks, meta["kind"], cfg, device)


def load_or_convert(source: str | Path, spec: str, device="cuda", log=print, merge=None,
                    merge_sig: str = "") -> MiniMaxH3Engine:
    if spec.split(":")[0] == "int8":
        # the checkpoint's own int8 weights (20 GB): nothing to convert, so nothing cached; read from the source
        log(f"[tfvideo] loading {Path(source).name} int8 weights{' with LoRA' if merge_sig else ''} (not cached)")
        engine = convert(source, spec, merge, device, log)
        engine.source, engine.lora_sig = str(source), merge_sig
        return engine
    path = cache_dir() / (key(source, spec, merge_sig) + ".safetensors")
    if path.is_file():
        t0 = time.perf_counter()
        engine = load(path, device, log)
        log(f"[tfvideo] loaded converted {path.name} in {time.perf_counter() - t0:.1f} s")
    else:
        what = " with LoRA" if merge_sig else ""
        log(f"[tfvideo] converting {Path(source).name} to {spec}{what} (once; cached at {path})")
        engine = convert(source, spec, merge, device, log)
        save(engine, path)
    engine.source, engine.lora_sig = str(source), merge_sig
    return engine
