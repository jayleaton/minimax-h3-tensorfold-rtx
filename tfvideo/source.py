"""Reading a MiniMax H3 DiT checkpoint as ComfyUI stores it.

The block linears may be plain floats or ComfyUI's ``int8_tensorwise`` + ConvRot: int8 rows q with one fp32 scale a
row, quantized in a rotated basis (each 256-input group multiplied by a normalized regular Hadamard matrix H, which is
symmetric and its own inverse). The weight in the model's basis is W = (q * s) @ H per group.
"""

from __future__ import annotations

import json
import math
import re
import struct
from functools import lru_cache
from pathlib import Path

import torch

LINEARS = ("attn.qkv_proj", "attn.out_proj", "mlp.fc1", "mlp.fc2")


@lru_cache(maxsize=4)
def hadamard(size: int, device: str = "cpu") -> torch.Tensor:
    """comfy-kitchen's ConvRot matrix: kron powers of a 4x4 regular Hadamard, divided by sqrt(size)."""

    if size < 4 or math.log(size, 4) % 1:
        raise ValueError(f"ConvRot group size must be a power of 4, got {size}")
    h4 = torch.tensor([[1, 1, 1, -1], [1, 1, -1, 1], [1, -1, 1, 1], [-1, 1, 1, 1]], dtype=torch.float32)
    h = h4
    while h.shape[0] < size:
        h = torch.kron(h, h4)
    return (h / math.sqrt(size)).to(device)


DTYPES = {"F64": torch.float64, "F32": torch.float32, "F16": torch.float16, "BF16": torch.bfloat16,
          "I64": torch.int64, "I32": torch.int32, "I16": torch.int16, "I8": torch.int8, "U8": torch.uint8,
          "BOOL": torch.bool, "F8_E4M3": torch.float8_e4m3fn, "F8_E5M2": torch.float8_e5m2}


class SafeTensors:
    """Plain reads of a .safetensors file (header + byte offsets): no mmap, no safetensors handle. ComfyUI's server
    patches file loading for its own models, and a second safetensors handle on the same file crashed there."""

    def __init__(self, path: str | Path):
        self.path = str(path)
        with open(self.path, "rb") as f:
            n = struct.unpack("<Q", f.read(8))[0]
            header = json.loads(f.read(n))
        self.metadata = header.pop("__metadata__", None) or {}
        self.header, self.base = header, 8 + n

    def keys(self):
        return self.header.keys()

    def shape(self, name: str) -> list[int]:
        return self.header[name]["shape"]

    def get(self, name: str) -> torch.Tensor:
        h = self.header[name]
        a, b = h["data_offsets"]
        with open(self.path, "rb") as f:
            f.seek(self.base + a)
            buf = bytearray(f.read(b - a))
        t = torch.frombuffer(buf, dtype=DTYPES[h["dtype"]]) if b > a else torch.empty(0, dtype=DTYPES[h["dtype"]])
        return t.reshape(h["shape"])


class Checkpoint:
    """A safetensors DiT: tensor access by name, block linears as fp32 weights in the model's basis."""

    def __init__(self, path: str | Path):
        self.path = str(path)
        self.f = SafeTensors(self.path)
        self.keys = set(self.f.keys())
        prefix = "model.diffusion_model."
        self.prefix = prefix if prefix + "video_patch_proj.weight" in self.keys else ""
        self.layers = 1 + max(int(m.group(1)) for k in self.keys if (m := re.match(re.escape(self.prefix) + r"blocks\.(\d+)\.", k)))

    def has(self, name: str) -> bool:
        return self.prefix + name in self.keys

    def get(self, name: str) -> torch.Tensor:
        return self.f.get(self.prefix + name)

    def names(self):
        n = len(self.prefix)
        return sorted(k[n:] for k in self.keys if k.startswith(self.prefix))

    def quant_conf(self, lin: str) -> dict | None:
        name = lin + ".comfy_quant"
        if not self.has(name):
            return None
        return json.loads(bytes(self.get(name).tolist()).decode())

    def linear(self, lin: str, device="cuda") -> torch.Tensor:
        """The weight of ``lin`` (e.g. ``blocks.3.mlp.fc1``) as fp32 [N, K] in the model's basis, on ``device``."""

        w = self.get(lin + ".weight").to(device)
        conf = self.quant_conf(lin)
        if conf is None:
            return w.float()
        if conf.get("format") != "int8_tensorwise":
            raise ValueError(f"{lin}: unsupported ComfyUI quant format {conf}")
        w = w.float() * self.get(lin + ".weight_scale").to(device).float()
        if conf.get("convrot"):
            g = int(conf.get("convrot_groupsize", 256))
            n, k = w.shape
            w = (w.view(n, k // g, g) @ hadamard(g, str(device))).view(n, k)
        return w
