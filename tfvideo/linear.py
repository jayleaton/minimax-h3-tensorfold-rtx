"""Linear layers for DiT blocks: bf16 (reference), FP8 W8A8 and NVFP4 W4A4 on TensorFold's prompt GEMMs.

A DiT step multiplies thousands of rows by every weight, so it is compute bound: the win is the tensor-core rate of the
operand format (bf16 ~85, FP8 ~170, NVFP4 ~400 TF/s on an RTX 5070 Ti), not the bytes read. Activations are quantized
under a per-layer input scale ``act``: measured on the fly (``act=None``, one host sync a call, for calibration) or
fixed from a calibration run (no sync, CUDA-graph safe). Per-16 (NVFP4) block scales still follow every row's range.
"""

from __future__ import annotations

from functools import lru_cache

import torch
import torch.nn.functional as F

FP4_MAX, E4M3_MAX = 6.0, 448.0
TILES = (0, 1, 2, 12, 13)        # TensorFold prompt tiles worth trying on sm_120 (12-13: bulk-copy tiles)


def _ck():
    from tensorfold.cuda.nvfp4 import checkpoint
    return checkpoint


@lru_cache(maxsize=1)
def nvfp4_supported() -> bool:
    return torch.cuda.is_available() and torch.cuda.get_device_capability()[0] == 12


def nvfp4_weight(w: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, float]:
    """bf16 [N, K] -> (e2m1 codes [N, K/2], e4m3 scales [N, K/16], global scale), ModelOpt's NVFP4 layout."""

    ck = _ck()
    n, k = w.shape
    g = max(float(w.float().abs().amax()), 1e-12) / (FP4_MAX * E4M3_MAX)
    rows = ck.quant4(w.to(torch.bfloat16).contiguous(), g)
    scales = rows.scales[:, :n, :].permute(1, 0, 2).reshape(n, k // 16)
    return rows.codes, scales.view(torch.float8_e4m3fn), g


def absmax(x: torch.Tensor) -> torch.Tensor:
    """max |x| as an fp32 scalar tensor, without materializing |x| (rows can be ~1 GB)."""

    lo, hi = torch.aminmax(x)
    return torch.maximum(-lo, hi).float()


class Linear:
    """y = x @ W^T for 2-D bf16 rows; ``kind`` names the backend."""

    kind = "base"
    n: int
    k: int

    def __call__(self, x: torch.Tensor, out: torch.Tensor | None = None) -> torch.Tensor:
        raise NotImplementedError

    def nbytes(self) -> int:
        raise NotImplementedError


class Bf16Linear(Linear):
    kind = "bf16"

    def __init__(self, w: torch.Tensor):
        self.w = w.to(torch.bfloat16).contiguous()
        self.n, self.k = self.w.shape

    def __call__(self, x: torch.Tensor, out: torch.Tensor | None = None) -> torch.Tensor:
        w = self.w if self.w.device == x.device else self.w.to(x.device, non_blocking=True)
        return torch.matmul(x, w.t(), out=out)

    def nbytes(self) -> int:
        return self.w.numel() * 2


class Fp8Linear(Linear):
    """W8A8 e4m3: one weight scale and one input scale (tensor-wise, what cuBLASLt takes on sm_120), ``_scaled_mm``."""

    kind = "fp8"

    def __init__(self, w: torch.Tensor, act: float | None = None):
        w = w.float()
        self.n, self.k = w.shape
        s = (w.abs().amax() / E4M3_MAX).clamp_min(1e-12)
        self.w8 = (w / s).to(torch.float8_e4m3fn).contiguous()
        self.ws = s.view(()).contiguous()                       # fp32 scalar
        self.act = act
        self._a = None

    def _act(self, x: torch.Tensor) -> torch.Tensor:
        if self.act is None:
            return (absmax(x) / E4M3_MAX).clamp_min(1e-12)
        if self._a is None or self._a.device != x.device:
            self._a = torch.tensor(self.act, dtype=torch.float32, device=x.device)
        return self._a

    def __call__(self, x: torch.Tensor, out: torch.Tensor | None = None) -> torch.Tensor:
        a = self._act(x)
        m = x.shape[0]
        pad = -m % 16
        xq = (x.float() / a).clamp(-E4M3_MAX, E4M3_MAX).to(torch.float8_e4m3fn)
        if pad:
            xq = F.pad(xq.view(torch.uint8), (0, 0, 0, pad)).view(torch.float8_e4m3fn)
        y = torch._scaled_mm(xq, self.w8.t(), scale_a=a, scale_b=self.ws, out_dtype=torch.bfloat16)
        y = y[:m] if pad else y
        return out.copy_(y) if out is not None else y

    def nbytes(self) -> int:
        return self.w8.numel() + 4


class Nvfp4Linear(Linear):
    """W4A4 NVFP4 (e2m1 values, e4m3 scale per 16, fp32 global) on TensorFold's block-scaled FP4 prompt GEMM.

    The input's global scale: measured per call (``dynamic``: one host sync a call), or delayed (``next_act``: the
    previous forward's absmax times DELAY_MARGIN, while this forward's absmax accumulates on the GPU in
    ``amax_dev``; the engine reads all of them back in one transfer per forward, see ``update_delayed``)."""

    kind = "nvfp4"
    next_act: float | None = None
    amax_dev: torch.Tensor | None = None
    used_act: float | None = None
    margin: float | None = None      # delayed-scale margin, doubled for this linear after a clipped forward

    def __init__(self, w: torch.Tensor, act: float | None = None, tile: int = 12):
        from tensorfold.cuda.nvfp4.linear import Fp4Linear

        codes, scales, g = nvfp4_weight(w)
        self.lin = Fp4Linear.from_checkpoint(codes, scales, g, act=act if act is not None else 1.0)
        self.n, self.k = self.lin.n, self.lin.k
        self.dynamic = act is None
        self.tile = tile
        self.amax = 0.0                 # largest input scale seen (calibration)

    def set_act(self, act: float | None) -> None:
        self.dynamic = act is None
        if act is not None:
            self.lin.act = float(act)

    def __call__(self, x: torch.Tensor, out: torch.Tensor | None = None) -> torch.Tensor:
        if self.dynamic:
            m = absmax(x)
            self.amax_dev = m if self.amax_dev is None else torch.maximum(self.amax_dev, m)
            if self.next_act is not None:
                self.lin.act = self.used_act = self.next_act
            else:
                a = max(float(m), 1e-6) / (FP4_MAX * E4M3_MAX)
                self.amax = max(self.amax, a)
                self.lin.act = a
        return _ck().prompt(_ck().A4, x, self.lin, out=out, tile=self.tile)

    def nbytes(self) -> int:
        return self.lin.words.numel() * 4 + self.lin.bs.numel()


class Int8Linear(Linear):
    """W8A8 int8 in ComfyUI's ConvRot basis (comfy-kitchen's int8 GEMM: inputs Hadamard-rotated per 256 and quantized
    per row on the fly). Same numerics as stock ComfyUI on an int8_convrot checkpoint; the fidelity precision."""

    kind = "int8"

    def __init__(self, q: torch.Tensor, scale: torch.Tensor, group: int = 256):
        self.q, self.scale, self.group = q.contiguous(), scale.float().reshape(-1, 1).contiguous(), group
        self.n, self.k = self.q.shape

    @classmethod
    def from_float(cls, w: torch.Tensor, group: int = 256) -> "Int8Linear":
        """fp32 [N, K] in the model's basis -> rotated, per-row int8 (absmax / 127), like comfy-kitchen's quantizer."""

        from .source import hadamard

        n, k = w.shape
        wr = (w.float().view(n, k // group, group) @ hadamard(group, str(w.device))).view(n, k)
        s = (wr.abs().amax(1, keepdim=True) / 127.0).clamp_min(1e-12)
        return cls((wr / s).round().clamp(-127, 127).to(torch.int8), s, group)

    def __call__(self, x: torch.Tensor, out: torch.Tensor | None = None) -> torch.Tensor:
        from comfy_kitchen.tensor.int8 import _dtype_code

        y = torch.ops.comfy_kitchen.int8_linear(x.contiguous(), self.q, self.scale, None, _dtype_code(torch.bfloat16),
                                                True, self.group)
        return out.copy_(y) if out is not None else y

    def nbytes(self) -> int:
        return self.q.numel() + self.scale.numel() * 4


DELAY_MARGIN = 2.0


def update_delayed(lins, margin: float = DELAY_MARGIN) -> int:
    """After a forward: one device-to-host read of every dynamic NVFP4 linear's input absmax; the next forward
    quantizes under absmax * margin with no host sync. Returns how many linears saw inputs beyond the range they
    were quantized under this forward (clipped; they adapt on the next one)."""

    lins = [lin for lin in lins if isinstance(lin, Nvfp4Linear) and lin.dynamic and lin.amax_dev is not None]
    if not lins:
        return 0
    vals = torch.stack([lin.amax_dev for lin in lins]).float().cpu().tolist()
    clipped = 0
    for lin, v in zip(lins, vals):
        m = lin.margin or margin
        if lin.used_act is not None and v > lin.used_act * FP4_MAX * E4M3_MAX:
            clipped += 1
            m = min(m * 2, 16.0)
        lin.margin = m
        lin.next_act = max(v, 1e-6) / (FP4_MAX * E4M3_MAX) * m
        lin.amax_dev, lin.used_act = None, None
    return clipped


def reset_delayed(lins) -> None:
    for lin in lins:
        if isinstance(lin, Nvfp4Linear):
            lin.next_act = lin.amax_dev = lin.used_act = lin.margin = None


def swiglu_mlp(gate_up: Linear, down: Linear, x: torch.Tensor):
    """down(silu(gate) * up) in one TensorFold pass when both are static-scale NVFP4: the gate/up GEMM's epilogue
    writes the SwiGLU rows as NVFP4 under down's input scale, so no bf16 [M, 2F] or [M, F] tensor exists. None if
    the layers cannot take it (the caller runs the unfused path)."""

    if not (isinstance(gate_up, Nvfp4Linear) and isinstance(down, Nvfp4Linear)) or gate_up.dynamic or down.dynamic:
        return None
    f = gate_up.n // 2
    if f % 64 or down.k != f:
        return None
    if getattr(gate_up, "_halves", None) is None:
        t = f // 64
        gate_up._halves = (gate_up.lin.tiles(0, t), gate_up.lin.tiles(t, 2 * t))
    gate, up = gate_up._halves
    return _ck().mlp_prompt(x, gate, up, down.lin)


def make(kind: str, w: torch.Tensor, act: float | None = None) -> Linear:
    """A backend by name for a bf16 [N, K] weight (dims NVFP4 cannot tile fall back to FP8, then bf16)."""

    n, k = w.shape
    if kind == "int8" and k % 256 == 0:
        return Int8Linear.from_float(w)
    if kind == "nvfp4" and nvfp4_supported() and k % 128 == 0 and n % 64 == 0:
        return Nvfp4Linear(w, act)
    if kind in ("nvfp4", "fp8") and k % 16 == 0 and n % 16 == 0:
        return Fp8Linear(w, act)
    return Bf16Linear(w)
