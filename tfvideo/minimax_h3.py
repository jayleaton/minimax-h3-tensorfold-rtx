"""MiniMax H3's 50 DiT blocks on tfvideo linears (hidden 5376, 56 heads x 128, SwiGLU 14336, ~19.3B parameters).

ComfyUI's ``MiniMaxH3Model`` keeps everything around the blocks (packing the [text | cond | audio | video] sequence,
per-token timesteps, RoPE table, final PDD heads); its ``blocks`` list is replaced by ``EngineBlock`` modules that run
on this engine. A block: RMSNorm + modulation -> qkv -> q/k RMSNorm + split-half RoPE (comfy-kitchen, in place) ->
attention -> out -> gated residual; RMSNorm + modulation -> fc1 -> SwiGLU -> fc2 -> gated residual.

Weights: NVFP4 (or FP8 / bf16) linears. A block's NVFP4 weights are ~216 MB and one 40k-token block computes for
~0.4 s, so blocks that do not fit beside the activations stream from pinned host memory on a copy stream, two slots
deep, and the copy hides behind the previous block's compute.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field

import torch

from . import attention as A
from . import kernels as K
from . import linear as L

LIN_NAMES = ("qkv", "out", "fc1", "fc2")
SOURCE = {"qkv": "attn.qkv_proj", "out": "attn.out_proj", "fc1": "mlp.fc1", "fc2": "mlp.fc2"}


@dataclass(frozen=True)
class Config:
    layers: int = 50
    dim: int = 5376
    heads: int = 56
    head_dim: int = 128
    ffn: int = 14336
    eps: float = 1e-5
    qk_eps: float = 1e-5


@dataclass
class Block:
    lins: dict[str, L.Linear]
    norm1: torch.Tensor            # fp32 [D]
    norm2: torch.Tensor
    q_norm: torch.Tensor           # [head_dim], the checkpoint dtype (kitchen casts)
    k_norm: torch.Tensor
    ada_w: torch.Tensor            # fp32 [3 * 6 * D, t_dim]
    ada_b: torch.Tensor            # fp32 [3 * 6 * D]
    host: dict[str, torch.Tensor] = field(default_factory=dict)   # pinned copies of the big tensors (streaming)
    resident: bool = True


def _lin_tensors(lin: L.Linear) -> dict[str, tuple[object, str]]:
    """The device tensors a linear holds, as (holder, attribute), for streaming."""

    if isinstance(lin, L.Nvfp4Linear):
        return {"words": (lin.lin, "words"), "bs": (lin.lin, "bs")}
    if isinstance(lin, L.Fp8Linear):
        return {"w8": (lin, "w8")}
    if isinstance(lin, L.Int8Linear):
        return {"q": (lin, "q"), "s": (lin, "scale")}
    return {"w": (lin, "w")}


class Streamer:
    """Blocks not resident in VRAM: pinned host copies, two device slots, copies on their own stream."""

    def __init__(self, engine: "MiniMaxH3Engine"):
        self.e = engine
        self.stream = torch.cuda.Stream(device=engine.device)
        self.slots: list[dict[str, torch.Tensor] | None] = [None, None]
        self.slot_block = [-1, -1]
        self.loaded = [None, None]         # events: copy done
        self.freed = [None, None]          # events: compute done with the slot
        self.busy = [-1, -1]               # block computing from the slot (its contents must not change)
        self.order: list[int] = []

    def refs(self, i: int):
        b = self.e.blocks[i]
        for name in LIN_NAMES:
            for t, ref in _lin_tensors(b.lins[name]).items():
                yield f"{name}.{t}", ref

    def plan(self):
        self.order = [i for i, b in enumerate(self.e.blocks) if not b.resident]
        self.slots = [None, None]
        if self.order:
            # one tensor per key across the streamed blocks (a key always has one shape: same kind, same layer)
            union = {}
            for i in self.order:
                for k, v in self.e.blocks[i].host.items():
                    union.setdefault(k, v)
            self.slots = [{k: torch.empty_like(v, device=self.e.device) for k, v in union.items()} for _ in range(2)]
        self.slot_block, self.busy = [-1, -1], [-1, -1]
        self.freed = [None, None]

    def _slot_of(self, i: int) -> int:
        return self.order.index(i) % 2

    def prefetch(self, i: int):
        if i >= len(self.e.blocks) or self.e.blocks[i].resident:
            return
        s = self._slot_of(i)
        if self.slot_block[s] == i or self.busy[s] not in (-1, i):
            return                         # loaded already, or the slot is in use (acquire loads it later)
        with torch.cuda.stream(self.stream):
            if self.freed[s] is not None:
                self.stream.wait_event(self.freed[s])
            for k, t in self.e.blocks[i].host.items():
                self.slots[s][k].copy_(t, non_blocking=True)
            ev = torch.cuda.Event()
            ev.record(self.stream)
        self.loaded[s], self.slot_block[s] = ev, i

    def acquire(self, i: int):
        """Make block i's weights usable on the current stream; start copying the next streamed block."""

        b = self.e.blocks[i]
        if not b.resident:
            self.prefetch(i)
            s = self._slot_of(i)
            torch.cuda.current_stream().wait_event(self.loaded[s])
            self.busy[s] = i
            for k, (holder, attr) in self.refs(i):
                setattr(holder, attr, self.slots[s][k])
        nxt = next((j for j in self.order if j > i), None)
        if nxt is None and self.order:
            nxt = self.order[0]            # the next forward starts with the first streamed block
        if nxt is not None:
            self.prefetch(nxt)

    def release(self, i: int):
        if not self.e.blocks[i].resident:
            ev = torch.cuda.Event()
            ev.record(torch.cuda.current_stream())
            s = self._slot_of(i)
            self.freed[s], self.busy[s] = ev, -1


class MiniMaxH3Engine:
    """The DiT blocks with linears of ``kind`` ("nvfp4", "fp8", "bf16"); attention backend ``attn``."""

    MLP_CHUNK = 8192
    SLACK = 1 << 30                       # VRAM left to allocations outside PyTorch (cuBLAS, cuDNN, modules)

    def __init__(self, blocks: list[Block], kind: str, cfg: Config = Config(), device="cuda", attn: str = "auto"):
        self.blocks, self.kind, self.cfg, self.device = blocks, kind, cfg, torch.device(device)
        self.attn = A.get(attn)
        self.attn_name = attn
        self.streamer = Streamer(self)
        self._idx_key = None
        self._idx = None
        self.reserve_bytes = None          # VRAM kept for activations (None: from the sequence length)
        self._plan_seq = None
        self.delayed_scales = os.environ.get("TFVIDEO_DELAYED_SCALES", "1") != "0"
        self.source: str | None = None     # the checkpoint it was converted from (LoRA re-merges)
        self.lora_sig = ""                 # the merged LoRA set ("" = none)
        self.on_forward = None             # hook(engine, S, transformer_options) at block 0 (the ComfyUI node: LoRA
                                           # sync, and making room before a new memory plan)
        dump = os.environ.get("TFVIDEO_DUMP_QKV")
        self.dump_qkv = (dump.rsplit(":", 1)[0], int(dump.rsplit(":", 1)[1])) if dump else None

    # ------------------------------------------------------------------ memory
    def block_bytes(self) -> int:
        b = self.blocks[0]
        return sum(t.numel() * t.element_size() for t in (b.host.values() if b.host else []))

    def activation_bytes(self, S: int) -> int:
        D, cfg = self.cfg.dim, self.cfg
        inner = cfg.heads * cfg.head_dim
        chunk = min(S, self.MLP_CHUNK)
        attn = S * (3 * inner + inner) * 2 + S * inner * 2              # qkv + attention out + its [S, inner] copy
        mlp = chunk * (2 * cfg.ffn + cfg.ffn) * 2
        return 2 * S * D * 2 + max(attn, mlp) + (1536 << 20)               # x, h, the larger phase, workspace + slack

    def wanted_bytes(self, S: int) -> int:
        """VRAM for every block resident plus this sequence's activations."""

        return len(self.blocks) * self.block_bytes() + self.activation_bytes(S)

    def plan_memory(self, S: int):
        """Resident blocks: as many as fit beside this sequence's activations (the rest stream)."""

        if self._plan_seq == S:
            return
        per = self.block_bytes()
        if per == 0:                       # converted with no host copies: everything stays resident
            self._plan_seq = S
            return
        # A computed budget, not allocation probing: on Windows the driver's sysmem fallback lets allocations past
        # the end of VRAM succeed (in shared system memory), so "allocate until it fails" over-commits. The ComfyUI
        # node asks ComfyUI to unload what it can before this runs (comfy_nodes._on_forward).
        self.release_vram()
        # libraries that allocate outside PyTorch's cache (cuBLAS handles and workspaces, cuDNN, kernel modules)
        # initialize first; SLACK stays free at the driver level for them
        torch.ones(8, 8, device=self.device) @ torch.ones(8, 8, device=self.device)
        torch.cuda.synchronize(self.device)
        free, _ = torch.cuda.mem_get_info(self.device)
        free += torch.cuda.memory_reserved(self.device) - torch.cuda.memory_allocated(self.device)
        need = self.reserve_bytes if self.reserve_bytes is not None else self.activation_bytes(S)
        n = int(max(0, min(len(self.blocks), (free - need - 2 * per - self.SLACK) // per)))
        for i in range(n):
            for k, (holder, attr) in self.streamer.refs(i):
                setattr(holder, attr, self.blocks[i].host[k].to(self.device, non_blocking=True))
            self.blocks[i].resident = True
        self.streamer.plan()
        self._plan_seq = S
        print(f"[tfvideo] {n}/{len(self.blocks)} blocks resident, {len(self.blocks) - n} streamed "
              f"(sequence {S}, {per / 2**20:.0f} MiB a block; {free / 2**30:.1f} GiB usable, "
              f"{need / 2**30:.1f} GiB kept for activations)",
              flush=True)

    def replace_blocks(self, blocks: list[Block], lora_sig: str):
        """Swap in re-converted blocks (another LoRA set); the old ones are dropped."""

        self.release_vram()
        self.blocks, self.lora_sig = blocks, lora_sig
        self.streamer = Streamer(self)
        torch.cuda.empty_cache()

    def begin_forward(self, S: int, transformer_options: dict):
        if self.on_forward is not None:
            self.on_forward(self, S, transformer_options)
        self.plan_memory(S)
        lins = [lin for _, lin in self.linears()]
        if self.delayed_scales:
            clipped = L.update_delayed(lins)
            if clipped:
                print(f"[tfvideo] {clipped} linears clipped last forward (scales adapt)", flush=True)
        else:
            L.reset_delayed(lins)

    def end_run(self):
        """A sampling run is over: the next run starts with exact per-call scales again."""

        L.reset_delayed([lin for _, lin in self.linears()])

    def release_vram(self):
        """Drop every block's device copy and the streaming slots (host copies stay); the next forward re-plans."""

        torch.cuda.current_stream(self.device).synchronize()
        for i, b in enumerate(self.blocks):
            if b.host:
                for k, (holder, attr) in self.streamer.refs(i):
                    setattr(holder, attr, None)
                b.resident = False
        self.streamer.order, self.streamer.slots = [], [None, None]
        self.streamer.slot_block, self.streamer.busy = [-1, -1], [-1, -1]
        self._plan_seq = None
        torch.cuda.empty_cache()

    # ------------------------------------------------------------------ per forward
    def row_index(self, mod_segments, S: int) -> torch.Tensor:
        """int32 [S]: each row's modulation row (the segments are the same list object for all 50 blocks)."""

        key = (id(mod_segments), S)
        if key != self._idx_key:
            idx = torch.empty(S, dtype=torch.int32)
            for a, b, row in mod_segments:
                idx[a:b] = row.cpu().to(torch.int32) if torch.is_tensor(row) else int(row)
            self._idx = idx.to(self.device, non_blocking=True)
            self._idx_key = key
            self._idx_ref = mod_segments       # keep the list alive so its id is not reused
        return self._idx

    def modulation(self, b: Block, t_emb: torch.Tensor) -> torch.Tensor:
        """fp32 [M * 3, 6, D]: AdalnProj (no SiLU on the curve form) viewed as ComfyUI's mod rows."""

        m = torch.addmm(b.ada_b, t_emb.float(), b.ada_w.t())
        return m.view(-1, 6, self.cfg.dim).contiguous()

    # ------------------------------------------------------------------ the block
    def _linear(self, lin: L.Linear, x: torch.Tensor, out: torch.Tensor | None = None) -> torch.Tensor:
        return lin(x, out)

    def block(self, i: int, x: torch.Tensor, t_emb, mod_segments, rope_freqs, attention=None,
              transformer_options=None) -> torch.Tensor:
        cfg, b = self.cfg, self.blocks[i]
        S = x.shape[0]
        H, Dh = cfg.heads, cfg.head_dim
        self.streamer.acquire(i)
        idx = self.row_index(mod_segments, S)
        mod = self.modulation(b, t_emb)

        h = K.norm_mod(x, b.norm1, mod, 0, idx, cfg.eps)
        if attention is not None:
            # ComfyUI's attention replacement (e.g. the Model Sparse Attention node): it projects through this
            # block's facade (EngineBlock.attn) and returns out_proj(attention)
            a = attention(h, rope_freqs=rope_freqs, transformer_options=transformer_options or {})
            K.gate_add(x, a, mod, 0, idx)
            del a
            return self._mlp_half(i, b, x, h, mod, idx)
        qkv = self._linear(b.lins["qkv"], h)
        q = qkv[:, : H * Dh].view(1, S, H, Dh)
        k = qkv[:, H * Dh: 2 * H * Dh].view(1, S, H, Dh)
        v = qkv[:, 2 * H * Dh:].view(S, H, Dh)
        import comfy_kitchen as ck
        ck.rms_rope_split_half_(q, k, rope_freqs, b.q_norm, b.k_norm, epsilon=cfg.qk_eps,
                                rot_dim=rope_freqs.shape[-3] * 2)
        qh, kh, vh = q[0].transpose(0, 1).unsqueeze(0), k[0].transpose(0, 1).unsqueeze(0), v.transpose(0, 1).unsqueeze(0)
        if self.dump_qkv and i == self.dump_qkv[1]:
            # TFVIDEO_DUMP_QKV=path.pt:block -> this block's attention inputs, once (tools/attn_bench.py --qkv)
            torch.save({"q": qh.contiguous().cpu(), "k": kh.contiguous().cpu(), "v": vh.contiguous().cpu()},
                       self.dump_qkv[0])
            print(f"[tfvideo] dumped block {i} q/k/v {tuple(qh.shape)} to {self.dump_qkv[0]}")
            self.dump_qkv = None
        o = self.attn(qh, kh, vh)
        del qkv, q, k, v
        o = o[0].transpose(0, 1).reshape(S, H * Dh)
        self._linear(b.lins["out"], o, out=h)
        del o
        K.gate_add(x, h, mod, 0, idx)
        return self._mlp_half(i, b, x, h, mod, idx)

    def _mlp_half(self, i, b, x, h, mod, idx):
        S, cfg = x.shape[0], self.cfg
        K.norm_mod(x, b.norm2, mod, 1, idx, cfg.eps, out=h)
        for r0 in range(0, S, self.MLP_CHUNK):
            r1 = min(S, r0 + self.MLP_CHUNK)
            a = K.swiglu(self._linear(b.lins["fc1"], h[r0:r1]))
            self._linear(b.lins["fc2"], a, out=h[r0:r1])
            del a
        K.gate_add(x, h, mod, 1, idx)
        self.streamer.release(i)
        return x

    # ------------------------------------------------------------------ info
    def linears(self):
        for i, b in enumerate(self.blocks):
            for name in LIN_NAMES:
                yield f"{i}.{name}", b.lins[name]

    def nbytes(self) -> int:
        return len(self.blocks) * self.block_bytes()


class _Proj:
    """A block linear as ComfyUI modules call it (x -> x @ W^T), weights acquired by the running block."""

    def __init__(self, engine, i, name):
        self.engine, self.i, self.name = engine, i, name

    def __call__(self, x):
        return self.engine.blocks[self.i].lins[self.name](x.contiguous())


class _Norm:
    def __init__(self, engine, i, name, eps):
        self.engine, self.i, self.name, self.eps = engine, i, name, eps

    @property
    def weight(self):
        return getattr(self.engine.blocks[self.i], self.name)


class AttnFacade:
    """What ComfyUI's H3 attention patches read from ``block.attn`` (heads, head_dim, qkv_proj, q/k norms, out_proj),
    backed by the engine, so patches such as the stock Model Sparse Attention node run on engine blocks."""

    def __init__(self, engine, i):
        cfg = engine.cfg
        self.heads, self.head_dim = cfg.heads, cfg.head_dim
        self.qkv_proj, self.out_proj = _Proj(engine, i, "qkv"), _Proj(engine, i, "out")
        self.q_norm, self.k_norm = _Norm(engine, i, "q_norm", cfg.qk_eps), _Norm(engine, i, "k_norm", cfg.qk_eps)
        self.to_gate_compress = None


class EngineBlock(torch.nn.Module):
    """ComfyUI's ``DiTBlock`` call signature on the engine; no parameters of its own (the engine owns them)."""

    def __init__(self, engine: MiniMaxH3Engine, i: int):
        super().__init__()
        self.engine, self.i = engine, i
        self.attn = AttnFacade(engine, i)

    def forward(self, x, t_emb, mod_segments, rope_freqs, transformer_options={}, attention=None):
        # ComfyUI records and replays a block's allocations (its malloc graph, under dynamic VRAM); the engine
        # allocates by sequence length and syncs on its own, so it runs outside that recording
        with _no_malloc_graph():
            if self.i == 0:
                self.engine.begin_forward(x.shape[0], transformer_options)
            return self.engine.block(self.i, x, t_emb, mod_segments, rope_freqs, attention, transformer_options)


def _no_malloc_graph():
    try:
        import comfy.model_prefetch
        return comfy.model_prefetch.pause_malloc_graph()
    except ImportError:
        import contextlib
        return contextlib.nullcontext()
