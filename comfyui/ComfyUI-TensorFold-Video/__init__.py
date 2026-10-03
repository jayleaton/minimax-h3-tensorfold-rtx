"""ComfyUI custom node: TensorFold video engine (tfvideo). Points at the tfvideo checkout (TFVIDEO_REPO or ../..)."""

import os
import sys
from pathlib import Path

_repo = Path(os.environ.get("TFVIDEO_REPO") or Path(__file__).resolve().parents[2])
for _p in (_repo, _repo / "vendor" / "TensorFold" / "src"):
    if _p.is_dir() and str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from tfvideo.comfy_nodes import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS  # noqa: E402

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]

if os.environ.get("TFVIDEO_TIME_NODES"):
    # benchmarking aid: log each node's wall time (GPU-synchronized) as "[tfvideo] node <class> <seconds>"
    import logging
    import time

    import execution
    import torch

    _get_output_data = execution.get_output_data

    async def _timed_get_output_data(prompt_id, unique_id, obj, *args, **kwargs):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        result = await _get_output_data(prompt_id, unique_id, obj, *args, **kwargs)
        torch.cuda.synchronize()
        logging.info(f"[tfvideo] node {type(obj).__name__} {time.perf_counter() - t0:.2f} s")
        return result

    execution.get_output_data = _timed_get_output_data
