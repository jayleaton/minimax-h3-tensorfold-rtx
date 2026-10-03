"""Drive a ComfyUI portable install headless on a MiniMax H3 video workflow (the "Image to Video (MiniMax H3)"
blueprint: res_multistep, simple schedule, BasicGuider, 24 fps), all directories redirected into runs/<tag>.

Per run: wall seconds, sampler s/it, VRAM peak (device-wide). Run 0 includes loading and text encoding; later runs
change only the seed, so they time sampling + VAE decode.
Usage: python bench/comfy_h3.py --tag base_768 --width 768 --height 448 --seconds 2 --steps 20 [--tf nvfp4]
"""
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import threading
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))
import paths  # noqa: E402

PORT = 8199
PROMPT = ("Cinematic handheld shot in a rainy neon-lit night market. A street cook in a white apron flips noodles in a "
          "flaming wok, steam and sparks rising, customers laughing in the background. Rain drips from red paper "
          "lanterns. Sound: sizzling wok, crackling fire, rain on tarp, distant chatter. The cook says: \"Two more, "
          "extra spicy!\"")


def frames_for(seconds: float) -> int:
    # the blueprint's duration -> frame count expression (snaps to the 17k + 5 grid)
    n = max(5, round(seconds * 24))
    return n + (5 - n % 17) % 17


def graph(a, prompt: str, seed: int, prefix: str) -> dict:
    if a.tf:
        unet = {"class_type": "TFMiniMaxH3Loader", "inputs": {"unet_name": paths.DIT_NAME, "precision": a.tf,
                                                               "attention": a.attn}}
    else:
        unet = {"class_type": "UNETLoader", "inputs": {"unet_name": paths.DIT_NAME, "weight_dtype": "default"}}
    g = {
        "1": unet,
        "2": {"class_type": "CLIPLoader", "inputs": {"clip_name": "qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors",
                                                     "type": "minimax", "device": "default"}},
        "3": {"class_type": "VAELoader", "inputs": {"vae_name": a.vae}},
        "4": {"class_type": "VAELoader", "inputs": {"vae_name": "minimax_h3_audio_vae_fp32.safetensors"}},
        "5": {"class_type": "MiniMaxH3ImageToVideo", "inputs": {"clip": ["2", 0], "vae": ["3", 0], "prompt": prompt,
                                                                "width": a.width, "height": a.height,
                                                                "length": frames_for(a.seconds)}},
        "6": {"class_type": "RandomNoise", "inputs": {"noise_seed": seed}},
        "7": {"class_type": "KSamplerSelect", "inputs": {"sampler_name": a.sampler}},
        "8": {"class_type": "BasicScheduler", "inputs": {"model": ["1", 0], "scheduler": "simple", "steps": a.steps,
                                                         "denoise": 1.0}},
        "9": {"class_type": "BasicGuider", "inputs": {"model": ["1", 0], "conditioning": ["5", 0]}},
        "10": {"class_type": "SamplerCustomAdvanced", "inputs": {"noise": ["6", 0], "guider": ["9", 0], "sampler": ["7", 0],
                                                                 "sigmas": ["8", 0], "latent_image": ["5", 1]}},
        "11": {"class_type": "VAEDecode", "inputs": {"samples": ["10", 0], "vae": ["3", 0]}},
        "12": {"class_type": "VAEDecodeAudio", "inputs": {"samples": ["10", 0], "vae": ["4", 0]}},
        "13": {"class_type": "CreateVideo", "inputs": {"images": ["11", 0], "audio": ["12", 0], "fps": 24.0}},
        "14": {"class_type": "SaveVideo", "inputs": {"video": ["13", 0], "filename_prefix": prefix, "format": "mp4",
                                                     "format.codec": "h264", "format.codec.encoding": "re-encode",
                                                     "format.codec.encoding.crf": 12.0}},
    }
    if a.first_frame:
        g["20"] = {"class_type": "LoadImage", "inputs": {"image": Path(a.first_frame).name}}
        g["5"]["inputs"]["first_frame"] = ["20", 0]
    model = ["1", 0]
    if a.lora:
        g["21"] = {"class_type": "LoraLoaderModelOnly", "inputs": {"model": model, "lora_name": a.lora,
                                                                   "strength_model": a.lora_strength}}
        model = ["21", 0]
    if a.sparse:
        # ComfyUI's Model Sparse Attention node, sol-attn at its defaults apart from tau
        g["22"] = {"class_type": "BlockSparseAttention", "inputs": {"model": model, "selection": "sol-attn",
                                                                    "selection.tau": a.sparse, "start_percent": 0.2,
                                                                    "end_percent": 1.0, "dense_blocks": "",
                                                                    "min_tokens": 12288, "extra_tokens": 256,
                                                                    "sink_conditioning": "exact_kv_and_rows",
                                                                    "verbose": True}}
        model = ["22", 0]
    g["8"]["inputs"]["model"] = g["9"]["inputs"]["model"] = model
    return g


class VramSampler(threading.Thread):
    def __init__(self):
        super().__init__(daemon=True)
        self.peak, self.stop = 0, False

    def run(self):
        p = subprocess.Popen(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits", "-lms", "200"],
                             stdout=subprocess.PIPE, text=True)
        for line in p.stdout:
            if self.stop:
                break
            try:
                self.peak = max(self.peak, int(line.strip()))
            except ValueError:
                pass
        p.kill()


def vram_now() -> int:
    return int(subprocess.check_output(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                                       text=True).strip())


def call(path, data=None):
    req = urllib.request.Request(f"http://127.0.0.1:{PORT}{path}", data=json.dumps(data).encode() if data else None,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=60) as r:
        return json.loads(r.read())


def run_once(g) -> float:
    t0 = time.perf_counter()
    pid = call("/prompt", {"prompt": g})["prompt_id"]
    while True:
        time.sleep(0.25)
        h = call(f"/history/{pid}")
        if pid in h:
            st = h[pid]["status"]
            if st.get("status_str") != "success":
                raise RuntimeError(json.dumps(st)[:3000])
            return time.perf_counter() - t0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--width", type=int, default=1344)
    ap.add_argument("--height", type=int, default=768)
    ap.add_argument("--seconds", type=float, default=5.0)
    ap.add_argument("--steps", type=int, default=20)
    ap.add_argument("--sampler", default="res_multistep")
    ap.add_argument("--seed", type=int, default=757358688076805)
    ap.add_argument("--prompt", default=PROMPT)
    ap.add_argument("--first-frame", default="", help="image file: fl2va with this first frame (else t2va)")
    ap.add_argument("--lora", default="", help="LoRA file name (e.g. the 8-step turbo LoRA)")
    ap.add_argument("--lora-strength", type=float, default=1.0)
    ap.add_argument("--sparse", type=float, default=0.0, help="add ComfyUI's sol-attn sparse attention at this tau")
    ap.add_argument("--repeat", type=int, default=2)
    ap.add_argument("--same-seed", action="store_true", help="every timing run uses --seed (determinism checks)")
    ap.add_argument("--jobs", default="", help="JSON file: [[prompt, seed], ...] rendered once each after the timing runs")
    ap.add_argument("--tf", default="", help="run the DiT on the TensorFold node at this precision")
    ap.add_argument("--attn", default="auto", help="TensorFold node attention backend")
    ap.add_argument("--vae", default="minimax_h3_video_vae_fp16.safetensors")
    ap.add_argument("--extra", nargs="*", default=[])
    a = ap.parse_args()

    portable = paths.portable()
    run_dir = ROOT / "runs" / a.tag
    run_dir.mkdir(parents=True, exist_ok=True)
    for d in ("user", "temp", "input"):
        (run_dir / d).mkdir(exist_ok=True)
    if a.first_frame:
        shutil.copy(a.first_frame, run_dir / "input" / Path(a.first_frame).name)
    idle = vram_now()
    log = open(run_dir / "comfy.log", "w", encoding="utf-8")
    # loads this repo's node and repo-local LoRAs without touching the ComfyUI folders
    extra = run_dir / "extra_paths.yaml"
    extra.write_text(f"tfvideo:\n  base_path: {ROOT.as_posix()}\n  custom_nodes: comfyui\n  loras: models/loras\n  vae: models/vae\n",
                     encoding="utf-8")
    cmd = [str(paths.python()), "-s", str(portable / "ComfyUI" / "main.py"), "--windows-standalone-build",
           "--port", str(PORT), "--disable-all-custom-nodes", "--whitelist-custom-nodes", "ComfyUI-TensorFold-Video",
           "--extra-model-paths-config", str(extra), "--disable-auto-launch", "--database-url", "sqlite:///:memory:",
           "--user-directory", str(run_dir / "user"), "--output-directory", str(run_dir / "out"),
           "--temp-directory", str(run_dir / "temp"), "--input-directory", str(run_dir / "input"), *a.extra]
    env = dict(os.environ, PYTHONIOENCODING="utf-8", PYTHONDONTWRITEBYTECODE="1")
    t_launch = time.perf_counter()
    proc = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, env=env, cwd=str(portable))
    times, job_times = [], []
    try:
        while True:
            try:
                call("/system_stats")
                break
            except Exception:
                if proc.poll() is not None:
                    raise SystemExit("ComfyUI exited; see " + str(run_dir / "comfy.log"))
                time.sleep(0.5)
        t_server = time.perf_counter() - t_launch
        vs = VramSampler()
        vs.start()
        for i in range(a.repeat):
            times.append(run_once(graph(a, a.prompt, a.seed + (0 if a.same_seed else i), f"{a.tag}_r{i}")))
            print(f"run {i}: {times[-1]:.2f} s", flush=True)
        if a.jobs:
            for j, (prompt, seed) in enumerate(json.load(open(a.jobs, encoding="utf-8"))):
                job_times.append(run_once(graph(a, prompt, seed, f"job{j:02d}")))
                print(f"job {j}: {job_times[-1]:.2f} s", flush=True)
        vs.stop = True
        time.sleep(0.5)
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=30)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait(timeout=30)
        log.close()
    text = (run_dir / "comfy.log").read_text(encoding="utf-8", errors="replace")
    engine_ran = "[tfvideo]" in text
    if a.tf and not engine_ran:
        raise SystemExit(f"--tf {a.tf}: the log shows no tfvideo engine; results are not the engine's")
    its = re.findall(r"(\d+)/(\d+) \[([0-9:]+)<[^,\]]*,\s+([0-9.]+)(s/it|it/s)\]", text)
    rate = [(float(r[3]) if r[4] == "s/it" else 1 / float(r[3])) for r in its if r[0] == r[1]]
    res = {"tag": a.tag, "w": a.width, "h": a.height, "frames": frames_for(a.seconds), "steps": a.steps,
           "sampler": a.sampler, "lora": a.lora, "first_frame": bool(a.first_frame), "tf": a.tf, "sparse_tau": a.sparse, "vae": a.vae,
           "attn": a.attn if a.tf else " ".join(a.extra),
           "server_start_s": round(t_server, 2), "runs_s": [round(t, 2) for t in times], "sampler_s_per_it": rate,
           "job_s": [round(t, 2) for t in job_times], "engine_ran": engine_ran, "vram_idle_mib": idle,
           "vram_peak_mib": vs.peak, "executed": re.findall(r"Prompt executed in ([0-9.:]+)", text)}
    (run_dir / "result.json").write_text(json.dumps(res, indent=2))
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
