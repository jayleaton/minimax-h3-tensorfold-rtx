"""Quality gate for videos: each candidate run's mp4s vs the baseline run's, same prompts and seeds (matched by order).

Video: 8 frames sampled evenly, per frame LPIPS (alex, lower is closer), DINOv2-S CLS cosine (higher is closer), PSNR;
reported as the mean over frames. Audio: log-mel spectrogram L1 distance (dB) and cosine. Writes
runs/<cand>/quality.json and a contact sheet runs/quality_<cands>.jpg (baseline first row of each clip).
Usage: tools\\env.cmd python bench\\quality.py base_tag cand_tag [cand_tag ...]
"""
import argparse
import glob
import json
import subprocess
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
FRAMES = 8


def clips(tag):
    return sorted(glob.glob(str(ROOT / "runs" / tag / "out" / "**" / "*.mp4"), recursive=True))


def frames(path, n=FRAMES):
    probe = json.loads(subprocess.check_output(["ffprobe", "-v", "error", "-select_streams", "v:0", "-count_frames",
                                                "-show_entries", "stream=nb_read_frames,width,height", "-of", "json",
                                                path]))["streams"][0]
    total, w, h = int(probe["nb_read_frames"]), int(probe["width"]), int(probe["height"])
    raw = subprocess.check_output(["ffmpeg", "-v", "error", "-i", path, "-f", "rawvideo", "-pix_fmt", "rgb24", "-"])
    video = np.frombuffer(raw, np.uint8).reshape(total, h, w, 3)
    idx = np.linspace(0, total - 1, n).round().astype(int)
    return [video[i] for i in idx]


def audio(path, sr=32000):
    raw = subprocess.check_output(["ffmpeg", "-v", "error", "-i", path, "-vn", "-ac", "1", "-ar", str(sr), "-f", "f32le",
                                   "-"])
    return torch.from_numpy(np.frombuffer(raw, np.float32).copy())


def logmel(wav, sr=32000):
    import torchaudio
    spec = torchaudio.transforms.MelSpectrogram(sample_rate=sr, n_fft=1024, hop_length=320, n_mels=80)(wav)
    return 10 * torch.log10(spec.clamp_min(1e-10))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("base")
    ap.add_argument("cands", nargs="+")
    ap.add_argument("--thumb", type=int, default=224)
    a = ap.parse_args()

    import lpips
    lp = lpips.LPIPS(net="alex", verbose=False).cuda()
    dino = torch.hub.load("facebookresearch/dinov2", "dinov2_vits14", verbose=False).cuda().eval()
    mean = torch.tensor([0.485, 0.456, 0.406], device="cuda").view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device="cuda").view(1, 3, 1, 1)

    def tens(img, size=None):
        im = Image.fromarray(img)
        if size:
            im = im.resize((size, size), Image.BICUBIC)
        return torch.from_numpy(np.asarray(im)).permute(2, 0, 1).float().div(255).unsqueeze(0).cuda()

    base = clips(a.base)
    base_frames = [frames(p) for p in base]
    summary, sheets = {}, []
    for cand in a.cands:
        res = []
        for ci, (bp, cp) in enumerate(zip(base, clips(cand))):
            fr = frames(cp)
            per = []
            for x, y in zip(base_frames[ci], fr):
                with torch.no_grad():
                    tx, ty = tens(x), tens(y)
                    lpv = lp(tx * 2 - 1, ty * 2 - 1).item()
                    d = torch.nn.functional.cosine_similarity(dino((tens(x, 448) - mean) / std),
                                                              dino((tens(y, 448) - mean) / std)).item()
                    mse = ((tx - ty) ** 2).mean().item()
                per.append((lpv, d, 10 * np.log10(1 / max(mse, 1e-10))))
            ab, ac = audio(bp), audio(cp)
            n = min(len(ab), len(ac))
            mb, mc = logmel(ab[:n]), logmel(ac[:n])
            r = {"clip": Path(cp).name, "lpips": float(np.mean([p[0] for p in per])),
                 "dino": float(np.mean([p[1] for p in per])), "psnr": float(np.mean([p[2] for p in per])),
                 "lpips_worst_frame": float(max(p[0] for p in per)),
                 "audio_logmel_l1_db": float((mb - mc).abs().mean()),
                 "audio_logmel_cos": float(torch.nn.functional.cosine_similarity(mb.flatten(), mc.flatten(), dim=0))}
            res.append(r)
            sheets.append((cand, ci, fr))
        agg = {k: round(float(np.mean([r[k] for r in res])), 4) for k in res[0] if k != "clip"}
        summary[cand] = agg
        (ROOT / "runs" / cand / "quality.json").write_text(json.dumps({"vs": a.base, "mean": agg, "clips": res}, indent=1))
        print(cand, json.dumps(agg), flush=True)
        for r in res:
            print("   ", json.dumps(r))

    # contact sheet: per clip, the baseline row then each candidate's row
    T = a.thumb
    h = int(T * base_frames[0][0].shape[0] / base_frames[0][0].shape[1])
    rows = []
    for ci in range(len(base)):
        rows.append((a.base, base_frames[ci]))
        rows += [(c, f) for c, i, f in sheets if i == ci]
    sheet = Image.new("RGB", (T * FRAMES + 120, (h + 4) * len(rows)), "white")
    dr = ImageDraw.Draw(sheet)
    for r, (tag, fr) in enumerate(rows):
        dr.text((4, r * (h + 4) + h // 2), tag[:18], fill="black")
        for j, f in enumerate(fr):
            sheet.paste(Image.fromarray(f).resize((T, h)), (120 + j * T, r * (h + 4)))
    out = ROOT / "runs" / f"quality_{'_'.join(a.cands)}.jpg"
    sheet.save(out, quality=88)
    print("sheet", out)


if __name__ == "__main__":
    main()
