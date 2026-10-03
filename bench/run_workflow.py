r"""Run a saved ComfyUI workflow file (UI format) on a running ComfyUI: converted to an API prompt the way the
frontend does (links, widget values by the node's widget order, dynamic sub-widgets), queued, timed. Proves a
workflow in workflows/ executes as saved.

Usage: python bench\run_workflow.py "workflows\Text to Video (MiniMax H3, TensorFold).json" [--server ...] [--image x.png]
"""
import argparse
import json
import sys
import time
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
from make_workflow import widget_names  # noqa: E402

SKIP = {"MarkdownNote", "Note", "PrimitiveNode", "Reroute"}
# dynamic-combo nodes: widget values -> API keys, in saved order
DYNAMIC = {"BlockSparseAttention": ["selection", "selection.tau", "start_percent", "end_percent", "dense_blocks",
                                    "min_tokens", "extra_tokens", "sink_conditioning", "verbose"],
           "SaveVideo": ["filename_prefix", "format", "format.codec"]}


def to_api(w: dict, info: dict) -> dict:
    links = {l[0]: l for l in w["links"]}
    api = {}
    for n in w["nodes"]:
        if n["type"] in SKIP or n.get("mode", 0) in (2, 4):
            continue
        inputs = {}
        linked = set()
        for p in n.get("inputs", []):
            if p.get("link") is not None:
                l = links[p["link"]]
                inputs[p["name"]] = [str(l[1]), l[2]]
                linked.add(p.get("widget", {}).get("name", p["name"]))
        names = DYNAMIC.get(n["type"]) or widget_names(info, n["type"])
        for name, value in zip(names, n.get("widgets_values") or []):
            if name == "control_after_generate" or name in linked or name == "upload":
                continue
            inputs[name] = value
        api[str(n["id"])] = {"class_type": n["type"], "inputs": inputs}
    return api


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("workflow")
    ap.add_argument("--server", default="http://127.0.0.1:8188")
    ap.add_argument("--image", default="", help="image file name (in ComfyUI's input folder) for LoadImage nodes")
    a = ap.parse_args()
    info = json.loads(urllib.request.urlopen(a.server + "/object_info").read())
    api = to_api(json.load(open(a.workflow, encoding="utf-8")), info)
    if a.image:
        for v in api.values():
            if v["class_type"] == "LoadImage":
                v["inputs"]["image"] = a.image
    req = urllib.request.Request(a.server + "/prompt", data=json.dumps({"prompt": api}).encode(),
                                 headers={"Content-Type": "application/json"})
    try:
        pid = json.loads(urllib.request.urlopen(req).read())["prompt_id"]
    except urllib.error.HTTPError as e:
        raise SystemExit("rejected: " + e.read().decode()[:3000])
    t0 = time.perf_counter()
    while True:
        time.sleep(1)
        h = json.loads(urllib.request.urlopen(f"{a.server}/history/{pid}").read())
        if pid in h:
            st = h[pid]["status"]
            files = [f["filename"] for o in h[pid].get("outputs", {}).values()
                     for k in ("images", "videos", "gifs", "video") for f in o.get(k, []) if isinstance(f, dict)]
            print(json.dumps({"status": st.get("status_str"), "seconds": round(time.perf_counter() - t0, 1),
                              "outputs": files}))
            if st.get("status_str") != "success":
                print(json.dumps(st)[:3000])
            return


if __name__ == "__main__":
    main()
