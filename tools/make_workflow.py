r"""Flat ComfyUI workflows from ComfyUI's MiniMax H3 templates, with the TensorFold loader in place of the stock one.

The bundled templates wrap the pipeline in one subgraph node ("Image to Video (MiniMax H3)"). This inlines the
subgraph (its nodes, links, groups and the values set on the wrapper), rewires the template's outer nodes (resolution
selector, image loaders, Save Video) straight to the inner ones, and swaps UNETLoader for TFMiniMaxH3Loader. Widget
orders come from a running ComfyUI's /object_info (with this node installed).

Usage: python_embeded\python.exe -B tools\make_workflow.py [--server http://127.0.0.1:8188]
"""
import argparse
import copy
import json
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TEMPLATES = {"video_minimax_h3_t2v.json": "Text to Video (MiniMax H3, TensorFold).json",
             "video_minimax_h3_i2v.json": "Image to Video (MiniMax H3, TensorFold).json"}
WIDGET_TYPES = {"INT", "FLOAT", "STRING", "BOOLEAN", "COMBO"}


def widget_names(info: dict, node_type: str) -> list[str]:
    spec = info[node_type]["input"]
    order = info[node_type].get("input_order", {})
    names = []
    for group in ("required", "optional"):
        for name in order.get(group, list(spec.get(group, {}))):
            t, opts = (spec[group][name] + [{}])[:2]
            if isinstance(t, list) or t in WIDGET_TYPES:
                names.append(name)
                if isinstance(opts, dict) and opts.get("control_after_generate"):
                    names.append("control_after_generate")
    return names


def set_widget(node: dict, info: dict, name: str, value):
    names = widget_names(info, node["type"])
    if name not in names:
        raise KeyError(f"{node['type']} has no widget {name!r} ({names})")
    vals = node.setdefault("widgets_values", [])
    i = names.index(name)
    while len(vals) <= i:
        vals.append(None)
    vals[i] = value


def flatten(w: dict, info: dict) -> dict:
    sg = w["definitions"]["subgraphs"][0]
    inst = next(n for n in w["nodes"] if n["type"] == sg["id"])
    out = copy.deepcopy(w)
    out["nodes"] = [n for n in out["nodes"] if n["id"] != inst["id"]]
    del out["definitions"]
    nid = max(n["id"] for n in w["nodes"]) + 1000
    lid = max([l[0] for l in w["links"]] + [l["id"] for l in sg["links"]]) + 1000
    nmap = {n["id"]: nid + i for i, n in enumerate(sg["nodes"])}
    minx = min(n["pos"][0] for n in sg["nodes"])
    miny = min(n["pos"][1] for n in sg["nodes"])
    shift = (inst["pos"][0] - minx, inst["pos"][1] - miny)
    nodes = {}
    for n in copy.deepcopy(sg["nodes"]):
        n["id"] = nmap[n["id"]]
        n["pos"] = [n["pos"][0] + shift[0], n["pos"][1] + shift[1]]
        nodes[n["id"]] = n
    links, lmap = [], {}
    for i, l in enumerate(sg["links"]):
        if l["origin_id"] == -10 or l["target_id"] == -20:
            continue
        lmap[l["id"]] = lid + i
        links.append([lid + i, nmap[l["origin_id"]], l["origin_slot"], nmap[l["target_id"]], l["target_slot"], l["type"]])
    for n in nodes.values():
        for p in n.get("inputs", []):
            if p.get("link") is not None:
                p["link"] = lmap.get(p["link"], p["link"])
        for o in n.get("outputs", []):
            o["links"] = [lmap.get(x, x) for x in (o.get("links") or [])]
    by_id = {l["id"]: l for l in sg["links"]}
    outer = {l[0]: l for l in out["links"]}
    inst_slot = {p["name"]: k for k, p in enumerate(inst.get("inputs", []))}
    values = iter(inst.get("widgets_values", []))
    for k, sgin in enumerate(sg["inputs"]):
        value = next(values) if sgin["type"] != "IMAGE" else None
        slot = inst_slot.get(sgin["name"])
        src = next((l for l in out["links"] if l[3] == inst["id"] and l[4] == slot), None) if slot is not None else None
        for inner_link in sgin.get("linkIds", []):
            l = by_id[inner_link]
            tgt = nodes[nmap[l["target_id"]]]
            tin = tgt["inputs"][l["target_slot"]]
            if src is not None:                         # an outer node feeds it: wire it straight in
                src[3], src[4] = tgt["id"], l["target_slot"]
                tin["link"] = src[0]
            else:
                tin["link"] = None
                if "widget" in tin and value is not None:
                    set_widget(tgt, info, tin["widget"]["name"], value)
    for sgout in sg["outputs"]:
        for inner_link in sgout.get("linkIds", []):
            l = by_id[inner_link]
            org = nodes[nmap[l["origin_id"]]]
            outs = org["outputs"][l["origin_slot"]]
            outs["links"] = [x for x in outs.get("links") or [] if x != inner_link]
            for ol in out["links"]:
                if ol[1] == inst["id"]:
                    ol[1], ol[2] = org["id"], l["origin_slot"]
                    outs["links"].append(ol[0])
    out["nodes"] += list(nodes.values())
    out["links"] += links
    out["groups"] = out.get("groups", []) + [
        {**g, "bounding": [g["bounding"][0] + shift[0], g["bounding"][1] + shift[1], *g["bounding"][2:]]}
        for g in sg.get("groups", [])]
    out["last_node_id"] = max(n["id"] for n in out["nodes"])
    out["last_link_id"] = max(l[0] for l in out["links"])
    for n in out["nodes"]:                          # the swap
        if n["type"] == "UNETLoader":
            n["type"] = "TFMiniMaxH3Loader"
            n["title"] = "TensorFold MiniMax H3 Loader"
            n.setdefault("properties", {})["Node name for S&R"] = "TFMiniMaxH3Loader"
            for p in n.get("inputs", []):
                if p["name"] == "weight_dtype":
                    p["name"] = "precision"
                    if "widget" in p:
                        p["widget"]["name"] = "precision"
            n["widgets_values"] = [n["widgets_values"][0], "nvfp4", "auto"]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--server", default="http://127.0.0.1:8188")
    a = ap.parse_args()
    info = json.loads(urllib.request.urlopen(a.server + "/object_info").read())
    if "TFMiniMaxH3Loader" not in info:
        raise SystemExit("the running ComfyUI does not have the TensorFold node installed")
    import comfyui_workflow_templates_json  # noqa: F401  (ComfyUI's template package)
    tdir = Path(comfyui_workflow_templates_json.__file__).parent / "templates"
    for src, dst in TEMPLATES.items():
        w = flatten(json.load(open(tdir / src, encoding="utf-8")), info)
        path = ROOT / "workflows" / dst
        path.write_text(json.dumps(w, indent=1, ensure_ascii=False), encoding="utf-8")
        print("wrote", path, len(w["nodes"]), "nodes", len(w["links"]), "links")


if __name__ == "__main__":
    main()
