"""Score photos with an exported model, outside LightCraft: a quick check of what the slider will see.

    python score.py --model D:/blurcull-data/blurnet.safetensors D:/photos/2026-10-trip --csv scores.csv --maps maps/

Prints each photo's subject score (0 = tack sharp, 100 = unrecognisable), the background blur and how the subject
was found (auto here; inside LightCraft the camera's focus point and face regions are used too). --maps writes a
picture per photo with the blur map over it (green = sharp, red = blurry, no tint = nothing to judge).
For rated/ folders, add --check to report how many land inside their folder's range.
"""
from __future__ import annotations

import argparse
import csv
import re
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from safetensors import safe_open

from kernels import S_MAX
from net import ARCH_NAME, FoldedBlurNet
from prepare import DIRECT, RAWISH, RANGE_RE, find_cli, read_image
from subject import subject_score
from synth import luma


def load(path: str):
    tensors, meta = {}, {}
    with safe_open(path, framework="pt") as f:
        meta = f.metadata() or {}
        for k in f.keys():
            tensors[k] = f.get_tensor(k)
    if meta.get("format") != "lightcraft-blurnet" or meta.get("arch") != ARCH_NAME:
        sys.exit(f"{path}: not a {ARCH_NAME} file ({meta.get('format')}, {meta.get('arch')})")
    return FoldedBlurNet(tensors).eval(), meta


def overlay(img, blur, conf):
    h, w = img.shape[:2]
    s = np.clip(blur / S_MAX * 100 / 60, 0, 1)  # full red at score 60
    col = np.stack([np.zeros_like(s), 1 - s, s], -1).astype(np.float32)  # BGR: green -> red
    a = (conf >= 0.5).astype(np.float32)[..., None] * 0.45
    col = cv2.resize(col, (w, h), interpolation=cv2.INTER_NEAREST)
    a = cv2.resize(a, (w, h), interpolation=cv2.INTER_NEAREST)[..., None]
    return (img.astype(np.float32) * (1 - a) + col * 255 * a).astype(np.uint8)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("paths", nargs="+", help="photos or folders")
    ap.add_argument("--model", required=True)
    ap.add_argument("--csv")
    ap.add_argument("--maps", help="folder for blur-map pictures")
    ap.add_argument("--check", action="store_true", help="folders named *_lo-hi: report hits inside the range")
    ap.add_argument("--cli")
    a = ap.parse_args()

    model, meta = load(a.model)
    analysis = int(meta.get("analysis_long_edge", 1600))
    cli = find_cli(a.cli)
    files = []
    for p in map(Path, a.paths):
        if p.is_dir():
            files += [f for f in sorted(p.rglob("*")) if f.suffix.lower() in DIRECT | RAWISH and "cache" not in f.parts]
        else:
            files.append(p)
    if a.maps:
        Path(a.maps).mkdir(parents=True, exist_ok=True)
    rows, hits, total = [], 0, 0
    for f in files:
        img, err = read_image(f, cli, analysis)
        if img is None:
            print(f"{f}: {err}")
            continue
        y = torch.from_numpy(luma(img.astype(np.float32) / 255))[None, None]
        with torch.no_grad():
            out = model(y)[0].numpy()
        blur, conf = out[0], 1 / (1 + np.exp(-out[1]))
        r = subject_score(blur, conf)
        rng = None
        if a.check:
            m = RANGE_RE.search(f.parent.name)
            if m:
                rng = (float(m.group(1)), float(m.group(2)))
                if r["score"] is not None:
                    total += 1
                    hits += rng[0] <= r["score"] <= rng[1]
        fmt = lambda v: "  -  " if v is None else f"{v:5.1f}"
        print(f"{fmt(r['score'])}  bg {fmt(r['background'])}  {r['source'] or 'no detail'}  {f}" + (f"  [{rng[0]:g}-{rng[1]:g}]" if rng else ""))
        rows.append({"file": str(f), "score": r["score"], "background": r["background"], "source": r["source"]})
        if a.maps:
            cv2.imwrite(str(Path(a.maps) / (f.stem + "_blur.jpg")), overlay(img, blur, conf))
    if a.csv:
        with open(a.csv, "w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=["file", "score", "background", "source"])
            w.writeheader()
            w.writerows(rows)
    if a.check and total:
        print(f"\n{hits}/{total} ({100 * hits / total:.0f} %) inside their folder's range")


if __name__ == "__main__":
    main()
