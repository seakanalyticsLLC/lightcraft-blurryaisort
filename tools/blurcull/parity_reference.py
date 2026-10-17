"""Reference outputs for the Rust port (crates/blur/tests/parity.rs): the same photo through PyTorch and Rust.

    python parity_reference.py --model blurnet.safetensors --image some.jpg --out parity/
    LIGHTCRAFT_BLUR_PARITY=parity/ cargo test -p lightcraft-blur --test parity

Writes parity/model.safetensors (a copy), parity/input.rgb (8-bit RGB at the analysis size), parity/expected.json.
"""
import argparse
import json
import shutil
from pathlib import Path

import cv2
import numpy as np
import torch

from score import load
from subject import subject_score


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--image", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    model, meta = load(a.model)
    edge = int(meta.get("analysis_long_edge", 1600))
    img = cv2.imdecode(np.fromfile(a.image, dtype=np.uint8), cv2.IMREAD_COLOR)
    h, w = img.shape[:2]
    if max(h, w) > edge:
        d = edge / max(h, w)
        img = cv2.resize(img, (max(1, round(w * d)), max(1, round(h * d))), interpolation=cv2.INTER_AREA)
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    h, w = rgb.shape[:2]
    rgb.tofile(out / "input.rgb")
    # the same luminance as the Rust side (float, from the 8-bit RGB)
    y = (0.2126 * rgb[..., 0] + 0.7152 * rgb[..., 1] + 0.0722 * rgb[..., 2]).astype(np.float32) / 255.0
    with torch.no_grad():
        o = model(torch.from_numpy(y)[None, None])[0].numpy()
    blur, conf = o[0], 1 / (1 + np.exp(-o[1]))
    shutil.copyfile(a.model, out / "model.safetensors")
    r = subject_score(blur, conf)
    (out / "expected.json").write_text(json.dumps({
        "width": w, "height": h, "rows": int(blur.shape[0]), "cols": int(blur.shape[1]),
        "blur": blur.ravel().tolist(), "conf": conf.ravel().tolist(), "score": r["score"], "background": r["background"],
    }))
    print(f"{w}x{h} -> {blur.shape[1]}x{blur.shape[0]} cells, subject score {r['score']}")


if __name__ == "__main__":
    main()
