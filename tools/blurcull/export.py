"""Write the trained network as the file LightCraft loads.

    python export.py --data D:/blurcull-data                       # runs/best.pt -> blurnet.safetensors
    python export.py --checkpoint runs/last.pt -o blurnet.safetensors

The file is a safetensors with BatchNorm folded into each convolution (names l0.weight ... l8.bias, head.weight,
head.bias) and metadata the app checks: format, arch, analysis size, score scale. Install it in LightCraft with
Library > Blur Cull > Install Model..., or the command  blur.model.install {"path": "...blurnet.safetensors"}.
"""
from __future__ import annotations

import argparse
import datetime
import hashlib
import json
from pathlib import Path

import torch
from safetensors.torch import save_file

from kernels import SIGMA_MAX
from net import ARCH_NAME, LUMA, STRIDE, BlurNet, FoldedBlurNet, fold

FORMAT = "lightcraft-blurnet"
FORMAT_VERSION = "1"


def metadata(analysis: int, note: str, step) -> dict:
    return {
        "format": FORMAT,
        "version": FORMAT_VERSION,
        "arch": ARCH_NAME,
        "analysis_long_edge": str(analysis),
        "sigma_max": str(SIGMA_MAX),
        "stride": str(STRIDE),
        "luma": ",".join(str(v) for v in LUMA),
        "created": datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "step": str(step),
        "note": note,
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", help="data folder (uses <data>/runs/best.pt, writes <data>/blurnet.safetensors)")
    ap.add_argument("--checkpoint")
    ap.add_argument("-o", "--out")
    ap.add_argument("--note", default="", help="free text kept in the file (e.g. what it was trained on)")
    ap.add_argument("--untrained", action="store_true", help="export random weights (for testing the plumbing only)")
    a = ap.parse_args()

    model = BlurNet()
    step, analysis = 0, 1600
    if not a.untrained:
        ck_path = Path(a.checkpoint) if a.checkpoint else Path(a.data) / "runs" / "best.pt"
        ck = torch.load(ck_path, map_location="cpu")
        model.load_state_dict(ck["model"])
        step, analysis = ck.get("step", 0), int(ck.get("analysis", 1600))
        print(f"{ck_path}: step {step}, validation {json.dumps(ck.get('val'))}")
    else:
        torch.manual_seed(7)
        model.eval()
    tensors = fold(model)

    # the folded network must give the same answer as the trained one
    with torch.no_grad():
        x = torch.rand(1, 1, 200, 280)
        d = (model.eval()(x) - FoldedBlurNet(tensors)(x)).abs().max().item()
    if d > 1e-4:
        raise SystemExit(f"folding changed the output by {d}; not writing")

    out = Path(a.out) if a.out else (Path(a.data) / "blurnet.safetensors" if a.data else Path("blurnet.safetensors"))
    note = a.note or ("untrained: plumbing test only" if a.untrained else "")
    save_file(tensors, str(out), metadata=metadata(analysis, note, step))
    digest = hashlib.sha256(out.read_bytes()).hexdigest()[:16]
    print(f"wrote {out} ({out.stat().st_size // 1024} KB, sha256 {digest}...)")


if __name__ == "__main__":
    main()
