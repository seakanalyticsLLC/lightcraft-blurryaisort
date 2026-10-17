"""Turn your photo folders into a training cache.

    python prepare.py --data D:/blurcull-data

Expected layout of --data (see README.md):

    sharp/                      tack-sharp photos, any format LightCraft opens (subfolders are fine)
    rated/<name>_<lo>-<hi>/     real photos you judged by eye; their subject score belongs in lo..hi (0..100)
    rated/<...>/subjects.csv    optional: file,x0,y0,x1,y1 (0..1) where the subject is, for photos without a
                                focus point, when "the sharpest part" is not the subject

JPEG / PNG / TIFF / WebP are read directly; anything else (raws, HEIC, ...) is rendered by lightcraft-cli, with
LightCraft's default develop settings, so training sees what the app analyses. Writes <data>/cache/ and
<data>/cache/manifest.json. Run it again after adding photos: files already cached are skipped.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import cv2
import numpy as np

from synth import cell_crispness, cell_detail, luma

DIRECT = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".webp", ".bmp"}
RAWISH = {
    ".arw", ".cr2", ".cr3", ".crw", ".nef", ".nrw", ".raf", ".orf", ".rw2", ".pef", ".srw", ".dng", ".3fr", ".iiq",
    ".erf", ".kdc", ".mef", ".mos", ".mrw", ".x3f", ".srf", ".sr2", ".heic", ".heif", ".hif", ".avif", ".jxl",
}
SHARP_LONG_EDGE = 2400  # room for the +-25 % scale jitter around the 1600 analysis size
RANGE_RE = re.compile(r"(\d+(?:\.\d+)?)\s*-\s*(\d+(?:\.\d+)?)$")


def find_cli(explicit: str | None) -> str | None:
    if explicit:
        return explicit
    here = Path(__file__).resolve().parents[2]
    for name in ("lightcraft-cli.exe", "lightcraft-cli"):
        for sub in ("target/release", "target/debug"):
            p = here / sub / name
            if p.exists():
                return str(p)
    return shutil.which("lightcraft-cli")


def read_image(path: Path, cli: str | None, long_edge: int):
    ext = path.suffix.lower()
    img = None
    if ext in DIRECT:
        data = np.fromfile(str(path), dtype=np.uint8)  # works with non-ASCII Windows paths
        img = cv2.imdecode(data, cv2.IMREAD_COLOR)  # applies EXIF orientation
    elif cli:
        with tempfile.TemporaryDirectory() as td:
            out = os.path.join(td, "r.png")
            r = subprocess.run([cli, "render", str(path), "-o", out, "--size", str(long_edge)], capture_output=True, text=True)
            if r.returncode == 0 and os.path.exists(out):
                img = cv2.imread(out, cv2.IMREAD_COLOR)
            else:
                return None, (r.stderr or r.stdout).strip()[:200]
    else:
        return None, "needs lightcraft-cli (build it: cargo build --release -p lightcraft-cli)"
    if img is None:
        return None, "could not decode"
    h, w = img.shape[:2]
    if max(h, w) > long_edge:
        d = long_edge / max(h, w)
        img = cv2.resize(img, None, fx=d, fy=d, interpolation=cv2.INTER_AREA)
    return img, None


def sharp_check(img) -> dict:
    """How much of the frame has detail, and whether that detail is crisp (a soft 'sharp' photo poisons labels)."""
    y = luma(img.astype(np.float32) / 255)
    conf = cell_detail(y)
    crisp = cell_crispness(y)
    detailed = conf > 0.5
    return {
        "detail": float(detailed.mean()),
        "crisp": float(np.median(crisp[detailed])) if detailed.sum() > 8 else 0.0,
    }


def job(args):
    src, dst, kind, cli, long_edge = args
    if os.path.exists(dst):
        img = cv2.imread(dst, cv2.IMREAD_COLOR)
        err = None if img is not None else "cache unreadable"
    else:
        img, err = read_image(Path(src), cli, long_edge)
        if img is not None:
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            cv2.imwrite(dst, img)
    if img is None:
        return src, None, err
    info = {"width": int(img.shape[1]), "height": int(img.shape[0])}
    if kind == "sharp":
        info.update(sharp_check(img))
    return src, info, None


def is_val(rel: str) -> bool:
    return int(hashlib.sha1(rel.encode()).hexdigest(), 16) % 10 == 0


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", required=True, help="folder holding sharp/ and rated/")
    ap.add_argument("--cli", help="path to lightcraft-cli (default: this repo's target/release)")
    ap.add_argument("--analysis", type=int, default=1600, help="analysis long edge (must match the app; default 1600)")
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    a = ap.parse_args()

    data = Path(a.data)
    cache = data / "cache"
    cli = find_cli(a.cli)
    exts = DIRECT | RAWISH
    jobs, entries = [], {}

    sharp_dir = data / "sharp"
    for p in sorted(sharp_dir.rglob("*")) if sharp_dir.exists() else []:
        if p.suffix.lower() in exts:
            rel = p.relative_to(data).as_posix()
            dst = str(cache / "sharp" / (hashlib.sha1(rel.encode()).hexdigest()[:16] + ".png"))
            entries[str(p)] = {"file": rel, "cache": os.path.relpath(dst, data), "kind": "sharp", "val": is_val(rel)}
            jobs.append((str(p), dst, "sharp", cli, SHARP_LONG_EDGE))

    rated_dir = data / "rated"
    for d in sorted(x for x in rated_dir.iterdir() if x.is_dir()) if rated_dir.exists() else []:
        m = RANGE_RE.search(d.name)
        if not m:
            print(f"skipping rated/{d.name}: name must end in a score range like _15-35", file=sys.stderr)
            continue
        lo, hi = float(m.group(1)), float(m.group(2))
        if not (0 <= lo < hi <= 100):
            print(f"skipping rated/{d.name}: range must be within 0-100 with lo < hi", file=sys.stderr)
            continue
        subjects = {}
        sc = d / "subjects.csv"
        if sc.exists():
            with open(sc, newline="", encoding="utf-8-sig") as fh:
                for row in csv.reader(fh):
                    if len(row) >= 5 and not row[0].startswith("#"):
                        try:
                            subjects[row[0].strip()] = dict(zip(("x0", "y0", "x1", "y1"), map(float, row[1:5])))
                        except ValueError:
                            pass
        for p in sorted(d.rglob("*")):
            if p.suffix.lower() in exts:
                rel = p.relative_to(data).as_posix()
                dst = str(cache / "rated" / (hashlib.sha1(rel.encode()).hexdigest()[:16] + ".png"))
                e = {"file": rel, "cache": os.path.relpath(dst, data), "kind": "rated", "lo": lo, "hi": hi, "val": is_val(rel)}
                if p.name in subjects:
                    e["subject"] = subjects[p.name]
                entries[str(p)] = e
                jobs.append((str(p), dst, "rated", cli, a.analysis))

    if not jobs:
        sys.exit(f"nothing found: put photos in {sharp_dir} and/or {rated_dir}/<name>_<lo>-<hi>/")
    if not cli:
        print("note: lightcraft-cli not found; raw and HEIC files will be skipped", file=sys.stderr)

    ok, failed = [], []
    with ProcessPoolExecutor(a.workers) as pool:
        for i, (src, info, err) in enumerate(pool.map(job, jobs, chunksize=4)):
            if info is None:
                failed.append((src, err))
            else:
                entries[src].update(info)
                ok.append(entries[src])
            if (i + 1) % 25 == 0 or i + 1 == len(jobs):
                print(f"{i + 1}/{len(jobs)}", flush=True)

    soft = [e for e in ok if e["kind"] == "sharp" and e.get("crisp", 0) < 1.3]
    for e in soft:
        e["suspect"] = True
    manifest = {"analysis_long_edge": a.analysis, "entries": ok}
    (cache).mkdir(parents=True, exist_ok=True)
    with open(cache / "manifest.json", "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=1)

    n_sharp = sum(e["kind"] == "sharp" for e in ok)
    n_rated = len(ok) - n_sharp
    print(f"\ncached {n_sharp} sharp and {n_rated} rated photos -> {cache / 'manifest.json'}")
    for e in soft[:30]:
        print(f"  looks soft for sharp/: {e['file']} (crispness {e['crisp']:.2f}; sharp photos are usually 1.4 or more)")
    if soft:
        print(f"  {len(soft)} sharp/ photos look soft; train.py skips them unless --keep-suspect. Move them to rated/ instead.")
    for src, err in failed[:30]:
        print(f"  failed: {src}: {err}")
    if failed:
        print(f"  {len(failed)} files could not be read")


if __name__ == "__main__":
    main()
