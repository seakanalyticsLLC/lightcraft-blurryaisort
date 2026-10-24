"""Check gpu_synth.py against synth.py: same photos, same random seed, sensor noise off on both sides (the only
randomness the two draw differently), so each render must take the same decisions and give the same labels.

    python check_gpu_synth.py --data E:/LLMs/bluecull-data --renders 200          # a sample of the sharp cache
    python check_gpu_synth.py photo1.jpg photo2.png --renders 50 --device cpu     # any photos, no GPU needed
    python check_gpu_synth.py --data E:/LLMs/bluecull-data --bench 300            # speed of real training steps

Reports, over all crops: whether the kind of blur matched, the sigma labels' largest difference, the confidence
labels' mean difference and the pixels' mean difference (0..255; JPEG is the GPU's approximation, so expect ~0.5).
A crop whose confidence lands within a hair of the crop picker's 0.25 cutoff can make the two sides draw a different
number of random values from there on: the remaining crops of that render are cut elsewhere (same blur, other
places), so they are left out and counted as "drifted at a 0.25 tie". Then times both: CPU (one process) vs GPU.

--bench N skips the comparison and runs N real training steps the way train.py does on the GPU (rendering threads,
shuffle pool, rated photos every 4th step), then prints seconds per step and what a 30,000-step run would take.
Run it with nothing else using the GPU.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import cv2
import numpy as np
import torch

import synth
from gpu_synth import GpuSynth, correlate, jpeg_roundtrip
from kernels import random_kernel, streak_kernel


class NoNoise:
    """A numpy Generator whose big normal() draws (synth.camera's noise fields) return zeros without consuming."""

    def __init__(self, seed):
        self.g = np.random.default_rng(seed)

    def normal(self, loc=0.0, scale=1.0, size=None):
        if isinstance(size, tuple) and len(size) >= 2:
            return np.zeros(size)
        return self.g.normal(loc, scale, size)

    def __getattr__(self, name):
        return getattr(self.g, name)


def unit_checks(img, dev):
    """Component checks: the big blur and JPEG against OpenCV."""
    x = torch.from_numpy(img.astype(np.float32) / 255).permute(2, 0, 1).to(dev)
    rng = np.random.default_rng(5)
    worst = 0.0
    for _ in range(6):
        k, _ = random_kernel(rng, float(rng.uniform(0.5, 24)))
        ref = cv2.filter2D(img.astype(np.float32) / 255, -1, k, borderType=cv2.BORDER_REFLECT)
        got = correlate(x, k).permute(1, 2, 0).cpu().numpy()
        worst = max(worst, float(np.abs(ref - got).max()))
    k = streak_kernel(480.0, 0.05)
    ref = cv2.filter2D(img.astype(np.float32) / 255, -1, k, borderType=cv2.BORDER_REFLECT)
    worst = max(worst, float(np.abs(ref - correlate(x, k).permute(1, 2, 0).cpu().numpy()).max()))
    print(f"blur (filter2D vs GPU, incl. a 480 px streak): max difference {worst:.2e}")
    for q in (60, 80, 97):
        ref = cv2.imdecode(cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, q])[1], cv2.IMREAD_COLOR).astype(np.float32)
        got = jpeg_roundtrip(torch.from_numpy(img.astype(np.float32)).permute(2, 0, 1).to(dev), q).permute(1, 2, 0).cpu().numpy()
        ly = synth.luma(ref) - synth.luma(got)
        print(f"JPEG q{q}: luma mean |diff| {np.abs(ly).mean():.3f}, max {np.abs(ly).max():.1f} (0..255)")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("photos", nargs="*")
    ap.add_argument("--data", help="data folder: use photos from cache/manifest.json (sharp)")
    ap.add_argument("--cache", default="cache", help="cache folder under --data (default cache)")
    ap.add_argument("--renders", type=int, default=100)
    ap.add_argument("--analysis", type=int, default=1600)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--bench", type=int, default=0, help="time N real training steps instead (needs --data)")
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--crops-per", type=int, default=32)
    ap.add_argument("--renderers", type=int, default=3)
    a = ap.parse_args()
    if a.bench:
        return bench(a)

    files = [Path(p) for p in a.photos]
    if a.data:
        man = json.loads((Path(a.data) / a.cache / "manifest.json").read_text(encoding="utf-8"))
        a.analysis = int(man.get("analysis_long_edge", a.analysis))
        files += [Path(a.data) / e["cache"] for e in man["entries"] if e["kind"] == "sharp" and not e.get("suspect")]
    if not files:
        raise SystemExit("give photos or --data")
    pick = np.random.default_rng(a.seed)
    files = [files[int(i)] for i in pick.choice(len(files), size=min(len(files), a.renders), replace=False)]
    imgs = [im for im in (cv2.imread(str(f), cv2.IMREAD_COLOR) for f in files) if im is not None]
    dev = torch.device(a.device)
    print(f"{len(imgs)} photos, device {dev}")
    unit_checks(imgs[0], dev)

    gs = GpuSynth(dev, seed=a.seed, noise=False)
    n_r, kinds_ok, n_crops, crop_count_ok, drifted = 0, 0, 0, 0, 0
    sig_max, conf_diffs, y_diffs = 0.0, [], []
    by_kind: dict[str, list] = {}
    t_cpu = t_gpu = 0.0
    for i in range(a.renders):
        img = imgs[i % len(imgs)]
        seed = a.seed * 100003 + i
        t = time.perf_counter()
        cpu = list(synth.examples(NoNoise(seed), img, a.analysis, 128, 8))
        t_cpu += time.perf_counter() - t
        rng = np.random.default_rng(seed)
        t = time.perf_counter()
        r = gs.render(rng, torch.from_numpy(img).to(dev).permute(2, 0, 1), a.analysis, 128)
        got = gs.crops(rng, r, 128, 8) if r is not None else None
        if dev.type == "cuda":
            torch.cuda.synchronize()
        t_gpu += time.perf_counter() - t
        n_r += 1
        if r is None or got is None:
            kinds_ok += not cpu
            continue
        kind_cpu = cpu[0][3] if cpu else None
        kinds_ok += kind_cpu == r[3]
        y, b, c = (t.cpu().numpy()[:, 0] for t in got)
        crop_count_ok += len(cpu) == y.shape[0]
        for j, (y_c, s_c, c_c, _) in enumerate(cpu[: y.shape[0]]):
            if j > 0 and abs(float(cpu[j - 1][2].mean()) - 0.25) < 2e-3:
                drifted += 1
                break
            n_crops += 1
            sig_max = max(sig_max, float(np.abs(np.log2(1 + np.maximum(s_c, 0)) - b[j]).max()))
            conf_diffs.append(float(np.abs(c_c - c[j]).mean()))
            d = float(np.abs(y_c.astype(np.float32) - y[j] * 255).mean())
            y_diffs.append(d)
            by_kind.setdefault(r[3], []).append(d)
    print(f"\n{n_r} renders: same kind of blur in {kinds_ok}, same number of crops in {crop_count_ok}, "
          f"drifted at a 0.25 tie in {drifted}")
    print(f"{n_crops} crops: sigma label max |diff| {sig_max:.2e} (log2 units)   conf mean |diff| {np.mean(conf_diffs):.4f}"
          f"   pixels mean |diff| {np.mean(y_diffs):.2f} (worst crop {np.max(y_diffs):.2f}) of 255")
    for k in sorted(by_kind):
        print(f"   {k:24s} {len(by_kind[k]):4d} crops   pixels mean |diff| {np.mean(by_kind[k]):.2f}")
    print(f"\ntime per render: CPU {1000 * t_cpu / n_r:.0f} ms (one process)   {dev.type.upper()} {1000 * t_gpu / n_r:.0f} ms")


def bench(a):
    from gpu_synth import GpuBatches
    from net import BlurNet
    from train import CROP, augment, cell_loss, rated_loss

    if not a.data:
        raise SystemExit("--bench needs --data")
    data = Path(a.data)
    man = json.loads((data / a.cache / "manifest.json").read_text(encoding="utf-8"))
    analysis = int(man.get("analysis_long_edge", a.analysis))
    ents = man["entries"]
    sharp = [e["cache"] for e in ents if e["kind"] == "sharp" and not e.get("suspect") and not e["val"]]
    rated = [e for e in ents if e["kind"] == "rated" and not e["val"]]
    dev = torch.device(a.device)
    print(f"bench: {len(sharp)} sharp / {len(rated)} rated photos, device {dev}, batch {a.batch}, "
          f"{a.crops_per} crops per render, {a.renderers} rendering threads")
    model = BlurNet().to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    feed = GpuBatches(data, sharp, analysis, CROP, a.batch, dev, a.crops_per, seed=a.seed, renderers=a.renderers)
    rng = np.random.default_rng(a.seed)

    def step(i):
        x, b, c = next(feed)
        x, b, c = augment(x, b, c, rng)
        l_blur, l_conf, _ = cell_loss(model(x), b, c)
        loss = l_blur + 0.3 * l_conf
        if rated and i % 4 == 0:
            lr = rated_loss(model, data, rated, rng, dev, 2)
            if lr is not None:
                loss = loss + lr
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()

    warm = min(50, max(5, a.bench // 5))  # fills the shuffle pool, caches the rated photos
    t = time.perf_counter()
    for i in range(warm):
        step(i)
    print(f"warm-up ({warm} steps, rated photos loaded, pool filled): {time.perf_counter() - t:.1f} s")
    if dev.type == "cuda":
        torch.cuda.synchronize()
    r0, t = feed.renders, time.perf_counter()
    for i in range(a.bench):
        step(i)
    if dev.type == "cuda":
        torch.cuda.synchronize()
    dt = (time.perf_counter() - t) / a.bench
    renders = (feed.renders - r0) / a.bench
    print(f"\n{dt:.3f} s/step ({1 / dt:.1f} steps/s, {renders:.1f} renders/step)  ->  30,000 steps ~ {30000 * dt / 3600:.2f} h")
    if dev.type == "cuda":
        print(f"GPU memory: {torch.cuda.max_memory_allocated(dev) / 2**30:.1f} GB peak")


if __name__ == "__main__":
    main()
