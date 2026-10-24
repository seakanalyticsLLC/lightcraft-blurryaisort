"""Train BlurNet on your photos (after prepare.py).

    python train.py --data D:/blurcull-data                 # GPU if PyTorch sees one, else CPU
    python train.py --data D:/blurcull-data --resume        # carry on from runs/last.pt
    python train.py --data D:/blurcull-data --cache cache-3200 --out D:/blurcull-data/runs-3200   # another size

The blurred training examples are rendered on the GPU (gpu_synth.py) when PyTorch sees a CUDA card, else by CPU
worker processes (synth.py); --synth cpu|gpu forces one. Both make the same examples (check_gpu_synth.py).

Two kinds of supervision:

* sharp/ photos, blurred synthetically (synth.py): exact per-cell labels, endless variety. This teaches the
  network what blur *is*.
* burst pairs: two rated photos taken within --pair-window seconds of each other (same burst: same car, light and
  place) that you put in different folders. The one you rated sharper must score lower, by --pair-margin. This teaches
  the fine differences between neighbouring folders (sharp vs slightly soft) that one-photo ranges barely constrain.
* rated/ photos: real blur you judged by eye. The photo's subject score (subject.py) is pushed into the folder's
  lo..hi range. This ties the score scale to your eye and to real-world blur the synthetic kinds miss.

Checkpoints go to --out (default <data>/runs): last.pt every --save-every steps, best.pt on the best validation.
Then run export.py to make the file LightCraft loads.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import re
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from kernels import S_MAX
from net import BlurNet
from gpu_synth import GpuBatches, GpuSynth
from synth import examples, luma
from subject import AUTO_PERCENTILE, CONF_MIN, REGION_PERCENTILE, region_cells

CROP = 128


def device_auto(name: str):
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


class Synthetic(torch.utils.data.IterableDataset):
    """Endless stream of (y, b, conf) crops from the sharp cache."""

    def __init__(self, data: Path, files: list[str], analysis: int, crops_per: int = 8, seed: int = 0):
        self.data, self.files, self.analysis, self.crops_per, self.seed = data, files, analysis, crops_per, seed

    def __iter__(self):
        info = torch.utils.data.get_worker_info()
        wid = info.id if info else 0
        rng = np.random.default_rng(self.seed * 1000 + wid + int(time.time() * 1000) % 100000)
        cache: dict[str, np.ndarray] = {}
        while True:
            f = self.files[int(rng.integers(len(self.files)))]
            img = cache.get(f)
            if img is None:
                img = cv2.imread(str(self.data / f), cv2.IMREAD_COLOR)
                if img is None:
                    continue
                if len(cache) > 24:
                    cache.pop(next(iter(cache)))
                cache[f] = img
            for y, s, c, _ in examples(rng, img, self.analysis, CROP, self.crops_per):
                yield (
                    torch.from_numpy(y.astype(np.float32) / 255.0)[None],
                    torch.from_numpy(np.log2(1.0 + np.maximum(s, 0))).float()[None],
                    torch.from_numpy(c.astype(np.float32))[None],
                )


def augment(x, b, c, rng):
    k = int(rng.integers(4))
    x, b, c = (torch.rot90(t, k, (2, 3)) for t in (x, b, c))
    if rng.random() < 0.5:
        x, b, c = (torch.flip(t, (3,)) for t in (x, b, c))
    n, dev = x.shape[0], x.device
    gain = torch.empty(n, 1, 1, 1, device=dev).uniform_(0.6, 1.3)
    off = torch.empty(n, 1, 1, 1, device=dev).uniform_(-0.15, 0.15)
    m = x.mean(dim=(2, 3), keepdim=True)
    x = ((x - m) * gain + m + off).clamp(0, 1)
    inv = (torch.rand(n, device=dev) < 0.1).view(n, 1, 1, 1)
    return torch.where(inv, 1 - x, x), b, c


def cell_loss(out, b, c, ring=1):
    """Per-cell blur regression (weighted by detail) + confidence classification. Border cells see padding: skipped."""
    sl = slice(ring, out.shape[2] - ring)
    pb, pc, tb, tc = out[:, :1, sl, sl], out[:, 1:, sl, sl], b[:, :, sl, sl], c[:, :, sl, sl]
    w = tc
    l_blur = (F.smooth_l1_loss(pb, tb, reduction="none", beta=0.25) * w).sum() / w.sum().clamp(min=1)
    l_conf = F.binary_cross_entropy_with_logits(pc, tc)
    mae = ((pb - tb).abs() * w).sum() / w.sum().clamp(min=1)
    return l_blur, l_conf, mae


_RATED: dict[str, torch.Tensor] = {}  # rated photos' luma, kept on the training device after the first read (fp16)


def load_rated(data: Path, e: dict, dev=None):
    y = _RATED.get(e["cache"])
    if y is None:
        img = cv2.imread(str(data / e["cache"]), cv2.IMREAD_COLOR)
        if img is None:
            return None
        y = torch.from_numpy(luma(img.astype(np.float32) / 255))[None, None]
        if dev is not None:
            y = y.to(dev)
        if len(_RATED) < 4000:
            _RATED[e["cache"]] = y.half()  # luma to 1/2048: far finer than the 8-bit photo it came from
            return y
    return y.float()


def subject_b(out, subject=None):
    """Differentiable subject blur (log2 units) of one photo's network output, as subject.py computes it."""
    blur = out[0, 0]
    conf = torch.sigmoid(out[0, 1]).detach()
    hc, wc = blur.shape
    judged = conf >= CONF_MIN
    if subject is not None:
        m = torch.from_numpy(region_cells(subject, hc, wc, 0.15)).to(blur.device) & judged
        if m.sum() >= 4:
            return torch.quantile(blur[m], REGION_PERCENTILE / 100)
    if judged.sum() >= 4:
        return torch.quantile(blur[judged], AUTO_PERCENTILE / 100)
    return None


_STAMP = re.compile(r"(\d{8})-(\d{2})-(\d{2})-(\d{2})")


def burst_pairs(entries: list[dict], window: float) -> list[tuple[dict, dict]]:
    """(sharper, blurrier) pairs of rated photos shot within `window` seconds of each other (camera-style
    YYYYMMDD-HH-MM-SS file names), from folders whose ranges are ordered: neither range contains the other."""
    stamped = []
    for e in entries:
        m = _STAMP.search(Path(e["file"]).name)
        if m:
            day, h, mi, sec = m.groups()
            stamped.append((day, int(h) * 3600 + int(mi) * 60 + int(sec), e))
    stamped.sort(key=lambda t: (t[0], t[1]))
    pairs = []
    for i, (d1, t1, a) in enumerate(stamped):
        for d2, t2, b in stamped[i + 1 :]:
            if d2 != d1 or t2 - t1 > window:
                break
            if (a["lo"], a["hi"]) == (b["lo"], b["hi"]):
                continue
            if (a["lo"] <= b["lo"] and b["hi"] <= a["hi"]) or (b["lo"] <= a["lo"] and a["hi"] <= b["hi"]):
                continue  # one range inside the other (e.g. artistic_30-80 around blurry_38-65): no order
            pairs.append((a, b) if a["lo"] + a["hi"] < b["lo"] + b["hi"] else (b, a))
    return pairs


def pair_loss(model, data, pairs, rng, dev, n, margin):
    """Hinge: the sharper-rated photo of each sampled burst pair scores at least `margin` (log2 sigma) lower."""
    losses = []
    for i in rng.choice(len(pairs), size=min(n, len(pairs)), replace=False):
        a, b = pairs[int(i)]
        ya, yb = load_rated(data, a, dev), load_rated(data, b, dev)
        if ya is None or yb is None:
            continue
        ba, bb = subject_b(model(ya), a.get("subject")), subject_b(model(yb), b.get("subject"))
        if ba is None or bb is None:
            continue
        losses.append(F.relu(margin + ba - bb))
    return torch.stack(losses).mean() if losses else None


def rated_loss(model, data, entries, rng, dev, n):
    """Push each rated photo's subject score into its folder's range (hinge, in log2-sigma units)."""
    losses = []
    for i in rng.choice(len(entries), size=min(n, len(entries)), replace=False):
        e = entries[int(i)]
        y = load_rated(data, e, dev)
        if y is None:
            continue
        out = model(y.to(dev))
        b = subject_b(out, e.get("subject"))
        if b is None:
            continue
        lo, hi = e["lo"] * S_MAX / 100, e["hi"] * S_MAX / 100
        losses.append(F.relu(lo - b) + F.relu(b - hi))
    return torch.stack(losses).mean() if losses else None


@torch.no_grad()
def validate(model, val_batches, rated_val, data, dev):
    model.eval()
    maes = []
    for x, b, c in val_batches:
        _, _, mae = cell_loss(model(x.to(dev)), b.to(dev), c.to(dev))
        maes.append(mae.item())
    inside, total, err = 0, 0, []
    for e in rated_val:
        y = load_rated(data, e, dev)
        if y is None:
            continue
        b = subject_b(model(y.to(dev)), e.get("subject"))
        if b is None:
            continue
        s = float(b) / S_MAX * 100
        total += 1
        inside += e["lo"] <= s <= e["hi"]
        err.append(max(0.0, e["lo"] - s, s - e["hi"]))
    model.train()
    return {
        "synthetic_mae_log2": float(np.mean(maes)) if maes else None,
        "rated_in_range": inside / total if total else None,
        "rated_mean_miss": float(np.mean(err)) if err else None,
        "rated_n": total,
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", required=True)
    ap.add_argument("--cache", default="cache", help="cache folder under --data written by prepare.py (default cache)")
    ap.add_argument("--out", help="checkpoint folder (default <data>/runs)")
    ap.add_argument("--steps", type=int, default=30000)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--rated-weight", type=float, default=0.5, help="weight of the rated-photo loss")
    ap.add_argument("--rated-every", type=int, default=4, help="add rated photos every N steps")
    ap.add_argument("--rated-per-step", type=int, default=2)
    ap.add_argument("--pair-weight", type=float, default=1.0, help="weight of the burst-pair loss (0 = off)")
    ap.add_argument("--pair-window", type=float, default=2.0, help="seconds between two shots of the same burst")
    ap.add_argument("--pair-margin", type=float, default=3.0, help="score points the sharper photo must be lower by")
    ap.add_argument("--pairs-per-step", type=int, default=2)
    ap.add_argument("--workers", type=int, default=max(0, min(8, (os.cpu_count() or 2) - 1)), help="CPU workers (--synth cpu)")
    ap.add_argument("--synth", choices=["auto", "cpu", "gpu"], default="auto", help="where training examples are rendered")
    ap.add_argument("--crops-per", type=int, help="crops cut from each rendered photo (default 8 on the CPU, 32 on the GPU)")
    ap.add_argument("--renderers", type=int, default=3, help="GPU rendering threads working ahead of training (--synth gpu)")
    ap.add_argument("--device", default="auto", help="auto | cpu | cuda | mps")
    ap.add_argument("--val-every", type=int, default=1000)
    ap.add_argument("--save-every", type=int, default=500)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--keep-suspect", action="store_true", help="also train on sharp/ photos prepare.py found soft")
    a = ap.parse_args()

    data = Path(a.data)
    man_path = data / a.cache / "manifest.json"
    if not man_path.exists():
        raise SystemExit(f"{man_path} not found: run prepare.py --data {data} --cache {a.cache} first")
    man = json.loads(man_path.read_text(encoding="utf-8"))
    analysis = int(man.get("analysis_long_edge", 1600))
    ents = man["entries"]
    sharp = [e for e in ents if e["kind"] == "sharp" and (a.keep_suspect or not e.get("suspect"))]
    sharp_tr = [e["cache"] for e in sharp if not e["val"]]
    sharp_va = [e["cache"] for e in sharp if e["val"]] or sharp_tr[:1]
    rated_tr = [e for e in ents if e["kind"] == "rated" and not e["val"]]
    rated_va = [e for e in ents if e["kind"] == "rated" and e["val"]]
    if len(sharp_tr) < 5:
        raise SystemExit(f"only {len(sharp_tr)} usable sharp photos; add at least 50 (a few hundred is better)")
    pairs = burst_pairs(rated_tr, a.pair_window) if a.pair_weight > 0 else []
    print(f"sharp: {len(sharp_tr)} train / {len(sharp_va)} val   rated: {len(rated_tr)} train / {len(rated_va)} val   "
          f"burst pairs: {len(pairs)}")

    dev = device_auto(a.device)
    gpu_synth = a.synth == "gpu" or (a.synth == "auto" and dev.type == "cuda")
    crops_per = a.crops_per or (32 if gpu_synth else 8)
    where = f"by {a.renderers} gpu_synth threads on {dev}" if gpu_synth else f"by {a.workers} CPU workers"
    print("device", dev, f"  examples rendered {where}, {crops_per} crops each")
    out = Path(a.out) if a.out else data / "runs"
    out.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(0)
    model = BlurNet().to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=a.lr, total_steps=a.steps, pct_start=0.06)
    step, best = 0, math.inf
    if a.resume and (out / "last.pt").exists():
        ck = torch.load(out / "last.pt", map_location=dev)
        model.load_state_dict(ck["model"])
        opt.load_state_dict(ck["opt"])
        sched.load_state_dict(ck["sched"])
        step, best = ck["step"], ck.get("best", math.inf)
        print("resumed at step", step)

    # fixed validation crops
    vrng = np.random.default_rng(1234)
    vx, vb, vc = [], [], []
    vsynth = GpuSynth(dev, seed=1234) if gpu_synth else None
    for f in sharp_va:
        img = cv2.imread(str(data / f), cv2.IMREAD_COLOR)
        if img is None:
            continue
        for _ in range(max(1, 600 // len(sharp_va))):
            if vsynth is not None:
                r = vsynth.render(vrng, torch.from_numpy(img).to(dev).permute(2, 0, 1), analysis, CROP)
                got = vsynth.crops(vrng, r, CROP, 4) if r is not None else None
                if got is not None:
                    vx += list(got[0][:, 0].cpu().numpy())
                    vb += list(got[1][:, 0].cpu().numpy())
                    vc += list(got[2][:, 0].cpu().numpy())
                continue
            for y, s, c, _ in examples(vrng, img, analysis, CROP, 4):
                vx.append(y.astype(np.float32) / 255)
                vb.append(np.log2(1.0 + np.maximum(s, 0)))
                vc.append(c)
    val_batches = []
    for i in range(0, len(vx), 128):
        val_batches.append(
            (
                torch.from_numpy(np.stack(vx[i : i + 128])).float()[:, None],
                torch.from_numpy(np.stack(vb[i : i + 128])).float()[:, None],
                torch.from_numpy(np.stack(vc[i : i + 128])).float()[:, None],
            )
        )

    if gpu_synth:
        dl = GpuBatches(data, sharp_tr, analysis, CROP, a.batch, dev, crops_per,
                        seed=int(time.time() * 1000) % 2**31, renderers=a.renderers)
    else:
        ds = Synthetic(data, sharp_tr, analysis, crops_per=crops_per)
        dl = torch.utils.data.DataLoader(ds, batch_size=a.batch, num_workers=a.workers, persistent_workers=a.workers > 0)
    rng = np.random.default_rng(step)
    t0, run = time.time(), []
    model.train()
    for x, b, c in dl:
        if step >= a.steps:
            break
        x, b, c = augment(x, b, c, rng)
        x, b, c = x.to(dev), b.to(dev), c.to(dev)
        l_blur, l_conf, mae = cell_loss(model(x), b, c)
        loss = l_blur + 0.3 * l_conf
        l_rated = l_pair = None
        if rated_tr and step % a.rated_every == 0:
            l_rated = rated_loss(model, data, rated_tr, rng, dev, a.rated_per_step)
            if l_rated is not None:
                loss = loss + a.rated_weight * l_rated
            if pairs:
                l_pair = pair_loss(model, data, pairs, rng, dev, a.pairs_per_step, a.pair_margin * S_MAX / 100)
                if l_pair is not None:
                    loss = loss + a.pair_weight * l_pair
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        opt.step()
        sched.step()
        step += 1
        run.append((l_blur.item(), l_conf.item(), mae.item(), l_rated.item() if l_rated is not None else np.nan,
                    l_pair.item() if l_pair is not None else np.nan))
        if step % 100 == 0:
            r = np.nanmean(run[-100:], axis=0)
            rate = (time.time() - t0) / len(run)
            print(f"step {step}/{a.steps}  blur {r[0]:.4f}  conf {r[1]:.4f}  mae {r[2]:.3f}  rated {r[3]:.3f}  "
                  + (f"pairs {r[4]:.3f}  " if pairs else "") +
                  f"{rate:.2f}s/step  ~{(a.steps - step) * rate / 3600:.1f} h left", flush=True)
        if step % a.save_every == 0 or step == a.steps:
            torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "sched": sched.state_dict(),
                        "step": step, "best": best, "analysis": analysis}, out / "last.pt")
        if step % a.val_every == 0 or step == a.steps:
            v = validate(model, val_batches, rated_va, data, dev)
            # one number to pick the best checkpoint: synthetic error plus rated misses (both in log2-sigma units)
            score = (v["synthetic_mae_log2"] or 0) + (v["rated_mean_miss"] or 0) * S_MAX / 100
            print("validation", json.dumps(v), flush=True)
            if score < best:
                best = score
                torch.save({"model": model.state_dict(), "step": step, "val": v, "analysis": analysis}, out / "best.pt")
                print(f"  new best -> {out / 'best.pt'}", flush=True)
    print("done. next: python export.py --data", data)


if __name__ == "__main__":
    main()
