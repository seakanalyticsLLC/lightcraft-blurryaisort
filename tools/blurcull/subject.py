"""From a blur map to one number per photo: the subject's blur score (0 = tack sharp, 100 = unrecognisable).

The Rust twin is crates/sharpness/src/subject.rs; keep both in step.

Where is the subject?
  1. the camera's focus point, when the file records one (LightCraft reads it into meta.focus)
  2. otherwise face / pet regions (meta.regions)
  3. otherwise "auto": the sharpest part of the frame that has detail. If anything in the photo is in focus, that's
     almost always what the photographer meant; a sharp subject on a creamy background scores sharp, a missed-focus
     or shaken shot has no sharp part and scores blurry.

Inside the subject area we take a low percentile of the blur over cells the network is confident about, so a few
soft cells at the subject's edge don't drag it down.
"""
from __future__ import annotations

import numpy as np

from kernels import sigma_to_score

CONF_MIN = 0.5  # cells below this confidence are not judged
AUTO_PERCENTILE = 15  # auto subject: the sharpest 15 % of judged cells
REGION_PERCENTILE = 35  # focus point / faces: robust low percentile inside the region
MIN_CELLS = 4


def region_cells(rect, hc, wc, grow=0.5):
    """Cell mask for a normalized rect {x0,y0,x1,y1}, grown by `grow` of its size on each side (focus boxes are small)."""
    x0, y0, x1, y1 = rect["x0"], rect["y0"], rect["x1"], rect["y1"]
    gw, gh = (x1 - x0) * grow, (y1 - y0) * grow
    x0, x1 = max(0.0, x0 - gw), min(1.0, x1 + gw)
    y0, y1 = max(0.0, y0 - gh), min(1.0, y1 + gh)
    m = np.zeros((hc, wc), bool)
    cx0, cx1 = int(np.floor(x0 * wc)), int(np.ceil(x1 * wc))
    cy0, cy1 = int(np.floor(y0 * hc)), int(np.ceil(y1 * hc))
    m[cy0 : max(cy1, cy0 + 1), cx0 : max(cx1, cx0 + 1)] = True
    return m


def subject_score(blur_log2: np.ndarray, conf: np.ndarray, focus=None, regions=()):
    """blur_log2, conf: (hc, wc) network outputs (conf as probability). Returns dict(score, background, source, cells)."""
    hc, wc = blur_log2.shape
    judged = conf >= CONF_MIN
    background = float(np.median(blur_log2[judged])) if judged.sum() >= MIN_CELLS else None

    def pct(mask, q):
        v = blur_log2[mask & judged]
        return float(np.percentile(v, q)) if v.size >= MIN_CELLS else None

    b, source, cells = None, None, 0
    if focus:
        for grow in (0.5, 1.5, 3.0):  # widen until there's enough detail to judge
            m = region_cells(focus, hc, wc, grow)
            b = pct(m, REGION_PERCENTILE)
            if b is not None:
                source, cells = "focus", int((m & judged).sum())
                break
    if b is None and regions:
        m = np.zeros((hc, wc), bool)
        for r in regions:
            m |= region_cells(r, hc, wc, 0.15)
        b = pct(m, REGION_PERCENTILE)
        if b is not None:
            source, cells = "faces", int((m & judged).sum())
    if b is None:
        b = pct(np.ones((hc, wc), bool), AUTO_PERCENTILE)
        if b is not None:
            source, cells = "auto", int(judged.sum())
    to_score = lambda v: None if v is None else float(sigma_to_score(2.0**v - 1.0))
    return {"score": to_score(b), "background": to_score(background), "source": source, "cells": cells}
