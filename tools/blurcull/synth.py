"""Synthetic training examples from your sharp photos.

Every sharp photo yields endless examples: we blur it by a known amount with a physically different kind of blur each
time (defocus disc, hand shake, motion streak, soft lens; sometimes only part of the frame, like a shallow depth of
field) and then run the camera pipeline that follows (sensor noise, in-camera sharpening, JPEG). Because we did the
blurring, every 16 x 16 cell gets an exact label:

  sigma  equivalent blur sigma in analysis pixels (kernels.py)
  conf   0..1, whether the sharp original had enough detail there for blur to be measurable at all
         (flat sky has none; cells already soft in the original, crispness < 1.2, are dropped rather than mislabelled)
"""
from __future__ import annotations

import cv2
import numpy as np

from kernels import S_MAX, equivalent_sigma, random_kernel
from net import LUMA, STRIDE

CELL = STRIDE


def to_lin(x):
    return np.power(np.clip(x, 0, 1), 2.2)


def to_srgb(x):
    return np.power(np.clip(x, 0, 1), 1 / 2.2)


def luma(bgr):
    """BT.709 luminance of an OpenCV BGR float image."""
    return LUMA[2] * bgr[..., 0] + LUMA[1] * bgr[..., 1] + LUMA[0] * bgr[..., 2]


def _cells(a, hc, wc):
    return a[: hc * CELL, : wc * CELL].reshape(hc, CELL, wc, CELL)


def cell_detail(y: np.ndarray) -> np.ndarray:
    """Per-cell detail of the sharp image -> confidence 0..1."""
    gx = cv2.Sobel(y, cv2.CV_32F, 1, 0, ksize=3) / 8
    gy = cv2.Sobel(y, cv2.CV_32F, 0, 1, ksize=3) / 8
    g = np.sqrt(gx * gx + gy * gy)
    hc, wc = y.shape[0] // CELL, y.shape[1] // CELL
    p = np.percentile(_cells(g, hc, wc).transpose(0, 2, 1, 3).reshape(hc, wc, -1), 75, axis=2)
    return np.clip((p - 0.004) / 0.012, 0, 1).astype(np.float32)


def cell_crispness(y: np.ndarray) -> np.ndarray:
    """Per cell: how much a small extra blur (sigma 1) softens it. Edge gradient *energy* falls as an edge widens
    (summed gradients don't), and pixel differences have the narrowest support, so the ratio separates well:
    about 1.6 for a crisp edge, 1.25 at sigma 0.8, 1.06 at sigma 2."""

    def g2(a):
        gx = np.diff(a, axis=1)[:-1, :]
        gy = np.diff(a, axis=0)[:, :-1]
        return np.pad(gx * gx + gy * gy, ((0, 1), (0, 1)))

    hc, wc = y.shape[0] // CELL, y.shape[1] // CELL
    a = _cells(g2(y), hc, wc).sum(axis=(1, 3))
    b = _cells(g2(cv2.GaussianBlur(y, (0, 0), 1.0)), hc, wc).sum(axis=(1, 3))
    return np.sqrt(a / np.maximum(b, 1e-9))


def depth_mask(rng, h, w):
    """Smooth random 'in focus' mask: 1 = sharp plane, 0 = blurred plane."""
    m = np.zeros((h, w), np.float32)
    for _ in range(rng.integers(1, 4)):
        cy, cx = rng.uniform(0.1, 0.9) * h, rng.uniform(0.1, 0.9) * w
        ay, ax = rng.uniform(0.08, 0.45) * h, rng.uniform(0.08, 0.45) * w
        cv2.ellipse(m, (int(cx), int(cy)), (max(int(ax), 1), max(int(ay), 1)), rng.uniform(0, 180), 0, 360, 1.0, -1)
    if rng.random() < 0.3:  # a plane of focus across the frame
        y0 = int(rng.uniform(0.2, 0.7) * h)
        m[y0 : y0 + int(rng.uniform(0.1, 0.3) * h)] = 1
    if rng.random() < 0.5:
        m = 1 - m
    return cv2.GaussianBlur(m, (0, 0), rng.uniform(2, 12))


def sample_sigma(rng):
    """Blur amount: 18 % untouched, the rest spread evenly on the log scale up to SIGMA_MAX."""
    if rng.random() < 0.18:
        return 0.0
    return 2.0 ** rng.uniform(0.05, S_MAX) - 1.0


def camera(rng, lin):
    img = to_srgb(lin)
    if rng.random() < 0.65:  # sensor noise, brighter = more shot noise
        n = rng.uniform(0.003, 0.04)
        shot = rng.normal(size=img.shape[:2]).astype(np.float32)[..., None]
        chroma = rng.normal(size=img.shape).astype(np.float32) * 0.4
        img = img + n * np.sqrt(np.clip(img, 0.02, 1)) * (shot + chroma)
    if rng.random() < 0.45:  # in-camera / default sharpening
        bl = cv2.GaussianBlur(img, (0, 0), rng.uniform(0.7, 1.6))
        img = img + rng.uniform(0.2, 1.2) * (img - bl)
    img8 = (np.clip(img, 0, 1) * 255 + 0.5).astype(np.uint8)
    if rng.random() < 0.7:
        q = int(rng.integers(60, 98))
        img8 = cv2.imdecode(cv2.imencode(".jpg", img8, [cv2.IMWRITE_JPEG_QUALITY, q])[1], cv2.IMREAD_COLOR)
    return img8


def examples(rng, bgr8: np.ndarray, analysis_long_edge: int, crop: int, n_crops: int):
    """One blurred version of a sharp photo, cut into crops. Yields (y uint8 crop, sigma cells, conf cells, kind)."""
    long_edge = max(bgr8.shape[:2])
    # analysis scale with jitter: the same photo seen a bit larger or smaller
    target = analysis_long_edge * rng.uniform(0.75, 1.25)
    d = min(1.0, target / long_edge)
    base = cv2.resize(bgr8, None, fx=d, fy=d, interpolation=cv2.INTER_AREA).astype(np.float32) / 255
    if rng.random() < 0.5:
        base = cv2.flip(base, 1)
    h, w = base.shape[:2]
    if h < crop or w < crop:
        return
    y_sharp = luma(base)
    conf = cell_detail(y_sharp)
    conf = np.where((conf > 0.3) & (cell_crispness(y_sharp) < 1.2), 0.0, conf)
    hc, wc = conf.shape

    lin = to_lin(base) * rng.uniform(0.6, 1.4)
    k, kind = random_kernel(rng, sample_sigma(rng))
    if kind == "none":
        blurred, s = lin, 0.0
    else:
        blurred, s = cv2.filter2D(lin, -1, k, borderType=cv2.BORDER_REFLECT), equivalent_sigma(k)
    sig = np.full((hc, wc), s, np.float32)
    if s > 0.5 and rng.random() < 0.22:
        m = depth_mask(rng, h, w)
        blurred = m[..., None] * lin + (1 - m[..., None]) * blurred
        mc = _cells(m, hc, wc).mean(axis=(1, 3))
        sig = np.clip((1 - mc) * s, 0, None).astype(np.float32)
        conf = np.where((mc > 0.15) & (mc < 0.85), conf * 0.3, conf)  # cells straddling the focus edge
        kind += "+depth"
    y8 = (luma(camera(rng, blurred).astype(np.float32)) + 0.5).astype(np.uint8)

    nc = crop // CELL
    for _ in range(n_crops):
        for _try in range(6):  # prefer crops with something to judge
            cy = int(rng.integers(0, (h - crop) // CELL + 1)) * CELL
            cx = int(rng.integers(0, (w - crop) // CELL + 1)) * CELL
            cc = conf[cy // CELL : cy // CELL + nc, cx // CELL : cx // CELL + nc]
            if cc.shape == (nc, nc) and (cc.mean() > 0.25 or rng.random() < 0.12):
                break
        else:
            continue
        yield (
            y8[cy : cy + crop, cx : cx + crop],
            sig[cy // CELL : cy // CELL + nc, cx // CELL : cx // CELL + nc],
            cc,
            kind,
        )
