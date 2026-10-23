"""Blur kernels with a common severity measure.

Every kernel is described by its *equivalent sigma*: the square root of the largest eigenvalue of the
kernel's spatial covariance, in pixels at the analysis resolution. A Gaussian of sigma s scores s, a
defocus disc of radius R scores R/2, a straight motion streak of length L scores L/sqrt(12). This one
number is what the model learns to predict, and what the plugin's 0-100 blur score is built from.
"""
from __future__ import annotations

import math

import cv2
import numpy as np

# Score scale: score = 100 * log2(1 + sigma) / log2(1 + SIGMA_MAX), clipped to 0..100.
SIGMA_MAX = 24.0
S_MAX = math.log2(1.0 + SIGMA_MAX)


def sigma_to_score(sigma):
    s = np.log2(1.0 + np.maximum(np.asarray(sigma, dtype=np.float64), 0.0))
    return np.clip(100.0 * s / S_MAX, 0.0, 100.0)


def score_to_sigma(score):
    return 2.0 ** (np.asarray(score, dtype=np.float64) / 100.0 * S_MAX) - 1.0


def equivalent_sigma(k: np.ndarray) -> float:
    k = k.astype(np.float64)
    k = k / k.sum()
    h, w = k.shape
    ys, xs = np.mgrid[0:h, 0:w]
    my, mx = (k * ys).sum(), (k * xs).sum()
    cyy = (k * (ys - my) ** 2).sum()
    cxx = (k * (xs - mx) ** 2).sum()
    cxy = (k * (ys - my) * (xs - mx)).sum()
    lam = 0.5 * (cxx + cyy) + math.sqrt(0.25 * (cxx - cyy) ** 2 + cxy**2)
    return math.sqrt(max(lam, 0.0))


def _odd(n: float) -> int:
    n = int(math.ceil(n))
    return n + 1 if n % 2 == 0 else n


def gaussian_kernel(sigma: float, aniso: float = 1.0, angle: float = 0.0) -> np.ndarray:
    """Gaussian; aniso < 1 squeezes the minor axis (lens softness, slight drift)."""
    s1, s2 = sigma, max(sigma * aniso, 0.05)
    size = _odd(6 * s1 + 1)
    r = size // 2
    ys, xs = np.mgrid[-r : r + 1, -r : r + 1].astype(np.float64)
    c, s = math.cos(angle), math.sin(angle)
    u, v = c * xs + s * ys, -s * xs + c * ys
    k = np.exp(-0.5 * ((u / s1) ** 2 + (v / s2) ** 2))
    return (k / k.sum()).astype(np.float32)


def disc_kernel(radius: float, cat_eye: float = 0.0) -> np.ndarray:
    """Defocus: an anti-aliased disc (optionally clipped like a cat's-eye bokeh)."""
    size = _odd(2 * radius + 3)
    r = size // 2
    ss = 4  # supersample for smooth edges at small radii
    ys, xs = np.mgrid[0 : size * ss, 0 : size * ss].astype(np.float64)
    ys = (ys + 0.5) / ss - r - 0.5
    xs = (xs + 0.5) / ss - r - 0.5
    inside = (xs**2 + ys**2) <= radius**2
    if cat_eye > 0:
        inside &= ((xs - cat_eye * radius) ** 2 + ys**2) <= radius**2
    k = inside.reshape(size, ss, size, ss).mean(axis=(1, 3))
    if k.sum() == 0:
        k[r, r] = 1
    return (k / k.sum()).astype(np.float32)


def line_kernel(length: float, angle: float) -> np.ndarray:
    """Straight motion streak (panning, subject motion)."""
    size = _odd(length + 3)
    canvas = np.zeros((size * 8, size * 8), np.float32)
    c = size * 4
    dx, dy = math.cos(angle) * length * 4, math.sin(angle) * length * 4
    p0 = (int(round(c - dx)), int(round(c - dy)))
    p1 = (int(round(c + dx)), int(round(c + dy)))
    cv2.line(canvas, p0, p1, 1.0, thickness=6, lineType=cv2.LINE_AA)
    k = cv2.resize(canvas, (size, size), interpolation=cv2.INTER_AREA)
    return k / k.sum()


def streak_kernel(length: float, angle: float) -> np.ndarray:
    """A long, straight motion streak (a panned background, or a car crossing a still frame), on a tight bounding box
    so streaks of hundreds of pixels stay cheap to build and apply."""
    ss = 4
    c, s = math.cos(angle), math.sin(angle)
    w, h = _odd(abs(c) * length + 3), _odd(abs(s) * length + 3)
    canvas = np.zeros((h * ss, w * ss), np.float32)
    cx, cy = w * ss / 2, h * ss / 2
    dx, dy = c * length * ss / 2, s * length * ss / 2
    p0 = (int(round(cx - dx)), int(round(cy - dy)))
    p1 = (int(round(cx + dx)), int(round(cy + dy)))
    cv2.line(canvas, p0, p1, 1.0, thickness=ss, lineType=cv2.LINE_AA)
    k = cv2.resize(canvas, (w, h), interpolation=cv2.INTER_AREA)
    if k.sum() <= 0:
        k[h // 2, w // 2] = 1
    return (k / k.sum()).astype(np.float32)


def shake_kernel(rng: np.random.Generator, target_sigma: float) -> np.ndarray:
    """Hand-shake: a random smooth trajectory with uneven dwell time, scaled to target_sigma."""
    n = 64
    vel = np.zeros((n, 2))
    v = rng.normal(size=2)
    for i in range(n):
        v = 0.8 * v + rng.normal(scale=0.6, size=2)
        if rng.random() < 0.05:  # sudden jerk
            v += rng.normal(scale=2.5, size=2)
        vel[i] = v
    traj = np.cumsum(vel, axis=0)
    traj -= traj.mean(axis=0)
    dwell = rng.gamma(0.7, size=n) + 0.05
    # scale so the covariance's major axis matches the target
    w = dwell / dwell.sum()
    cov = np.cov(traj.T, aweights=w)
    lam = np.linalg.eigvalsh(cov).max()
    traj *= target_sigma / max(math.sqrt(lam), 1e-6)
    ext = np.abs(traj).max()
    size = _odd(2 * ext + 5)
    ss = 4
    canvas = np.zeros((size * ss, size * ss), np.float32)
    pts = ((traj + size / 2) * ss).astype(np.float64)
    for i in range(n - 1):
        cv2.line(
            canvas,
            (int(pts[i, 0]), int(pts[i, 1])),
            (int(pts[i + 1, 0]), int(pts[i + 1, 1])),
            float(w[i]),
            thickness=max(1, ss // 2),
            lineType=cv2.LINE_AA,
        )
    k = cv2.resize(canvas, (size, size), interpolation=cv2.INTER_AREA)
    if k.sum() <= 0:
        k[size // 2, size // 2] = 1
    return k / k.sum()


def random_kernel(rng: np.random.Generator, sigma: float) -> tuple[np.ndarray, str]:
    """A kernel of the given equivalent sigma, of a random physical kind."""
    if sigma < 0.05:
        return np.ones((1, 1), np.float32), "none"
    kind = rng.choice(["gauss", "disc", "line", "shake"], p=[0.25, 0.32, 0.18, 0.25])
    if kind == "gauss":
        k = gaussian_kernel(sigma, aniso=float(rng.uniform(0.5, 1.0)), angle=float(rng.uniform(0, math.pi)))
    elif kind == "disc":
        k = disc_kernel(2.0 * sigma, cat_eye=float(rng.choice([0.0, rng.uniform(0.2, 0.6)], p=[0.8, 0.2])))
        if sigma > 1 and rng.random() < 0.5:  # real lenses are not perfect discs
            k = cv2.GaussianBlur(k, (0, 0), 0.15 * sigma)
            k /= k.sum()
    elif kind == "line":
        k = line_kernel(sigma * math.sqrt(12.0), float(rng.uniform(0, math.pi)))
    else:
        k = shake_kernel(rng, sigma)
    # correct small discretisation errors so the label is exact
    es = equivalent_sigma(k)
    return k, str(kind) if es > 0 else "none"
