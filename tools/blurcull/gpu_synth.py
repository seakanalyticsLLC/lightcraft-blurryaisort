"""synth.py on the GPU: the same training examples, rendered with PyTorch so the graphics card does the heavy work.

What stays on the CPU, unchanged: every random choice (the same numpy calls in the same order as synth.py), the blur
kernels (kernels.py: tiny arrays, microseconds each) and the shapes drawn with OpenCV (focus masks, the car's blob,
night lamps). So the kinds of blur, their amounts and every label mean exactly what they mean in synth.py.

What moves to the GPU: the resize to analysis size, the blur itself (FFT correlation for big kernels: a 500 px pan
streak costs the same as a 5 px one), the mask feathering, the per-cell detail / crispness that set the labels'
confidence, highlight boosting, sensor noise, sharpening, JPEG (8 x 8 DCT with the standard tables, 4:2:0 chroma) and
cutting the crops.

check_gpu_synth.py compares this against synth.py: with sensor noise switched off on both sides, a render takes the
same random decisions, gives the same sigma labels and nearly the same confidence and pixels (JPEG is the only
approximation: the GPU's float DCT vs libjpeg's integer one).
"""
from __future__ import annotations

import math
import queue
import threading
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from kernels import GHOST_SIGMA_MAX, SIGMA_MAX, equivalent_sigma, ghost_kernel, random_kernel, streak_kernel
from net import LUMA
from synth import CELL, P_BOOST, P_NIGHT, P_PAN, _cells, pan_angle, sample_sigma

# ---------------------------------------------------------------------------------------------------------------
# small GPU building blocks (images are (C, H, W) float32 tensors, channels in OpenCV's BGR order)


def _sym_index(n: int, before: int, after: int, mode: str, device) -> torch.Tensor:
    """Source index for each padded position. mode 'reflect' = OpenCV BORDER_REFLECT (fedcba|abcd|dcba),
    'reflect101' = BORDER_REFLECT_101 (dcb|abcd|cba, OpenCV's default for blurs and Sobel)."""
    i = torch.arange(-before, n + after, device=device)
    if n == 1:
        return torch.zeros_like(i)
    if mode == "reflect":
        p = 2 * n
        m = torch.remainder(i, p)
        return torch.where(m >= n, p - 1 - m, m)
    p = 2 * n - 2
    m = torch.remainder(i, p)
    return torch.where(m >= n, p - m, m)


def pad2d(x: torch.Tensor, top: int, bottom: int, left: int, right: int, mode: str) -> torch.Tensor:
    h, w = x.shape[-2:]
    iy = _sym_index(h, top, bottom, mode, x.device)
    ix = _sym_index(w, left, right, mode, x.device)
    return x.index_select(-2, iy).index_select(-1, ix)


def _fast_len(n: int) -> int:
    """Smallest m >= n whose only prime factors are 2, 3, 5, 7 (quick FFT sizes)."""
    m = n
    while True:
        r = m
        for p in (2, 3, 5, 7):
            while r % p == 0:
                r //= p
        if r == 1:
            return m
        m += 1


def correlate(x: torch.Tensor, k: np.ndarray) -> torch.Tensor:
    """cv2.filter2D(x, -1, k, borderType=BORDER_REFLECT) for an odd-sized kernel: correlation, anchor at the centre."""
    kh, kw = k.shape
    if kh == 1 and kw == 1:
        return x * float(k[0, 0])
    c, h, w = x.shape
    xp = pad2d(x, kh // 2, kh // 2, kw // 2, kw // 2, "reflect")
    kt = torch.from_numpy(np.ascontiguousarray(k, dtype=np.float32)).to(x.device)
    if kh * kw <= 15 * 15:
        return F.conv2d(xp[:, None], kt[None, None])[:, 0]
    # circular correlation over the padded size gives the exact valid part: no index wraps for rows < h, cols < w
    nh, nw = _fast_len(xp.shape[-2]), _fast_len(xp.shape[-1])
    X = torch.fft.rfft2(xp, s=(nh, nw))
    K = torch.fft.rfft2(kt, s=(nh, nw))
    out = torch.fft.irfft2(X * torch.conj(K), s=(nh, nw))
    return out[:, :h, :w].contiguous()


_GAUSS_CACHE: dict = {}


def _gauss_1d(sigma: float, device) -> torch.Tensor:
    ks = int(np.rint(sigma * 4 * 2 + 1)) | 1  # OpenCV's size for float images: cvRound(sigma*8+1)|1
    key = (ks, round(sigma, 9), str(device))
    g = _GAUSS_CACHE.get(key)
    if g is None:
        g = torch.from_numpy(cv2.getGaussianKernel(ks, sigma, cv2.CV_32F).ravel().astype(np.float32)).to(device)
        if len(_GAUSS_CACHE) > 256:
            _GAUSS_CACHE.clear()
        _GAUSS_CACHE[key] = g
    return g


def gauss_blur(x: torch.Tensor, sigma: float) -> torch.Tensor:
    """cv2.GaussianBlur(x, (0, 0), sigma) for float images (BORDER_REFLECT_101). x: (C, H, W)."""
    g = _gauss_1d(sigma, x.device)
    r = g.numel() // 2
    xp = pad2d(x, r, r, r, r, "reflect101")[:, None]
    xp = F.conv2d(xp, g.view(1, 1, 1, -1))
    xp = F.conv2d(xp, g.view(1, 1, -1, 1))
    return xp[:, 0]


_SOBEL_X = torch.tensor([[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]])


def luma(bgr: torch.Tensor) -> torch.Tensor:
    """BT.709 luminance of a (3, H, W) BGR image -> (H, W)."""
    return LUMA[2] * bgr[0] + LUMA[1] * bgr[1] + LUMA[0] * bgr[2]


def cells_mean(a: torch.Tensor, hc: int, wc: int) -> torch.Tensor:
    return a[: hc * CELL, : wc * CELL].reshape(hc, CELL, wc, CELL).mean(dim=(1, 3))


def cell_detail(y: torch.Tensor) -> torch.Tensor:
    """synth.cell_detail: per-cell 75th percentile of the Sobel gradient -> confidence 0..1."""
    sx = _SOBEL_X.to(y.device)
    yp = pad2d(y[None], 1, 1, 1, 1, "reflect101")[:, None]
    gx = F.conv2d(yp, (sx / 8)[None, None])[0, 0]
    gy = F.conv2d(yp, (sx.T / 8)[None, None])[0, 0]
    g = torch.sqrt(gx * gx + gy * gy)
    hc, wc = y.shape[0] // CELL, y.shape[1] // CELL
    v = g[: hc * CELL, : wc * CELL].reshape(hc, CELL, wc, CELL).permute(0, 2, 1, 3).reshape(hc, wc, CELL * CELL)
    v = torch.sort(v, dim=-1).values
    pos = 0.75 * (CELL * CELL - 1)  # numpy's default (linear) percentile
    lo = int(math.floor(pos))
    p = v[..., lo] + (pos - lo) * (v[..., lo + 1] - v[..., lo])
    return torch.clamp((p - 0.004) / 0.012, 0, 1)


def cell_crispness(y: torch.Tensor) -> torch.Tensor:
    """synth.cell_crispness: gradient energy before / after an extra sigma-1 blur, per cell."""

    def g2(a):
        gx = (a[:, 1:] - a[:, :-1])[:-1, :]
        gy = (a[1:, :] - a[:-1, :])[:, :-1]
        return F.pad(gx * gx + gy * gy, (0, 1, 0, 1))

    hc, wc = y.shape[0] // CELL, y.shape[1] // CELL
    a = g2(y)[: hc * CELL, : wc * CELL].reshape(hc, CELL, wc, CELL).sum(dim=(1, 3))
    b = g2(gauss_blur(y[None], 1.0)[0])[: hc * CELL, : wc * CELL].reshape(hc, CELL, wc, CELL).sum(dim=(1, 3))
    return torch.sqrt(a / torch.clamp(b, min=1e-9))


def area_resize(img: torch.Tensor, d: float) -> torch.Tensor:
    """cv2.resize(img8, None, fx=d, fy=d, interpolation=INTER_AREA) for d <= 1, uint8 result as float 0..255.
    Exact area weights as two matrix products. img: (C, H, W) float 0..255."""
    c, h, w = img.shape
    oh, ow = int(round(h * d)), int(round(w * d))
    if oh == h and ow == w:
        return img

    def weights(n_in, n_out):  # built on the device: the scale is random every render, so nothing to cache
        scale = 1.0 / d
        lo = torch.arange(n_out, dtype=torch.float64, device=img.device) * scale
        hi = lo + scale
        src = torch.arange(n_in, dtype=torch.float64, device=img.device)
        ov = torch.clamp(torch.minimum(hi[:, None], src[None, :] + 1) - torch.maximum(lo[:, None], src[None, :]), min=0)
        return (ov / ov.sum(dim=1, keepdim=True)).float()

    wy, wx = weights(h, oh), weights(w, ow)
    out = torch.matmul(torch.matmul(wy, img), wx.T)
    return torch.round(out).clamp(0, 255)


def to_lin(x):
    return torch.pow(torch.clamp(x, 0, 1), 2.2)


def to_srgb(x):
    return torch.pow(torch.clamp(x, 0, 1), 1 / 2.2)


# ---------------------------------------------------------------------------------------------------------------
# JPEG round trip on the GPU (baseline JPEG: BT.601 YCbCr, 4:2:0 chroma, standard Annex K tables)

_Q_LUMA = np.array([
    16, 11, 10, 16, 24, 40, 51, 61, 12, 12, 14, 19, 26, 58, 60, 55, 14, 13, 16, 24, 40, 57, 69, 56,
    14, 17, 22, 29, 51, 87, 80, 62, 18, 22, 37, 56, 68, 109, 103, 77, 24, 35, 55, 64, 81, 104, 113, 92,
    49, 64, 78, 87, 103, 121, 120, 101, 72, 92, 95, 98, 112, 100, 103, 99], np.float32).reshape(8, 8)
_Q_CHROMA = np.full((8, 8), 99.0, np.float32)
_Q_CHROMA[:4, :4] = np.array([17, 18, 24, 47, 18, 21, 26, 66, 24, 26, 56, 99, 47, 66, 99, 99], np.float32).reshape(4, 4)


def _qtable(base: np.ndarray, quality: int) -> np.ndarray:
    q = min(max(int(quality), 1), 100)
    scale = 5000 // q if q < 50 else 200 - 2 * q
    return np.clip((base * scale + 50) // 100, 1, 255).astype(np.float32)


def _dct_matrix(device) -> torch.Tensor:
    n = 8
    m = np.zeros((n, n), np.float64)
    for u in range(n):
        a = math.sqrt(1 / n) if u == 0 else math.sqrt(2 / n)
        for x in range(n):
            m[u, x] = a * math.cos((2 * x + 1) * u * math.pi / (2 * n))
    return torch.from_numpy(m.astype(np.float32)).to(device)


def _block_quant(p: torch.Tensor, q: torch.Tensor, D: torch.Tensor) -> torch.Tensor:
    """(H, W) plane with H, W multiples of 8: DCT -> quantise -> dequantise -> inverse DCT, level-shifted by 128."""
    h, w = p.shape
    b = (p - 128.0).reshape(h // 8, 8, w // 8, 8).permute(0, 2, 1, 3)
    coef = D @ b @ D.T
    coef = torch.round(coef / q) * q
    b = D.T @ coef @ D
    return b.permute(0, 2, 1, 3).reshape(h, w) + 128.0


def jpeg_roundtrip(bgr8: torch.Tensor, quality: int) -> torch.Tensor:
    """cv2.imdecode(cv2.imencode('.jpg', img8, quality)) approximately. bgr8: (3, H, W) float 0..255 integer values."""
    dev = bgr8.device
    _, h, w = bgr8.shape
    B, G, R = bgr8[0], bgr8[1], bgr8[2]
    Y = torch.round(0.299 * R + 0.587 * G + 0.114 * B)
    Cb = torch.round(-0.168736 * R - 0.331264 * G + 0.5 * B + 128)
    Cr = torch.round(0.5 * R - 0.418688 * G - 0.081312 * B + 128)
    H16, W16 = -(-h // 16) * 16, -(-w // 16) * 16
    ycc = F.pad(torch.stack([Y, Cb, Cr])[None], (0, W16 - w, 0, H16 - h), mode="replicate")[0]
    D = _dct_matrix(dev)
    ql = torch.from_numpy(_qtable(_Q_LUMA, quality)).to(dev)
    qc = torch.from_numpy(_qtable(_Q_CHROMA, quality)).to(dev)
    Yq = torch.clamp(torch.round(_block_quant(ycc[0], ql, D)), 0, 255)
    sub = torch.round(F.avg_pool2d(ycc[1:][None], 2)[0])  # 2 x 2 chroma
    Cq = torch.stack([torch.clamp(torch.round(_block_quant(sub[i], qc, D)), 0, 255) for i in range(2)])
    Cu = F.interpolate(Cq[None], scale_factor=2, mode="bilinear", align_corners=False)[0]  # libjpeg's "fancy" 3/4, 1/4
    Y, Cb, Cr = Yq[:h, :w], Cu[0, :h, :w] - 128, Cu[1, :h, :w] - 128
    R = Y + 1.402 * Cr
    G = Y - 0.344136 * Cb - 0.714136 * Cr
    B = Y + 1.772 * Cb
    return torch.clamp(torch.round(torch.stack([B, G, R])), 0, 255)


# ---------------------------------------------------------------------------------------------------------------
# the synth.py pipeline, step for step


class GpuSynth:
    """Renders synth.examples() on a torch device. `noise=False` switches the sensor noise off (for check_gpu_synth)."""

    def __init__(self, device, seed: int = 0, noise: bool = True):
        self.dev = torch.device(device)
        self.gen = torch.Generator(device=self.dev)
        self.gen.manual_seed(int(seed))
        self.noise = noise
        se = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15)).astype(np.float32)
        self.se = torch.from_numpy(se).to(self.dev)[None, None]
        self.se_sum = float(se.sum())

    # -- drawing on the CPU (OpenCV, as in synth.py), feathering on the GPU

    def _upload(self, a: np.ndarray) -> torch.Tensor:
        return torch.from_numpy(np.ascontiguousarray(a)).to(self.dev, non_blocking=True)

    def depth_mask(self, rng, h, w):
        m = np.zeros((h, w), np.float32)
        for _ in range(rng.integers(1, 4)):
            cy, cx = rng.uniform(0.1, 0.9) * h, rng.uniform(0.1, 0.9) * w
            ay, ax = rng.uniform(0.08, 0.45) * h, rng.uniform(0.08, 0.45) * w
            cv2.ellipse(m, (int(cx), int(cy)), (max(int(ax), 1), max(int(ay), 1)), rng.uniform(0, 180), 0, 360, 1.0, -1)
        if rng.random() < 0.3:
            y0 = int(rng.uniform(0.2, 0.7) * h)
            m[y0 : y0 + int(rng.uniform(0.1, 0.3) * h)] = 1
        if rng.random() < 0.5:
            m = 1 - m
        return gauss_blur(self._upload(m)[None], rng.uniform(2, 12))[0]

    def subject_mask(self, rng, h, w):
        m = np.zeros((h, w), np.float32)
        for _ in range(1 if rng.random() < 0.8 else 2):
            ax = rng.uniform(0.08, 0.35) * w
            ay = min(ax / rng.uniform(1.3, 3.5), 0.45 * h)
            cx, cy = rng.uniform(0.15, 0.85) * w, rng.uniform(0.2, 0.8) * h
            cv2.ellipse(m, (int(cx), int(cy)), (max(int(ax), 1), max(int(ay), 1)), float(rng.normal(0, 8)), 0, 360, 1.0, -1)
        return gauss_blur(self._upload(m)[None], rng.uniform(1.5, 5.0))[0]

    # -- the stages

    def boost_highlights(self, rng, lin, gain=None, small_only=False):
        hi = lin.max(dim=0).values > 0.85
        if small_only and bool(hi.any()):
            r = 7
            f = hi.float()[None, None]
            ero = F.conv2d(F.pad(f, (r, r, r, r), value=1.0), self.se) >= self.se_sum - 0.5  # erode: border counts as set
            opened = F.conv2d(F.pad(ero.float(), (r, r, r, r), value=0.0), self.se) > 0.5  # dilate
            grown = F.conv2d(F.pad(opened.float(), (r, r, r, r), value=0.0), self.se) > 0.5
            hi = hi & ~grown[0, 0]
        if not bool(hi.any()):
            return lin
        g = float(2.0 ** rng.uniform(1.0, 5.0)) if gain is None else gain
        return torch.where(hi[None], lin * g, lin)

    def night(self, rng, lin, conf, hc, wc):
        _, h, w = lin.shape
        dark = float(2.0 ** rng.uniform(math.log2(0.008), math.log2(0.12)))
        out = self.boost_highlights(rng, lin, float(2.0 ** rng.uniform(1.0, 5.0)) / dark, small_only=True) * dark
        lamps = np.zeros((h, w, 3), np.float32)
        cover = np.zeros((h, w), np.float32)
        for _ in range(int(rng.integers(5, 60))):
            x, y = int(rng.uniform(0, w)), int(rng.uniform(0, h))
            v = float(2.0 ** rng.uniform(0.5, 5.5))
            tint = np.array(rng.choice([(1.0, 1.0, 1.0), (0.7, 0.85, 1.0), (0.4, 0.7, 1.0), (0.3, 0.3, 1.0), (1.0, 0.6, 0.3)]), np.float32)
            col = tuple(float(c * v) for c in tint)
            if rng.random() < 0.6:
                r = int(rng.integers(1, 4))
                cv2.circle(lamps, (x, y), r, col, -1, cv2.LINE_AA)
                cv2.circle(cover, (x, y), r, 1.0, -1)
            else:
                L, t = int(rng.integers(4, 40)), int(rng.integers(1, 4))
                cv2.line(lamps, (x, y), (x + L, y), col, t, cv2.LINE_AA)
                cv2.line(cover, (x, y), (x + L, y), 1.0, t)
        out = out + self._upload(lamps).permute(2, 0, 1)
        seen = cell_detail(luma(to_srgb(torch.clamp(out, 0, 1))))
        conf = torch.minimum(conf, seen)
        lit = self._upload(_cells(cover, hc, wc).mean(axis=(1, 3)) > 0.01)
        conf = torch.where(lit, torch.clamp(conf, min=0.8), conf)
        return out, conf

    def _streak(self, rng, lin, lo, hi, angle):
        k = streak_kernel(2.0 ** rng.uniform(math.log2(lo), math.log2(hi)), angle)
        return correlate(lin, k), min(equivalent_sigma(k), SIGMA_MAX)

    def pan_shot(self, rng, lin, conf, hc, wc):
        _, h, w = lin.shape
        angle = pan_angle(rng)
        r = rng.random()
        if r < 0.55:
            kind = "pan"
            bg, s_bg = self._streak(rng, lin, 25, 500, angle)
            q = rng.random()
            if q < 0.40:
                subj, s_subj = lin, 0.0
            elif q < 0.70:
                a = angle + float(rng.normal(0, 0.05))
                subj, s_subj = self._streak(rng, lin, 2, 40, a)
            elif q < 0.88:
                ts = float(2.0 ** rng.uniform(math.log2(0.6), math.log2(GHOST_SIGMA_MAX)))
                a = angle + float(rng.normal(0, 0.1))
                k = ghost_kernel(rng, ts, a)
                subj, s_subj = correlate(lin, k), min(equivalent_sigma(k), SIGMA_MAX)
            else:
                k, _ = random_kernel(rng, sample_sigma(rng))
                subj = correlate(lin, k)
                s_subj = min(equivalent_sigma(k), SIGMA_MAX) if k.size > 1 else 0.0
        elif r < 0.80:
            kind = "pan-missed"
            bg, s_bg = self._streak(rng, lin, 25, 500, angle)
            a = angle + float(rng.normal(0, 0.15))
            subj, s_subj = self._streak(rng, lin, 15, 300, a)
        else:
            kind = "moving-car"
            if rng.random() < 0.7:
                bg, s_bg = lin, 0.0
            else:
                k, _ = random_kernel(rng, min(sample_sigma(rng), 2.0))
                bg = correlate(lin, k)
                s_bg = min(equivalent_sigma(k), SIGMA_MAX) if k.size > 1 else 0.0
            subj, s_subj = self._streak(rng, lin, 10, 400, angle)
        m = self.subject_mask(rng, h, w)
        out = m[None] * subj + (1 - m[None]) * bg
        mc = torch.clamp(cells_mean(m, hc, wc), 0.0, 1.0)
        sig = torch.clamp(mc * s_subj + (1 - mc) * s_bg, 0.0, SIGMA_MAX)
        conf = torch.where((mc > 0.15) & (mc < 0.85), conf * 0.3, conf)
        return out, sig, conf, kind

    def camera(self, rng, lin, night=False):
        img = to_srgb(lin)
        if night or rng.random() < 0.65:
            n = rng.uniform(0.01, 0.08) if night else rng.uniform(0.003, 0.04)
            if self.noise:
                _, h, w = img.shape
                shot = torch.randn((1, h, w), generator=self.gen, device=self.dev)
                chroma = torch.randn((3, h, w), generator=self.gen, device=self.dev) * 0.4
                img = img + n * torch.sqrt(torch.clamp(img, 0.02, 1)) * (shot + chroma)
        if rng.random() < 0.45:
            bl = gauss_blur(img, rng.uniform(0.7, 1.6))
            img = img + rng.uniform(0.2, 1.2) * (img - bl)
        img8 = torch.floor(torch.clamp(img, 0, 1) * 255 + 0.5)
        if rng.random() < 0.7:
            q = int(rng.integers(60, 98))
            img8 = jpeg_roundtrip(img8, q)
        return img8

    def render(self, rng, bgr8: torch.Tensor, analysis_long_edge: int, crop: int):
        """One blurred version of a sharp photo (synth.examples before the crops). bgr8: (3, H, W) uint8 on the device.
        Returns (y8 (H, W) float 0..255, sigma cells, conf cells, kind) or None if the photo is too small."""
        _, h0, w0 = bgr8.shape
        long_edge = max(h0, w0)
        target = analysis_long_edge * rng.uniform(0.75, 1.25)
        d = min(1.0, target / long_edge)
        base = area_resize(bgr8.float(), d) / 255
        if rng.random() < 0.5:
            base = torch.flip(base, dims=(2,))
        _, h, w = base.shape
        if h < crop or w < crop:
            return None
        y_sharp = luma(base)
        conf = cell_detail(y_sharp)
        conf = torch.where((conf > 0.3) & (cell_crispness(y_sharp) < 1.2), torch.zeros_like(conf), conf)
        hc, wc = conf.shape

        lin = to_lin(base) * rng.uniform(0.6, 1.4)
        is_night = rng.random() < P_NIGHT
        if is_night:
            lin, conf = self.night(rng, lin, conf, hc, wc)
        elif rng.random() < P_BOOST:
            lin = self.boost_highlights(rng, lin)
        if rng.random() < P_PAN:
            blurred, sig, conf, kind = self.pan_shot(rng, lin, conf, hc, wc)
            s = 0.0
        else:
            k, kind = random_kernel(rng, sample_sigma(rng))
            if kind == "none":
                blurred, s = lin, 0.0
            else:
                blurred, s = correlate(lin, k), equivalent_sigma(k)
            sig = torch.full((hc, wc), float(s), device=self.dev)
        if s > 0.5 and rng.random() < 0.22:
            m = self.depth_mask(rng, h, w)
            blurred = m[None] * lin + (1 - m[None]) * blurred
            mc = cells_mean(m, hc, wc)
            sig = torch.clamp((1 - mc) * s, min=0)
            conf = torch.where((mc > 0.15) & (mc < 0.85), conf * 0.3, conf)
            kind += "+depth"
        if is_night:
            kind += "+night"
        y8 = torch.floor(luma(self.camera(rng, blurred, is_night)) + 0.5)
        return y8, sig, conf, kind

    def crops(self, rng, rendered, crop: int, n_crops: int):
        """synth.examples' crop loop: up to n_crops (y 0..1 (1, crop, crop), log2 blur (1, n, n), conf (1, n, n))."""
        y8, sig, conf, _ = rendered
        h, w = y8.shape
        conf_np = conf.detach().cpu().numpy()
        nc = crop // CELL
        ys, bs, cs = [], [], []
        for _ in range(n_crops):
            for _try in range(6):
                cy = int(rng.integers(0, (h - crop) // CELL + 1)) * CELL
                cx = int(rng.integers(0, (w - crop) // CELL + 1)) * CELL
                cc = conf_np[cy // CELL : cy // CELL + nc, cx // CELL : cx // CELL + nc]
                if cc.shape == (nc, nc) and (cc.mean() > 0.25 or rng.random() < 0.12):
                    break
            else:
                continue
            gy, gx = cy // CELL, cx // CELL
            ys.append(y8[cy : cy + crop, cx : cx + crop])
            bs.append(sig[gy : gy + nc, gx : gx + nc])
            cs.append(conf[gy : gy + nc, gx : gx + nc])
        if not ys:
            return None
        y = torch.stack(ys)[:, None] / 255.0
        b = torch.log2(1.0 + torch.clamp(torch.stack(bs), min=0))[:, None]
        c = torch.stack(cs)[:, None]
        return y, b, c


# ---------------------------------------------------------------------------------------------------------------
# feeding the training loop


class PhotoPool:
    """Sharp photos decoded by background threads (OpenCV releases the GIL), kept in a pool and each used for a few
    renders before it is swapped for a fresh one: the CPU only decodes a photo every few renders."""

    def __init__(self, data: Path, files: list[str], seed: int, pool: int = 24, uses: int = 4, threads: int = 4):
        self.data, self.files, self.uses = data, files, uses
        self.q: queue.Queue = queue.Queue(maxsize=pool)
        self.slots: list[list] = []
        self.pool = pool
        self.rng = np.random.default_rng(seed)
        self.stop = False
        for t in range(threads):
            threading.Thread(target=self._loader, args=(seed * 7919 + t,), daemon=True).start()

    def _loader(self, seed):
        rng = np.random.default_rng(seed)
        while not self.stop:
            f = self.files[int(rng.integers(len(self.files)))]
            img = cv2.imread(str(self.data / f), cv2.IMREAD_COLOR)
            if img is not None:
                self.q.put(img)

    def get(self) -> np.ndarray:
        while len(self.slots) < self.pool:
            self.slots.append([self.q.get(), self.uses])
        i = int(self.rng.integers(len(self.slots)))
        slot = self.slots[i]
        slot[1] -= 1
        img = slot[0]
        if slot[1] <= 0:
            try:
                self.slots[i] = [self.q.get_nowait(), self.uses]
            except queue.Empty:
                self.slots.pop(i)
        return img


class GpuBatches:
    """Endless (x, b, conf) batches rendered on the device, like DataLoader(Synthetic(...)) in train.py."""

    def __init__(self, data: Path, files: list[str], analysis: int, crop: int, batch: int, device, crops_per: int,
                 seed: int, pool: int = 24, uses: int = 4, threads: int = 4):
        self.synth = GpuSynth(device, seed=seed)
        self.photos = PhotoPool(data, files, seed, pool=pool, uses=uses, threads=threads)
        self.rng = np.random.default_rng(seed)
        self.analysis, self.crop, self.batch, self.crops_per = analysis, crop, batch, crops_per
        self.dev = torch.device(device)
        self.buf: list[tuple] = []
        self.n_buf = 0

    def __iter__(self):
        return self

    def __next__(self):
        while self.n_buf < self.batch:
            img = torch.from_numpy(self.photos.get()).to(self.dev, non_blocking=True).permute(2, 0, 1)
            r = self.synth.render(self.rng, img, self.analysis, self.crop)
            if r is None:
                continue
            got = self.synth.crops(self.rng, r, self.crop, self.crops_per)
            if got is None:
                continue
            self.buf.append(got)
            self.n_buf += got[0].shape[0]
        x, b, c = (torch.cat([t[i] for t in self.buf]) for i in range(3))
        out = (x[: self.batch], b[: self.batch], c[: self.batch])
        rest = (x[self.batch :], b[self.batch :], c[self.batch :])
        self.buf = [rest] if rest[0].shape[0] else []
        self.n_buf = rest[0].shape[0]
        return out
