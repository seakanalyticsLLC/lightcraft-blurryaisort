//! Blur Cull: how blurred a photo's *subject* is, on a 0–100 scale (0 = tack sharp, 100 = unrecognisable).
//!
//! A small learned network, BlurNet ([`net`]), turns the photo's luminance at the analysis size (long edge 1600 by
//! default) into a blur map: for every 16 × 16-pixel cell, how blurred that spot is and whether it has enough detail
//! to judge. [`subject`] then finds the subject (the camera's focus point, else faces, else the sharpest part of the
//! frame) and reads its blur off the map.
//!
//! **The model is optional and not part of LightCraft.** It is trained on the user's own photos with the Python
//! tools in `tools/blurcull` (see its README) and installed as one `.safetensors` file. [`Model::load`] checks the
//! file thoroughly (format, architecture, every tensor's shape) and never panics on a bad one.
//!
//! The score scale is shared with the training tools: `score = 100 · log2(1 + σ) / log2(1 + SIGMA_MAX)`, σ being the
//! blur's equivalent Gaussian sigma in analysis pixels.

#![forbid(unsafe_code)]
#![deny(clippy::unwrap_used, clippy::expect_used, clippy::panic, clippy::unimplemented, clippy::todo, clippy::unreachable)]

pub mod subject;
pub mod weights;

#[cfg(not(target_arch = "wasm32"))]
mod net;

use std::path::Path;

pub use subject::{BlurMap, Rect, Subject, SubjectScore};

/// What a model file declares itself as (`format` in the safetensors metadata).
pub const FORMAT: &str = "lightcraft-blurnet";
/// The architecture this build runs (`arch` in the metadata); `tools/blurcull/net.py` is its Python twin.
pub const ARCH: &str = "blurnet-v1";
/// Where the model lives inside the models folder.
pub const MODEL_FILE: &str = "blurnet.safetensors";
/// Largest model file accepted (BlurNet v1 is ~600 KB).
pub const MAX_MODEL_BYTES: u64 = 64 << 20;
/// Default analysis long edge (a model file can name another).
pub const ANALYSIS_LONG_EDGE: usize = 1600;
/// Cell size of the blur map (the network's output stride).
pub const STRIDE: usize = 16;
/// Blur sigma (analysis pixels) that maps to a score of 100.
pub const SIGMA_MAX: f64 = 24.0;

/// Score 0..100 for a blur sigma in analysis pixels.
pub fn sigma_to_score(sigma: f64) -> f64 {
    let s = (1.0 + sigma.max(0.0)).log2() / (1.0 + SIGMA_MAX).log2();
    (100.0 * s).clamp(0.0, 100.0)
}

/// The network's blur unit, `log2(1 + σ)`, as a score 0..100.
pub fn log2_to_score(b: f64) -> f64 {
    (100.0 * b.max(0.0) / (1.0 + SIGMA_MAX).log2()).clamp(0.0, 100.0)
}

#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("{0}")]
    Invalid(String),
    #[error("{0}")]
    Io(#[from] std::io::Error),
    #[error("Blur Cull isn't available in this build")]
    Unsupported,
    #[cfg(not(target_arch = "wasm32"))]
    #[error("the network failed: {0}")]
    Candle(#[from] candle_core::Error),
}

pub type Result<T> = std::result::Result<T, Error>;

/// What a model file says about itself.
#[derive(Clone, Debug, Default, PartialEq, serde::Serialize)]
pub struct ModelInfo {
    pub arch: String,
    pub analysis_long_edge: usize,
    pub created: String,
    pub note: String,
    /// Training steps (0 = untrained: a plumbing test file).
    pub step: u64,
    /// First 16 hex digits of the file's SHA-256: scores remember which model made them.
    pub id: String,
    pub bytes: u64,
}

/// A loaded BlurNet.
pub struct Model {
    pub info: ModelInfo,
    #[cfg(not(target_arch = "wasm32"))]
    net: net::Net,
}

impl Model {
    /// Read and check a model file (size, format, architecture, every tensor) and build the network.
    pub fn load(path: &Path) -> Result<Model> {
        let bytes = weights::read_capped(path, MAX_MODEL_BYTES)?;
        Self::from_bytes(&bytes)
    }

    pub fn from_bytes(bytes: &[u8]) -> Result<Model> {
        let file = weights::SafeTensors::parse(bytes)?;
        let info = model_info(&file, bytes)?;
        #[cfg(not(target_arch = "wasm32"))]
        {
            let net = net::Net::new(&file)?;
            Ok(Model { info, net })
        }
        #[cfg(target_arch = "wasm32")]
        {
            let _ = info;
            Err(Error::Unsupported)
        }
    }

    /// Analysis size for a source of `w × h`: the long edge brought down to the model's (never up).
    pub fn analysis_size(&self, w: usize, h: usize) -> (usize, usize) {
        fit(w, h, self.info.analysis_long_edge)
    }

    /// The blur map of an 8-bit RGB image (3 bytes per pixel, rows top to bottom), already at the analysis size
    /// (see [`Model::analysis_size`]; a larger image is analysed as it is, which makes blur read smaller).
    pub fn blur_map(&self, rgb: &[u8], w: usize, h: usize) -> Result<BlurMap> {
        let need = w.checked_mul(h).and_then(|n| n.checked_mul(3)).ok_or_else(|| Error::Invalid("image too large".into()))?;
        if w < STRIDE || h < STRIDE || rgb.len() < need {
            return Err(Error::Invalid(format!("image {w} × {h} is too small or its pixels are missing")));
        }
        // network input pixels are capped: 64 MP is far beyond any analysis size
        if w.saturating_mul(h) > 64 << 20 {
            return Err(Error::Invalid(format!("image {w} × {h} is too large to analyse")));
        }
        let y = luma(rgb, w, h);
        #[cfg(not(target_arch = "wasm32"))]
        {
            self.net.forward(&y, w, h)
        }
        #[cfg(target_arch = "wasm32")]
        {
            let _ = y;
            Err(Error::Unsupported)
        }
    }
}

/// An *untrained* BlurNet file with deterministic weights: for tests and for checking the plumbing (install, scan,
/// apply) without a real model. Its scores mean nothing. `arch` is normally [`ARCH`].
pub fn untrained_model_bytes(arch: &str) -> Vec<u8> {
    let mut k = 0x2545_f491u32;
    let mut val = move || {
        k = k.wrapping_mul(1_103_515_245).wrapping_add(12_345);
        ((k >> 16) as f32 / 65_536.0 - 0.5) * 0.4
    };
    let layers = [(1, 12), (12, 16), (16, 32), (32, 32), (32, 48), (48, 48), (48, 64), (64, 64), (64, 64)];
    let mut owned: Vec<(String, Vec<usize>, Vec<f32>)> = Vec::new();
    for (i, (cin, cout)) in layers.iter().copied().enumerate() {
        owned.push((format!("l{i}.weight"), vec![cout, cin, 3, 3], (0..cout * cin * 9).map(|_| val()).collect()));
        owned.push((format!("l{i}.bias"), vec![cout], (0..cout).map(|_| val()).collect()));
    }
    owned.push(("head.weight".into(), vec![2, 64, 1, 1], (0..128).map(|_| val()).collect()));
    // every cell "judged" (confidence ≈ 1), so scans yield (meaningless) scores to exercise the whole path
    owned.push(("head.bias".into(), vec![2], vec![0.1, 6.0]));
    let t: Vec<(&str, &[usize], &[f32])> = owned.iter().map(|(n, s, v)| (n.as_str(), s.as_slice(), v.as_slice())).collect();
    weights::encode(
        &t,
        &[
            ("format", FORMAT),
            ("version", "1"),
            ("arch", arch),
            ("analysis_long_edge", "1600"),
            ("sigma_max", "24.0"),
            ("stride", "16"),
            ("note", "untrained: plumbing test only"),
        ],
    )
}

/// `w × h` scaled so the long edge is at most `long_edge` (aspect kept, at least 1 × 1).
pub fn fit(w: usize, h: usize, long_edge: usize) -> (usize, usize) {
    let m = w.max(h).max(1);
    if m <= long_edge {
        return (w.max(1), h.max(1));
    }
    let s = long_edge as f64 / m as f64;
    (((w as f64 * s).round() as usize).max(1), ((h as f64 * s).round() as usize).max(1))
}

/// Box-filter (area) downscale of 8-bit RGB so the long edge is at most `long_edge` — what OpenCV's `INTER_AREA`
/// does in the training tools. An image already small enough comes back unchanged.
pub fn downscale_rgb(rgb: &[u8], w: usize, h: usize, long_edge: usize) -> (Vec<u8>, usize, usize) {
    let (ow, oh) = fit(w, h, long_edge);
    if (ow, oh) == (w, h) || w == 0 || h == 0 {
        return (rgb.to_vec(), w, h);
    }
    let mut out = vec![0u8; ow * oh * 3];
    for oy in 0..oh {
        let (y0, y1) = (oy * h / oh, ((oy + 1) * h / oh).max(oy * h / oh + 1).min(h));
        for ox in 0..ow {
            let (x0, x1) = (ox * w / ow, ((ox + 1) * w / ow).max(ox * w / ow + 1).min(w));
            let mut acc = [0u32; 3];
            for y in y0..y1 {
                for x in x0..x1 {
                    let i = (y * w + x) * 3;
                    if let Some(p) = rgb.get(i..i + 3) {
                        acc[0] += u32::from(p[0]);
                        acc[1] += u32::from(p[1]);
                        acc[2] += u32::from(p[2]);
                    }
                }
            }
            let n = ((y1 - y0) * (x1 - x0)).max(1) as u32;
            let o = (oy * ow + ox) * 3;
            if let Some(d) = out.get_mut(o..o + 3) {
                for c in 0..3 {
                    d[c] = ((acc[c] + n / 2) / n) as u8;
                }
            }
        }
    }
    (out, ow, oh)
}

/// BT.709 luminance of sRGB-encoded 8-bit pixels, 0..1 (as `tools/blurcull/synth.py` `luma`).
pub fn luma(rgb: &[u8], w: usize, h: usize) -> Vec<f32> {
    rgb.chunks_exact(3)
        .take(w.saturating_mul(h))
        .map(|p| (0.2126 * f32::from(p[0]) + 0.7152 * f32::from(p[1]) + 0.0722 * f32::from(p[2])) / 255.0)
        .collect()
}

fn meta<'a>(file: &'a weights::SafeTensors, key: &str) -> &'a str {
    file.metadata.get(key).map(String::as_str).unwrap_or("")
}

fn model_info(file: &weights::SafeTensors, bytes: &[u8]) -> Result<ModelInfo> {
    if meta(file, "format") != FORMAT {
        return Err(Error::Invalid(format!("not a Blur Cull model (format `{}`, expected `{FORMAT}`)", meta(file, "format"))));
    }
    if meta(file, "version") != "1" {
        return Err(Error::Invalid(format!("model format version `{}` is not supported (this build reads 1)", meta(file, "version"))));
    }
    if meta(file, "arch") != ARCH {
        return Err(Error::Invalid(format!("model architecture `{}` is not supported (this build runs `{ARCH}`)", meta(file, "arch"))));
    }
    let analysis_long_edge = match meta(file, "analysis_long_edge") {
        "" => ANALYSIS_LONG_EDGE,
        v => v.parse::<usize>().ok().filter(|v| (256..=8192).contains(v)).ok_or_else(|| Error::Invalid(format!("bad analysis size `{v}`")))?,
    };
    if let Ok(s) = meta(file, "sigma_max").parse::<f64>()
        && (s - SIGMA_MAX).abs() > 1e-6
    {
        return Err(Error::Invalid(format!("model uses score scale σmax {s}; this build uses {SIGMA_MAX}")));
    }
    if let Ok(s) = meta(file, "stride").parse::<usize>()
        && s != STRIDE
    {
        return Err(Error::Invalid(format!("model stride {s} is not supported")));
    }
    use sha2::Digest;
    let digest = sha2::Sha256::digest(bytes);
    let id: String = digest.iter().take(8).map(|b| format!("{b:02x}")).collect();
    Ok(ModelInfo {
        arch: ARCH.into(),
        analysis_long_edge,
        created: meta(file, "created").chars().take(64).collect(),
        note: meta(file, "note").chars().take(400).collect(),
        step: meta(file, "step").parse().unwrap_or(0),
        id,
        bytes: bytes.len() as u64,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn score_scale_matches_the_training_tools() {
        // kernels.py: sigma_to_score(0) = 0, sigma_to_score(24) = 100, log scale between
        assert_eq!(sigma_to_score(0.0), 0.0);
        assert!((sigma_to_score(SIGMA_MAX) - 100.0).abs() < 1e-9);
        assert!((sigma_to_score(1.0) - 100.0 / 25f64.log2()).abs() < 1e-9);
        assert!((log2_to_score(1.0) - sigma_to_score(1.0)).abs() < 1e-9);
        assert_eq!(sigma_to_score(-3.0), 0.0);
        assert_eq!(sigma_to_score(1e9), 100.0);
    }

    #[test]
    fn fit_keeps_aspect_and_never_enlarges() {
        assert_eq!(fit(6000, 4000, 1600), (1600, 1067));
        assert_eq!(fit(4000, 6000, 1600), (1067, 1600));
        assert_eq!(fit(800, 600, 1600), (800, 600));
        assert_eq!(fit(0, 0, 1600), (1, 1));
    }

    #[test]
    fn downscale_averages_and_fits() {
        let (w, h) = (40, 20);
        let rgb: Vec<u8> = (0..w * h).flat_map(|i| if (i % w) < 20 { [0, 0, 0] } else { [200, 100, 50] }).collect();
        let (out, ow, oh) = downscale_rgb(&rgb, w, h, 10);
        assert_eq!((ow, oh), (10, 5));
        assert_eq!(&out[..3], &[0, 0, 0]);
        assert_eq!(&out[out.len() - 3..], &[200, 100, 50]);
        let (same, sw, sh) = downscale_rgb(&rgb, w, h, 100);
        assert_eq!((sw, sh, same.len()), (w, h, rgb.len()));
        // a short buffer is tolerated (zeros), not a panic
        let _ = downscale_rgb(&rgb[..10], w, h, 10);
    }

    #[test]
    fn junk_is_an_error_not_a_crash() {
        for bytes in [&b""[..], b"\x01", b"\xff\xff\xff\xff\xff\xff\xff\xff{}", b"\x02\x00\x00\x00\x00\x00\x00\x00{}"] {
            assert!(Model::from_bytes(bytes).is_err());
        }
    }
}
