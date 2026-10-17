//! BlurNet v1 on candle (CPU). Mirrors `tools/blurcull/net.py`: nine 3 × 3 convolutions with ReLU (BatchNorm already
//! folded into them by `export.py`) and a 1 × 1 head; output stride 16.

use candle_core::{DType, Device, Tensor};

use crate::subject::BlurMap;
use crate::weights::SafeTensors;
use crate::{Error, Result};

/// (in, out, stride, dilation) of each 3 × 3 convolution; padding = dilation. Same table as `net.py` `ARCH`.
pub(crate) const LAYERS: [(usize, usize, usize, usize); 9] =
    [(1, 12, 1, 1), (12, 16, 2, 1), (16, 32, 2, 1), (32, 32, 1, 1), (32, 48, 2, 1), (48, 48, 1, 1), (48, 64, 2, 1), (64, 64, 1, 2), (64, 64, 1, 4)];

struct Conv {
    w: Tensor,
    b: Tensor,
    stride: usize,
    dil: usize,
}

pub(crate) struct Net {
    convs: Vec<Conv>,
    head_w: Tensor,
    head_b: Tensor,
    device: Device,
}

fn tensor(file: &SafeTensors, name: &str, shape: &[usize], device: &Device) -> Result<Tensor> {
    let t = file.get(name, shape)?;
    Ok(Tensor::from_vec(t.data.clone(), shape, device)?)
}

impl Net {
    pub(crate) fn new(file: &SafeTensors) -> Result<Net> {
        let device = Device::Cpu;
        let mut convs = Vec::with_capacity(LAYERS.len());
        for (i, (cin, cout, stride, dil)) in LAYERS.iter().copied().enumerate() {
            convs.push(Conv {
                w: tensor(file, &format!("l{i}.weight"), &[cout, cin, 3, 3], &device)?,
                b: tensor(file, &format!("l{i}.bias"), &[cout], &device)?.reshape((1, cout, 1, 1))?,
                stride,
                dil,
            });
        }
        let last = LAYERS.last().map(|l| l.1).unwrap_or(64);
        let head_w = tensor(file, "head.weight", &[2, last, 1, 1], &device)?;
        let head_b = tensor(file, "head.bias", &[2], &device)?.reshape((1, 2, 1, 1))?;
        Ok(Net { convs, head_w, head_b, device })
    }

    /// `y`: luminance 0..1, `w × h`, row-major.
    pub(crate) fn forward(&self, y: &[f32], w: usize, h: usize) -> Result<BlurMap> {
        if y.len() != w * h {
            return Err(Error::Invalid("luminance buffer size".into()));
        }
        let mut x = (Tensor::from_slice(y, (1, 1, h, w), &self.device)? - 0.5)?;
        for c in &self.convs {
            x = x.conv2d(&c.w, c.dil, c.stride, c.dil, 1)?.broadcast_add(&c.b)?.relu()?;
        }
        let out = x.conv2d(&self.head_w, 0, 1, 1, 1)?.broadcast_add(&self.head_b)?;
        let (_, _, rows, cols) = out.dims4()?;
        let out = out.to_dtype(DType::F32)?.flatten_all()?.to_vec1::<f32>()?;
        let n = rows * cols;
        let blur = out.get(..n).ok_or_else(|| Error::Invalid("network output".into()))?.iter().map(|v| v.max(0.0)).collect();
        let conf = out.get(n..2 * n).ok_or_else(|| Error::Invalid("network output".into()))?.iter().map(|v| 1.0 / (1.0 + (-v).exp())).collect();
        Ok(BlurMap { cols, rows, blur, conf })
    }
}

#[cfg(test)]
mod tests {
    use crate::*;

    #[test]
    fn runs_and_gives_one_cell_per_16_pixels() {
        let m = Model::from_bytes(&untrained_model_bytes(ARCH)).unwrap();
        assert_eq!(m.info.analysis_long_edge, 1600);
        assert_eq!(m.info.id.len(), 16);
        let (w, h) = (100, 70);
        let rgb: Vec<u8> = (0..w * h * 3).map(|i| ((i * 37) % 251) as u8).collect();
        let map = m.blur_map(&rgb, w, h).unwrap();
        assert_eq!((map.cols, map.rows), (7, 5)); // ceil(100/16), ceil(70/16)
        assert_eq!(map.blur.len(), 35);
        assert!(map.blur.iter().all(|v| *v >= 0.0 && v.is_finite()));
        assert!(map.conf.iter().all(|v| (0.0..=1.0).contains(v)));
    }

    #[test]
    fn wrong_architecture_or_shapes_are_refused() {
        assert!(Model::from_bytes(&untrained_model_bytes("blurnet-v9")).is_err());
        let mut bytes = untrained_model_bytes(ARCH);
        // flip the declared format
        let s = String::from_utf8_lossy(&bytes).replace(FORMAT, "something-else");
        bytes = s.into_bytes();
        assert!(Model::from_bytes(&bytes).is_err());
        let m = Model::from_bytes(&untrained_model_bytes(ARCH)).unwrap();
        assert!(m.blur_map(&[0; 10], 10, 10).is_err());
        assert!(m.blur_map(&[0; 30], 100, 100).is_err());
    }
}
