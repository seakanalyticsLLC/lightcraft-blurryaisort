//! A small, bounded safetensors reader: F32 tensors and string metadata. Every length and offset comes from the
//! file, so every one is checked; a malformed file is an error, never a panic or a huge allocation.

use std::collections::BTreeMap;
use std::io::Read;
use std::path::Path;

use serde_json::Value;

use crate::{Error, Result};

/// Header size cap (BlurNet's is ~1 KB).
const MAX_HEADER: u64 = 1 << 20;
/// Tensor count cap.
const MAX_TENSORS: usize = 4096;

pub struct Tensor {
    pub shape: Vec<usize>,
    pub data: Vec<f32>,
}

pub struct SafeTensors {
    pub tensors: BTreeMap<String, Tensor>,
    pub metadata: BTreeMap<String, String>,
}

/// Read a whole file of at most `max` bytes.
pub fn read_capped(path: &Path, max: u64) -> Result<Vec<u8>> {
    let len = std::fs::metadata(path)?.len();
    if len > max {
        return Err(Error::Invalid(format!("{} is {len} bytes; a Blur Cull model is at most {max}", path.display())));
    }
    let mut out = Vec::with_capacity(usize::try_from(len).unwrap_or(0));
    std::fs::File::open(path)?.take(max.saturating_add(1)).read_to_end(&mut out)?;
    if out.len() as u64 > max {
        return Err(Error::Invalid(format!("{} grew past {max} bytes while reading", path.display())));
    }
    Ok(out)
}

fn bad(msg: impl Into<String>) -> Error {
    Error::Invalid(format!("not a valid safetensors file: {}", msg.into()))
}

impl SafeTensors {
    pub fn parse(bytes: &[u8]) -> Result<SafeTensors> {
        let len_bytes: [u8; 8] = bytes.get(..8).and_then(|b| b.try_into().ok()).ok_or_else(|| bad("too short"))?;
        let n = u64::from_le_bytes(len_bytes);
        if n > MAX_HEADER {
            return Err(bad(format!("header of {n} bytes")));
        }
        let end = 8usize.checked_add(usize::try_from(n).map_err(|_| bad("header length"))?).ok_or_else(|| bad("header length"))?;
        let header = bytes.get(8..end).ok_or_else(|| bad("header runs past the end"))?;
        let data = bytes.get(end..).unwrap_or(&[]);
        let header: Value = serde_json::from_slice(header).map_err(|e| bad(format!("header: {e}")))?;
        let obj = header.as_object().ok_or_else(|| bad("header is not an object"))?;
        if obj.len() > MAX_TENSORS + 1 {
            return Err(bad("too many tensors"));
        }
        let mut tensors = BTreeMap::new();
        let mut metadata = BTreeMap::new();
        for (name, v) in obj {
            if name == "__metadata__" {
                if let Some(m) = v.as_object() {
                    for (k, v) in m.iter().take(256) {
                        if let Some(s) = v.as_str() {
                            metadata.insert(k.clone(), s.chars().take(4096).collect());
                        }
                    }
                }
                continue;
            }
            let dtype = v.get("dtype").and_then(Value::as_str).unwrap_or("");
            if dtype != "F32" {
                return Err(bad(format!("tensor `{name}` is {dtype}; only F32 is read")));
            }
            let shape: Vec<usize> = v
                .get("shape")
                .and_then(Value::as_array)
                .ok_or_else(|| bad(format!("tensor `{name}` has no shape")))?
                .iter()
                .map(|d| d.as_u64().and_then(|d| usize::try_from(d).ok()).ok_or_else(|| bad(format!("tensor `{name}` shape"))))
                .collect::<Result<_>>()?;
            if shape.len() > 8 {
                return Err(bad(format!("tensor `{name}` has {} dimensions", shape.len())));
            }
            let count = shape.iter().try_fold(1usize, |a, d| a.checked_mul(*d)).ok_or_else(|| bad(format!("tensor `{name}` is too large")))?;
            let offs = v.get("data_offsets").and_then(Value::as_array).ok_or_else(|| bad(format!("tensor `{name}` has no offsets")))?;
            let (Some(a), Some(b)) = (offs.first().and_then(Value::as_u64), offs.get(1).and_then(Value::as_u64)) else {
                return Err(bad(format!("tensor `{name}` offsets")));
            };
            let (a, b) = (usize::try_from(a).map_err(|_| bad("offset"))?, usize::try_from(b).map_err(|_| bad("offset"))?);
            if b < a || b - a != count.checked_mul(4).ok_or_else(|| bad("size"))? {
                return Err(bad(format!("tensor `{name}`: {} bytes for {count} values", b.saturating_sub(a))));
            }
            let raw = data.get(a..b).ok_or_else(|| bad(format!("tensor `{name}` runs past the end of the file")))?;
            let values: Vec<f32> = raw.chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect();
            if values.iter().any(|v| !v.is_finite()) {
                return Err(bad(format!("tensor `{name}` holds NaN or infinite values")));
            }
            tensors.insert(name.clone(), Tensor { shape, data: values });
        }
        Ok(SafeTensors { tensors, metadata })
    }

    /// The tensor `name`, which must have exactly `shape`.
    pub fn get(&self, name: &str, shape: &[usize]) -> Result<&Tensor> {
        let t = self.tensors.get(name).ok_or_else(|| Error::Invalid(format!("the model has no tensor `{name}`")))?;
        if t.shape != shape {
            return Err(Error::Invalid(format!("tensor `{name}` is {:?}, expected {shape:?}", t.shape)));
        }
        Ok(t)
    }
}

/// Encode F32 tensors and string metadata as a safetensors file (what `export.py` writes). Used to make test models.
pub fn encode(tensors: &[(&str, &[usize], &[f32])], meta: &[(&str, &str)]) -> Vec<u8> {
    let mut header = serde_json::Map::new();
    let mut data = Vec::new();
    for (name, shape, values) in tensors {
        let a = data.len();
        for v in *values {
            data.extend_from_slice(&v.to_le_bytes());
        }
        header.insert((*name).into(), serde_json::json!({"dtype": "F32", "shape": shape, "data_offsets": [a, data.len()]}));
    }
    let m: serde_json::Map<String, Value> = meta.iter().map(|(k, v)| ((*k).to_string(), Value::from(*v))).collect();
    header.insert("__metadata__".into(), Value::Object(m));
    let h = serde_json::to_vec(&Value::Object(header)).unwrap_or_default();
    let mut out = (h.len() as u64).to_le_bytes().to_vec();
    out.extend_from_slice(&h);
    out.extend_from_slice(&data);
    out
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;

    pub(crate) fn build(tensors: &[(&str, Vec<usize>, Vec<f32>)], meta: &[(&str, &str)]) -> Vec<u8> {
        let t: Vec<(&str, &[usize], &[f32])> = tensors.iter().map(|(n, s, v)| (*n, s.as_slice(), v.as_slice())).collect();
        encode(&t, meta)
    }

    #[test]
    fn reads_tensors_and_metadata() {
        let f = build(&[("a", vec![2, 2], vec![1.0, 2.0, 3.0, 4.0])], &[("format", "x")]);
        let st = SafeTensors::parse(&f).unwrap();
        assert_eq!(st.get("a", &[2, 2]).unwrap().data, vec![1.0, 2.0, 3.0, 4.0]);
        assert!(st.get("a", &[4]).is_err());
        assert!(st.get("b", &[1]).is_err());
        assert_eq!(st.metadata["format"], "x");
    }

    #[test]
    fn hostile_files_are_errors() {
        let good = build(&[("a", vec![2], vec![1.0, 2.0])], &[]);
        // truncated anywhere
        for n in 0..good.len() {
            let _ = SafeTensors::parse(&good[..n]);
        }
        // offsets past the end, wrong sizes, huge shapes, NaN, other dtypes
        let hdr = |h: &str| {
            let mut v = (h.len() as u64).to_le_bytes().to_vec();
            v.extend_from_slice(h.as_bytes());
            v.extend_from_slice(&[0u8; 8]);
            v
        };
        for h in [
            r#"{"a":{"dtype":"F32","shape":[2],"data_offsets":[0,800]}}"#,
            r#"{"a":{"dtype":"F32","shape":[3],"data_offsets":[0,8]}}"#,
            r#"{"a":{"dtype":"F32","shape":[18446744073709551615,2],"data_offsets":[0,8]}}"#,
            r#"{"a":{"dtype":"F16","shape":[4],"data_offsets":[0,8]}}"#,
            r#"{"a":{"dtype":"F32","shape":[2],"data_offsets":[8,0]}}"#,
            r#"[1,2]"#,
            r#"{"a":{"dtype":"F32","shape":[2],"data_offsets":[0]}}"#,
        ] {
            assert!(SafeTensors::parse(&hdr(h)).is_err(), "{h}");
        }
        let nan = build(&[("a", vec![1], vec![f32::NAN])], &[]);
        assert!(SafeTensors::parse(&nan).is_err());
    }
}
