//! The Rust network against PyTorch on the same photo and weights (`tools/blurcull/parity_reference.py` writes the
//! reference folder). Set LIGHTCRAFT_BLUR_PARITY to that folder.

use std::path::PathBuf;

use lightcraft_blur::Model;

#[test]
fn matches_pytorch() {
    let Some(dir) = std::env::var_os("LIGHTCRAFT_BLUR_PARITY").map(PathBuf::from) else {
        lightcraft_testkit::skip!(Model, "LIGHTCRAFT_BLUR_PARITY is not set (see tools/blurcull/parity_reference.py)");
        return;
    };
    let model = Model::load(&dir.join("model.safetensors")).unwrap();
    let rgb = std::fs::read(dir.join("input.rgb")).unwrap();
    let exp: serde_json::Value = serde_json::from_slice(&std::fs::read(dir.join("expected.json")).unwrap()).unwrap();
    let (w, h) = (exp["width"].as_u64().unwrap() as usize, exp["height"].as_u64().unwrap() as usize);
    let map = model.blur_map(&rgb, w, h).unwrap();
    assert_eq!((map.cols as u64, map.rows as u64), (exp["cols"].as_u64().unwrap(), exp["rows"].as_u64().unwrap()));
    let worst = |got: &[f32], key: &str| {
        got.iter().zip(exp[key].as_array().unwrap()).map(|(a, b)| (f64::from(*a) - b.as_f64().unwrap()).abs()).fold(0.0, f64::max)
    };
    let (db, dc) = (worst(&map.blur, "blur"), worst(&map.conf, "conf"));
    eprintln!("max difference: blur {db:.2e}, confidence {dc:.2e}");
    assert!(db < 1e-3 && dc < 1e-3, "blur {db}, conf {dc}");
    let s = map.subject_score(None, &[]);
    let want = exp["score"].as_f64();
    assert_eq!(s.score.is_some(), want.is_some());
    if let (Some(a), Some(b)) = (s.score, want) {
        assert!((a - b).abs() <= 0.11, "score {a} vs {b}"); // Rust rounds to 0.1
    }
}
