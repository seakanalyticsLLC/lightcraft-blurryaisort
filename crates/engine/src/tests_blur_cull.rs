//! Blur Cull (`blur.*`): model install and checks, the background scan, scores beside the library, flags in one undo
//! step. The model here is untrained (`lightcraft_blur::untrained_model_bytes`): it exercises the plumbing, not the
//! quality of the scores.

use std::path::PathBuf;

use lightcraft_catalog::{Flag, PhotoId};
use serde_json::{Value, json};

use crate::{AiModel, ErrorKind, Session, blur_cull};

fn scratch(name: &str) -> PathBuf {
    let d = std::env::temp_dir().join(format!("lc-blur-{name}-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&d);
    std::fs::create_dir_all(&d).unwrap();
    d
}

/// A demo session whose model goes to `dir/models/blurnet.safetensors`, with the untrained model installed.
fn with_model(dir: &std::path::Path) -> Session {
    let mut s = Session::with_demo();
    s.set_blur_model_path(Some(dir.join("models").join("blurnet.safetensors")));
    let src = dir.join("export.safetensors");
    std::fs::write(&src, lightcraft_blur::untrained_model_bytes(lightcraft_blur::ARCH)).unwrap();
    let r = s.execute("blur.model.install", &json!({"path": src.to_string_lossy()})).unwrap();
    assert_eq!(r["untrained"], true, "{r}");
    s
}

#[test]
fn without_a_model_scanning_says_which_model_is_missing() {
    let mut s = Session::with_demo();
    let st = s.execute("blur.model.status", &json!({})).unwrap();
    assert_eq!(st["installed"], false);
    let e = s.execute("blur.scan", &json!({"scope": "view"})).unwrap_err();
    assert_eq!(e.kind(), ErrorKind::ModelMissing(AiModel::BlurCull), "{e}");
    // somewhere to keep it, but nothing there yet: the same
    let d = scratch("missing");
    s.set_blur_model_path(Some(d.join("blurnet.safetensors")));
    let e = s.execute("blur.scan", &json!({"scope": "view"})).unwrap_err();
    assert_eq!(e.kind(), ErrorKind::ModelMissing(AiModel::BlurCull), "{e}");
}

#[test]
fn install_refuses_files_that_are_not_blurnet() {
    let d = scratch("install");
    let mut s = Session::with_demo();
    s.set_blur_model_path(Some(d.join("models").join("blurnet.safetensors")));
    for (name, bytes) in [("junk.safetensors", b"not a model".to_vec()), ("other.safetensors", lightcraft_blur::untrained_model_bytes("blurnet-v99"))]
    {
        let p = d.join(name);
        std::fs::write(&p, bytes).unwrap();
        assert!(s.execute("blur.model.install", &json!({"path": p.to_string_lossy()})).is_err(), "{name}");
    }
    assert!(s.execute("blur.model.install", &json!({"path": d.join("nope.safetensors").to_string_lossy()})).is_err());
    assert!(!d.join("models").join("blurnet.safetensors").exists());
    // a good one installs, reports itself, and can be removed
    let mut s = with_model(&d);
    let st = s.execute("blur.model.status", &json!({})).unwrap();
    assert_eq!(st["installed"], true, "{st}");
    assert_eq!(st["model"]["analysis_long_edge"], 1600);
    assert_eq!(s.execute("blur.model.remove", &json!({})).unwrap()["removed"], true);
    assert_eq!(s.execute("blur.model.status", &json!({})).unwrap()["installed"], false);
}

#[test]
fn scanning_scores_every_photo_once_and_shows_in_the_activity_stack() {
    let d = scratch("scan");
    let mut s = with_model(&d);
    let ids: Vec<u64> = s.visible_cloned().iter().take(3).map(|i| i.0).collect();
    let r = s.execute("blur.scan", &json!({"ids": ids})).unwrap();
    assert_eq!(r["queued"], 3, "{r}");
    // the row is there while it runs
    let tasks = s.execute("activity.list", &json!({})).unwrap();
    assert!(tasks["tasks"].as_array().unwrap().iter().any(|t| t["kind"] == "blur"), "{tasks}");
    let t0 = std::time::Instant::now();
    while s.execute("blur.pump", &json!({})).unwrap()["active"] == true {
        assert!(t0.elapsed().as_secs() < 120, "scan didn't finish");
        std::thread::sleep(std::time::Duration::from_millis(5));
    }
    let last = s.execute("blur.pump", &json!({})).unwrap()["last"].clone();
    assert_eq!(last["done"], 3, "{last}");
    assert!(last["failed"].as_array().unwrap().is_empty(), "{last}");
    let sc = s.execute("blur.scores", &json!({"ids": ids})).unwrap();
    assert_eq!(sc["unscored"], 0, "{sc}");
    // done photos aren't scanned again unless asked
    let r = s.execute("blur.scan", &json!({"ids": ids, "wait": true})).unwrap();
    assert_eq!((r["queued"].as_u64(), r["alreadyScored"].as_u64()), (Some(0), Some(3)), "{r}");
    let r = s.execute("blur.scan", &json!({"ids": ids, "rescan": true, "wait": true})).unwrap();
    assert_eq!(r["queued"], 3, "{r}");
    assert_eq!(r["running"], false);
}

#[test]
fn apply_flags_inside_and_outside_the_window_in_one_undo_step() {
    let d = scratch("apply");
    let mut s = with_model(&d);
    let mid = s.execute("blur.model.status", &json!({})).unwrap()["model"]["id"].as_str().unwrap().to_string();
    let v: Vec<PhotoId> = s.visible_cloned().into_iter().take(5).collect();
    for (id, score) in v.iter().zip([Some(5.0), Some(20.0), Some(45.0), None, Some(90.0)]) {
        blur_cull::set_score(&mut s, *id, score, &mid);
    }
    // a score from another model doesn't count
    blur_cull::set_score(&mut s, v[4], Some(10.0), "someothermodel0");
    let ids: Vec<u64> = v.iter().map(|i| i.0).collect();
    let before: Vec<Flag> = v.iter().map(|i| s.catalog.photo(*i).unwrap().flag).collect();

    let sc = s.execute("blur.scores", &json!({"ids": ids})).unwrap();
    assert_eq!((sc["scored"].as_u64(), sc["noDetail"].as_u64(), sc["unscored"].as_u64()), (Some(3), Some(1), Some(1)), "{sc}");
    assert_eq!(sc["histogram"].as_array().unwrap().len(), 20);
    assert_eq!(sc["photos"].as_array().unwrap().len(), 5);
    // the dialog's per-frame query: no list, live counts for the window
    let q = s.execute("blur.scores", &json!({"ids": ids, "photos": false, "min": 0, "max": 25})).unwrap();
    assert!(q.get("photos").is_none(), "{q}");
    assert_eq!((q["inRange"].as_u64(), q["outOfRange"].as_u64()), (Some(2), Some(1)), "{q}");
    let h: u64 = q["histogram"].as_array().unwrap().iter().map(|v| v.as_u64().unwrap()).sum();
    assert_eq!(h, 3);

    let r = s.execute("blur.apply", &json!({"ids": ids, "min": 0, "max": 25})).unwrap();
    assert_eq!((r["inRange"].as_u64(), r["outOfRange"].as_u64(), r["notScored"].as_u64()), (Some(2), Some(1), Some(2)), "{r}");
    let flag = |s: &Session, i: usize| s.catalog.photo(v[i]).unwrap().flag;
    assert_eq!([flag(&s, 0), flag(&s, 1), flag(&s, 2)], [Flag::Pick, Flag::Pick, Flag::Reject]);
    assert_eq!((flag(&s, 3), flag(&s, 4)), (before[3], before[4]), "unscored photos are left alone");
    // one undo step brings every flag back
    s.execute("edit.undo", &json!({})).unwrap();
    let after: Vec<Flag> = v.iter().map(|i| s.catalog.photo(*i).unwrap().flag).collect();
    assert_eq!(after, before);

    // keeping the soft ones: inside → leave alone, outside → reject
    let r = s.execute("blur.apply", &json!({"ids": ids, "min": 0, "max": 50, "inRange": "keep", "outOfRange": "reject"})).unwrap();
    // inside (5, 20, 45) left as they were; nothing is outside 0..50 among usable scores; photo 4's score is stale
    assert_eq!((r["inRange"].as_u64(), r["outOfRange"].as_u64(), r["changed"].as_u64()), (Some(3), Some(0), Some(0)), "{r}");
    for bad in [
        json!({"min": 50, "max": 10}),
        json!({"min": 0}),
        json!({"min": 0, "max": 10, "inRange": "maybe"}),
        json!({"min": 0, "max": 10, "scope": "everything"}),
    ] {
        assert!(s.execute("blur.apply", &bad).is_err(), "{bad}");
    }
}

#[test]
fn scores_survive_on_disk_and_a_damaged_file_starts_afresh() {
    let d = scratch("store");
    let path = d.join(blur_cull::STORE_FILE);
    let e =
        blur_cull::Entry { k: "file:a.jpg:10".into(), m: "abc".into(), s: Some(12.5), b: Some(40.0), from: Some(lightcraft_blur::Subject::Focus) };
    let back = blur_cull::store_roundtrip(&path, &[(7, e.clone())]);
    assert_eq!(back.get(&7), Some(&e));
    std::fs::write(&path, b"{ broken").unwrap();
    assert!(blur_cull::store_roundtrip(&d.join("other.json"), &[]).is_empty());
    let mut s = Session::with_demo();
    // a session reading the damaged file doesn't fail
    let _ = s.execute("blur.scores", &json!({"scope": "view"})).unwrap();
}

#[test]
fn shutter_brackets_give_slow_shutter_speeds_their_own_window() {
    let d = scratch("brackets");
    let mut s = with_model(&d);
    let mid = s.execute("blur.model.status", &json!({})).unwrap()["model"]["id"].as_str().unwrap().to_string();
    let v: Vec<PhotoId> = s.visible_cloned().into_iter().take(5).collect();
    // (shutter, score): a fast frame a little soft, a 1/500 frame, a 1/30 pan, a 2 s pan, and a photo with no shutter
    // speed recorded
    for (id, (shutter, score)) in v.iter().zip([("1/1000", 18.0), ("1/500", 12.0), ("1/30", 35.0), ("2\"", 55.0), ("", 30.0)]) {
        let mut m = s.catalog.photo(*id).unwrap().meta.clone();
        m.shutter = shutter.to_string();
        s.commit("t", lightcraft_catalog::Op::SetMeta { id: *id, meta: Box::new(m) }).unwrap();
        blur_cull::set_score(&mut s, *id, Some(score), &mid);
    }
    let ids: Vec<u64> = v.iter().map(|i| i.0).collect();
    let brackets = json!([
        {"upTo": "1/500", "min": 0, "max": 15},
        {"upTo": 1.0 / 60.0, "min": 0, "max": 25},
        {"upTo": "1/15", "min": 0, "max": 40},
        {"min": 0, "max": 60},
    ]);

    // one window for all: only the two fast frames are kept
    let one = s.execute("blur.scores", &json!({"ids": ids, "photos": false, "min": 0, "max": 25})).unwrap();
    assert_eq!((one["inRange"].as_u64(), one["outOfRange"].as_u64()), (Some(2), Some(3)), "{one}");
    assert!(one.get("brackets").is_none(), "{one}");

    // by shutter speed: the pans are kept, the soft 1/1000 frame isn't, and the photo without a shutter speed
    // goes by min..max
    let q = s.execute("blur.scores", &json!({"ids": ids, "photos": true, "min": 0, "max": 25, "brackets": brackets})).unwrap();
    assert_eq!((q["inRange"].as_u64(), q["outOfRange"].as_u64()), (Some(3), Some(2)), "{q}");
    let per: Vec<(u64, u64, u64)> = q["brackets"]
        .as_array()
        .unwrap()
        .iter()
        .map(|b| (b["scored"].as_u64().unwrap(), b["inRange"].as_u64().unwrap(), b["outOfRange"].as_u64().unwrap()))
        .collect();
    assert_eq!(per, [(2, 1, 1), (0, 0, 0), (1, 1, 0), (1, 1, 0)], "{q}");
    assert_eq!((q["unknownShutter"]["scored"].as_u64(), q["unknownShutter"]["outOfRange"].as_u64()), (Some(1), Some(1)), "{q}");
    let hist: u64 = q["brackets"][0]["histogram"].as_array().unwrap().iter().map(|v| v.as_u64().unwrap()).sum();
    assert_eq!(hist, 2, "each bracket has its own histogram");
    let which: Vec<Value> = q["photos"].as_array().unwrap().iter().map(|p| p["bracket"].clone()).collect();
    assert_eq!(which, [json!(0), json!(0), json!(2), json!(3), Value::Null]);

    let r = s.execute("blur.apply", &json!({"ids": ids, "min": 0, "max": 25, "brackets": brackets})).unwrap();
    assert_eq!((r["inRange"].as_u64(), r["outOfRange"].as_u64(), r["notScored"].as_u64()), (Some(3), Some(2), Some(0)), "{r}");
    assert_eq!(r["byBracket"][0], json!({"inRange": 1, "outOfRange": 1}), "{r}");
    assert_eq!(r["unknownShutter"], json!({"inRange": 0, "outOfRange": 1}), "{r}");
    let flags: Vec<Flag> = v.iter().map(|i| s.catalog.photo(*i).unwrap().flag).collect();
    assert_eq!(flags, [Flag::Reject, Flag::Pick, Flag::Pick, Flag::Pick, Flag::Reject]);

    // without an open-ended last bracket, a speed slower than every bracket goes by min..max too
    let r = s
        .execute("blur.scores", &json!({"ids": ids, "photos": false, "min": 0, "max": 25, "brackets": [{"upTo": "1/15", "min": 0, "max": 40}]}))
        .unwrap();
    assert_eq!(r["unknownShutter"]["scored"].as_u64(), Some(2), "{r}");

    for bad in [
        json!("1/60"),
        json!([{"upTo": "1/60", "min": 0}]),
        json!([{"upTo": "1/60", "min": 30, "max": 10}]),
        json!([{"upTo": "fast", "min": 0, "max": 10}]),
        json!([{"upTo": -2, "min": 0, "max": 10}]),
        json!([{"min": 0, "max": 10}, {"upTo": "1/60", "min": 0, "max": 10}]),
        json!([{"upTo": "1/60", "min": 0, "max": 10}, {"upTo": "1/500", "min": 0, "max": 10}]),
        json!([{"upTo": "1/60", "min": 0, "max": 10}, {"upTo": "1/60", "min": 0, "max": 20}]),
        Value::Array(vec![json!({"upTo": "1/60", "min": 0, "max": 10}); 17]),
    ] {
        assert!(s.execute("blur.apply", &json!({"ids": ids, "min": 0, "max": 25, "brackets": bad})).is_err(), "{bad}");
        assert!(s.execute("blur.scores", &json!({"ids": ids, "brackets": bad})).is_err(), "{bad}");
    }
}
