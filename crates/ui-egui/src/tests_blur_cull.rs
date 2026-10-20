//! Photo ▸ Blur Cull…: without a model the dialog offers to install one; with one it scores the photos in the
//! background, the histogram's handles move the window, and Apply Flags runs `blur.apply` on the chosen photos.
#![cfg(not(target_arch = "wasm32"))]

use crate::headless::Headless;
use crate::state::Dialog;
use crate::{LightcraftApp, Services};
use serde_json::json;
use std::time::{Duration, Instant};

const T: Duration = Duration::from_secs(20);

fn app(model: Option<&std::path::Path>) -> Headless {
    let mut s = lightcraft_engine::Session::with_demo();
    if let Some(dir) = model {
        let dst = dir.join("models").join("blurnet.safetensors");
        s.set_blur_model_path(Some(dst));
        let src = dir.join("export.safetensors");
        std::fs::write(&src, lightcraft_blur::untrained_model_bytes(lightcraft_blur::ARCH)).unwrap();
        s.execute("blur.model.install", &json!({"path": src.to_string_lossy()})).unwrap();
    } else {
        s.set_blur_model_path(None);
    }
    // (PNG out for `LIGHTCRAFT_TEST_SHOTS` screenshots)
    let png = |img: &lightcraft_raster::Rgba8| {
        lightcraft_codecs::encode_png(&lightcraft_codecs::EncodeImage::rgba8(img), &lightcraft_codecs::EncodeMeta::default()).unwrap_or_default()
    };
    let services = Services {
        png: Some(Box::new(png)),
        write: Some(Box::new(|p: &str, b: &[u8]| std::fs::write(p, b).map_err(|e| e.to_string()))),
        ..Default::default()
    };
    let mut h = Headless::new(LightcraftApp::new(s, services), [1400.0, 1000.0], 1.0);
    h.settle(Duration::from_secs(120));
    h
}

fn has_widget(h: &mut Headless, id: &str) -> bool {
    let r = h.request("ui.widgets", json!({"filter": id}), T);
    r["result"].as_array().is_some_and(|a| a.iter().any(|w| w["id"] == id))
}

#[test]
fn without_a_model_the_dialog_offers_to_install_one() {
    let mut h = app(None);
    h.app.run("dialog.blurCull", json!({})).unwrap();
    h.step();
    h.step();
    assert!(has_widget(&mut h, "button:blurInstall"));
    assert!(!has_widget(&mut h, "button:blurScan"));
}

#[test]
fn scores_in_the_background_moves_the_window_and_applies() {
    let dir = std::env::temp_dir().join(format!("lc-blur-ui-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    let mut h = app(Some(&dir));
    let ids: Vec<u64> = h.app.session.visible_cloned().iter().take(2).map(|i| i.0).collect();
    h.app.run("library.select", json!({"ids": ids})).unwrap();
    h.app.run("dialog.blurCull", json!({"scope": "selection", "min": 0, "max": 25})).unwrap();
    h.step();
    h.step();
    assert!(has_widget(&mut h, "button:blurScan"), "two unscored photos: the dialog offers to score them");
    // nothing scored yet: Apply Flags does nothing
    let _ = h.request("ui.clickWidget", json!({"id": "button:dialogOk"}), T);
    h.step();
    assert!(h.app.ui.dialog.is_some(), "Apply Flags is disabled until photos have scores");
    let r = h.request("ui.clickWidget", json!({"id": "button:blurScan"}), T);
    assert_eq!(r["ok"], true, "{r}");
    let t0 = Instant::now();
    loop {
        h.step();
        let sc = h.app.session.execute("blur.scores", &json!({"ids": ids, "photos": false})).unwrap();
        if sc["unscored"] == 0 {
            break;
        }
        assert!(t0.elapsed() < Duration::from_secs(180), "the scan didn't finish: {sc}");
        std::thread::sleep(Duration::from_millis(20));
    }
    // a click near the right end moves the nearer handle (the upper one) there
    let r = h.request("ui.clickWidget", json!({"id": "image:blurHistogram", "fx": 0.9, "fy": 0.5}), T);
    assert_eq!(r["ok"], true, "{r}");
    h.step();
    let Some(Dialog::BlurCull { min, max, .. }) = h.app.ui.dialog.clone() else { panic!("the dialog closed") };
    assert_eq!(min, 0.0);
    assert!(max > 80.0, "{max}");
    // dragging the lower handle to the right moves only it
    let r = h.request("ui.dragWidget", json!({"id": "slider:blurMin", "dx": 100.0, "dy": -20.0, "steps": 6}), T);
    assert_eq!(r["ok"], true, "{r}");
    h.step();
    let Some(Dialog::BlurCull { min: min2, max: max2, .. }) = h.app.ui.dialog.clone() else { panic!("the dialog closed") };
    assert!(min2 > 10.0 && max2 == max, "{min2} {max2}");
    let r = h.request("ui.clickWidget", json!({"id": "button:dialogOk"}), T);
    assert_eq!(r["ok"], true, "{r}");
    h.step();
    assert!(h.app.ui.dialog.is_none());
    assert!(h.app.ui.status.starts_with("Blur Cull:"), "{}", h.app.ui.status);
}

fn brackets(h: &Headless) -> crate::panels::blur_cull::Brackets {
    let Some(Dialog::BlurCull { brackets, .. }) = h.app.ui.dialog.clone() else { panic!("the dialog closed") };
    brackets
}

#[test]
fn shutter_brackets_have_their_own_windows_and_are_kept_for_next_time() {
    let dir = std::env::temp_dir().join(format!("lc-blur-ui-brackets-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    let mut h = app(Some(&dir));
    assert!(!h.app.ui.blur_brackets.on, "off until asked for");
    let ids: Vec<u64> = h.app.session.visible_cloned().iter().take(2).map(|i| i.0).collect();
    h.app.run("library.select", json!({"ids": ids})).unwrap();
    h.app.run("dialog.blurCull", json!({"scope": "selection", "byShutter": true})).unwrap();
    h.step();
    h.step();
    for id in [
        "check:blurByShutter",
        "row:blurBracket0",
        "row:blurBracket3",
        "row:blurBracketUnknown",
        "combo:blurBracketLimit0",
        "button:blurBracketRemove0",
        "button:blurBracketAdd",
    ] {
        assert!(has_widget(&mut h, id), "{id}");
    }
    assert!(!has_widget(&mut h, "button:blurBracketRemove3"), "the open-ended bracket can't be removed");
    let r = h.request("ui.clickWidget", json!({"id": "button:blurScan"}), T);
    assert_eq!(r["ok"], true, "{r}");
    let t0 = Instant::now();
    loop {
        h.step();
        let sc = h.app.session.execute("blur.scores", &json!({"ids": ids, "photos": false})).unwrap();
        if sc["unscored"] == 0 {
            break;
        }
        assert!(t0.elapsed() < Duration::from_secs(180), "the scan didn't finish: {sc}");
        std::thread::sleep(Duration::from_millis(20));
    }

    // the second bracket on the histogram: its handles move its window, no one else's
    let r = h.request("ui.clickWidget", json!({"id": "row:blurBracket1"}), T);
    assert_eq!(r["ok"], true, "{r}");
    h.step();
    assert_eq!(brackets(&h).active, Some(1));
    let r = h.request("ui.clickWidget", json!({"id": "image:blurHistogram", "fx": 0.9, "fy": 0.5}), T);
    assert_eq!(r["ok"], true, "{r}");
    h.step();
    let b = brackets(&h);
    assert!(b.rows[1].max > 80.0, "{b:?}");
    assert_eq!((b.rows[0].max, b.rows[2].max, b.rows[3].max), (15.0, 40.0, 60.0), "{b:?}");
    let Some(Dialog::BlurCull { min, max, .. }) = h.app.ui.dialog.clone() else { panic!("the dialog closed") };
    assert_eq!((min, max), (0.0, 25.0), "the unknown-shutter window is unchanged");
    if let Some(p) = std::env::var_os("LIGHTCRAFT_TEST_SHOTS") {
        let r =
            h.request("ui.screenshot", json!({"path": std::path::Path::new(&p).join("blur-brackets.png").to_string_lossy(), "headless": true}), T);
        assert_eq!(r["ok"], true, "{r}");
    }

    // a new bracket goes before the open-ended one, a stop slower than the last limit; removing one hands its speeds on
    let r = h.request("ui.clickWidget", json!({"id": "button:blurBracketAdd"}), T);
    assert_eq!(r["ok"], true, "{r}");
    h.step();
    let b = brackets(&h);
    let limits: Vec<Option<&str>> = b.rows.iter().map(|r| r.up_to.as_deref()).collect();
    assert_eq!(limits, [Some("1/500"), Some("1/60"), Some("1/15"), Some("1/8"), None]);
    assert_eq!(b.active, Some(3));
    let r = h.request("ui.clickWidget", json!({"id": "button:blurBracketRemove0"}), T);
    assert_eq!(r["ok"], true, "{r}");
    h.step();
    let b = brackets(&h);
    let limits: Vec<Option<&str>> = b.rows.iter().map(|r| r.up_to.as_deref()).collect();
    assert_eq!(limits, [Some("1/60"), Some("1/15"), Some("1/8"), None]);
    assert!(h.app.ui.blur_brackets.on && h.app.ui.blur_brackets.rows == b.rows, "kept for next time: {:?}", h.app.ui.blur_brackets);

    // Apply Flags culls with the brackets: the same counts blur.scores gives for them
    let want = h.app.session.execute("blur.scores", &json!({"ids": ids, "photos": false, "min": 0, "max": 25, "brackets": b.param()})).unwrap();
    let r = h.request("ui.clickWidget", json!({"id": "button:dialogOk"}), T);
    assert_eq!(r["ok"], true, "{r}");
    h.step();
    assert!(h.app.ui.dialog.is_none());
    let expect = format!("Blur Cull: {} inside the range, {} outside", want["inRange"], want["outOfRange"]);
    assert!(h.app.ui.status.starts_with(&expect), "{} vs {expect}", h.app.ui.status);

    // the next time the dialog opens, the brackets are as they were left
    h.app.run("dialog.blurCull", json!({"scope": "selection"})).unwrap();
    h.step();
    let again = brackets(&h);
    assert!(again.on && again.rows == b.rows, "{again:?}");
}
