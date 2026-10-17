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
    let services = Services { png: None, ..Default::default() };
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
