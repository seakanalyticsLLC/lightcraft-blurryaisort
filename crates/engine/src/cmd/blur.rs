//! Blur Cull commands (Photo ▸ Blur Cull…): install the optional BlurNet model, score how blurred each photo's subject
//! is in the background, and flag photos inside / outside a blur window. The work itself is `crate::blur_cull`.

use serde_json::{Value, json};

use super::{CommandSpec, always, bad, cmd};
use crate::{Result, Session};

/// The photos a command works on: `ids`/`id`; else `scope` (`selection` | `view`); else the selection when more than
/// one photo is selected, otherwise every photo in view (as Assisted Culling).
#[cfg_attr(target_arch = "wasm32", allow(dead_code))]
fn photos(s: &mut Session, p: &Value, cmd: &str) -> Result<Vec<lightcraft_catalog::PhotoId>> {
    if p.get("ids").is_some() || p.get("id").is_some() {
        return s.targets(p);
    }
    match super::str_param(p, "scope") {
        Some("selection") => Ok(s.selected()),
        Some("view") => Ok(s.visible_cloned()),
        None if s.selected().len() > 1 => Ok(s.selected()),
        None => Ok(s.visible_cloned()),
        Some(other) => Err(bad(cmd, format!("unknown scope `{other}` (selection | view)"))),
    }
}

/// Most brackets a cull takes.
#[cfg_attr(target_arch = "wasm32", allow(dead_code))]
const MAX_BRACKETS: usize = 16;

/// `brackets`: `[{upTo?, min, max}]`, fastest first. `upTo` is the bracket's slowest shutter speed, as cameras write it
/// (`"1/60"`, `"2"`) or in seconds; each bracket starts just slower than the one before. Only the last may leave
/// `upTo` out: it takes every slower speed.
#[cfg(not(target_arch = "wasm32"))]
fn brackets(p: &Value, cmd: &str) -> Result<Vec<crate::blur_cull::Bracket>> {
    let list = match p.get("brackets") {
        None | Some(Value::Null) => return Ok(Vec::new()),
        Some(Value::Array(a)) => a,
        Some(_) => return Err(bad(cmd, "`brackets` must be a list of {upTo?, min, max}")),
    };
    if list.len() > MAX_BRACKETS {
        return Err(bad(cmd, format!("at most {MAX_BRACKETS} `brackets`")));
    }
    let mut out: Vec<crate::blur_cull::Bracket> = Vec::with_capacity(list.len());
    for (i, b) in list.iter().enumerate() {
        let num = |k: &str| b.get(k).and_then(Value::as_f64).filter(|v| v.is_finite());
        let (Some(min), Some(max)) = (num("min"), num("max")) else {
            return Err(bad(cmd, format!("`brackets[{i}]` needs numbers `min` and `max`")));
        };
        if min > max {
            return Err(bad(cmd, format!("`brackets[{i}]`: `min` must be ≤ `max`")));
        }
        let up_to = match b.get("upTo") {
            None | Some(Value::Null) => None,
            Some(Value::String(t)) => Some(
                lightcraft_catalog::parse_shutter_seconds(t)
                    .ok_or_else(|| bad(cmd, format!("`brackets[{i}].upTo` needs a shutter speed like 1/60 or 2, not {t}")))?,
            ),
            Some(v) => Some(
                v.as_f64()
                    .filter(|t| t.is_finite() && *t > 0.0)
                    .ok_or_else(|| bad(cmd, format!("`brackets[{i}].upTo` needs a shutter speed like 1/60 or 2, not {v}")))?,
            ),
        };
        if up_to.is_none() && i + 1 < list.len() {
            return Err(bad(cmd, format!("`brackets[{i}]` needs `upTo`: only the last bracket can take every slower speed")));
        }
        if let (Some(t), Some(prev)) = (up_to, out.last().and_then(|b| b.up_to))
            && t <= prev
        {
            return Err(bad(cmd, format!("`brackets[{i}].upTo` must be slower than the bracket before it")));
        }
        out.push(crate::blur_cull::Bracket { up_to, min, max });
    }
    Ok(out)
}

#[cfg(not(target_arch = "wasm32"))]
mod native {
    use super::*;
    use crate::blur_cull::{self, Action, Windows};
    use lightcraft_catalog::Flag;

    pub(super) fn status(s: &mut Session, _: &Value) -> Result<Value> {
        Ok(blur_cull::status(s))
    }

    pub(super) fn install(s: &mut Session, p: &Value) -> Result<Value> {
        const C: &str = "blur.model.install";
        let path = super::super::str_param(p, "path").filter(|v| !v.trim().is_empty()).ok_or_else(|| bad(C, "missing `path`"))?;
        blur_cull::install(s, std::path::Path::new(path.trim()))
    }

    pub(super) fn remove(s: &mut Session, _: &Value) -> Result<Value> {
        blur_cull::remove(s)
    }

    pub(super) fn scan(s: &mut Session, p: &Value) -> Result<Value> {
        const C: &str = "blur.scan";
        let ids = photos(s, p, C)?;
        let wait = super::super::bool_or(p, "wait", false);
        let mut v = blur_cull::start(s, ids, super::super::bool_or(p, "rescan", false))?;
        if wait {
            while blur_cull::pump(s, false) {
                std::thread::sleep(std::time::Duration::from_millis(5));
            }
            v["running"] = json!(false);
            v["last"] = blur_cull::progress(s)["last"].clone();
        }
        Ok(v)
    }

    pub(super) fn pump(s: &mut Session, p: &Value) -> Result<Value> {
        let paused = super::super::str_param(p, "pace") == Some("pause");
        blur_cull::pump(s, paused);
        Ok(blur_cull::progress(s))
    }

    pub(super) fn scores(s: &mut Session, p: &Value) -> Result<Value> {
        const C: &str = "blur.scores";
        let ids = photos(s, p, C)?;
        let window = match (p.get("min").and_then(Value::as_f64), p.get("max").and_then(Value::as_f64)) {
            (Some(a), Some(b)) => Some((a, b)),
            _ => None,
        };
        let windows = Windows { brackets: super::brackets(p, C)?, other: window };
        Ok(blur_cull::scores(s, &ids, super::super::bool_or(p, "photos", true), &windows))
    }

    pub(super) fn apply(s: &mut Session, p: &Value) -> Result<Value> {
        const C: &str = "blur.apply";
        let min = super::super::f64_req(p, "min", C)?;
        let max = super::super::f64_req(p, "max", C)?;
        if !(min.is_finite() && max.is_finite()) || min > max {
            return Err(bad(C, "`min` and `max` must be numbers with min ≤ max"));
        }
        let inside = Action::parse(super::super::str_param(p, "inRange"), Action::Flag(Flag::Pick))
            .ok_or_else(|| bad(C, "`inRange` must be pick | reject | none | keep"))?;
        let outside = Action::parse(super::super::str_param(p, "outOfRange"), Action::Flag(Flag::Reject))
            .ok_or_else(|| bad(C, "`outOfRange` must be pick | reject | none | keep"))?;
        let windows = Windows { brackets: super::brackets(p, C)?, other: Some((min, max)) };
        let ids = photos(s, p, C)?;
        blur_cull::apply(s, &ids, &windows, inside, outside)
    }
}

#[cfg(target_arch = "wasm32")]
mod native {
    use super::*;

    fn unsupported(cmd: &str) -> crate::EngineError {
        bad(cmd, "Blur Cull runs in the desktop app only")
    }
    pub(super) fn status(_: &mut Session, _: &Value) -> Result<Value> {
        Ok(json!({"available": false, "installed": false}))
    }
    pub(super) fn install(_: &mut Session, _: &Value) -> Result<Value> {
        Err(unsupported("blur.model.install"))
    }
    pub(super) fn remove(_: &mut Session, _: &Value) -> Result<Value> {
        Err(unsupported("blur.model.remove"))
    }
    pub(super) fn scan(_: &mut Session, _: &Value) -> Result<Value> {
        Err(unsupported("blur.scan"))
    }
    pub(super) fn pump(_: &mut Session, _: &Value) -> Result<Value> {
        Ok(json!({"active": false}))
    }
    pub(super) fn scores(_: &mut Session, _: &Value) -> Result<Value> {
        Err(unsupported("blur.scores"))
    }
    pub(super) fn apply(_: &mut Session, _: &Value) -> Result<Value> {
        Err(unsupported("blur.apply"))
    }
}

pub fn specs() -> Vec<CommandSpec> {
    vec![
        cmd!(
            query "blur.model.status",
            "Blur Cull Model",
            [],
            None,
            "{} → {available, installed, path, model: {arch, analysisLongEdge, created, note, step, id, bytes} | null, error, scores, scan} — whether the optional Blur Cull model (BlurNet, trained with tools/blurcull) is installed and loads",
            always,
            native::status
        ),
        cmd!(
            query "blur.model.install",
            "Install Blur Cull Model",
            [],
            None,
            "{path} → {installed, path, untrained} — check a .safetensors file exported by tools/blurcull/export.py (format, architecture, every tensor) and copy it into the models folder, replacing the one installed",
            always,
            native::install
        ),
        cmd!(query "blur.model.remove", "Remove Blur Cull Model", [], None, "{} → {removed} — delete the installed Blur Cull model (scores are kept)", always, native::remove),
        cmd!(
            query "blur.scan",
            "Score Blur",
            [],
            None,
            "{ids?, scope?: selection | view, rescan?: bool, wait?: bool} → {queued, alreadyScored, running, last?} — score how blurred each photo's subject is (0 = tack sharp … 100), on a background job with a row in the activity stack. The subject is the camera's focus point, else faces, else the sharpest detailed part of the frame. Photos already scored by the installed model are skipped unless `rescan`. Without ids or scope: the selection when more than one photo is selected, else the photos in view. Scores are kept beside the library (no catalog change, no undo step). `wait: true` returns when done (scripts)",
            always,
            native::scan
        ),
        cmd!(
            query "blur.pump",
            "Score Blur in the Background",
            [],
            None,
            "{pace?: pause | normal} → {active, done, total, scored, noDetail, failed} or {active: false, last} — cheap to call every frame (the app does): collects the scan's results and feeds it more photos; feeds nothing while paused",
            always,
            native::pump
        ),
        cmd!(
            query "blur.scores",
            "Blur Scores",
            [],
            None,
            "{ids?, scope?: selection | view, photos?: bool (true), min?, max?, brackets?: [{upTo?, min, max}]} → {photos?: [{id, score, background, subject: focus | faces | auto, bracket?} | {id, score: null, unscored: true, bracket?}], histogram: [20 counts over 0..100], scored, noDetail, unscored, total, model, inRange?, outOfRange?, brackets?: [{histogram, scored, noDetail, unscored, inRange, outOfRange}], unknownShutter?: {histogram, scored, noDetail, unscored, inRange?, outOfRange?}} — `photos: false` leaves the list out (the dialog asks every frame); with min and max, how many scored photos fall inside / outside; with shutter `brackets` (as blur.apply), each photo is counted against its bracket's window and min..max covers the photos without a known shutter speed — the photos' subject blur scores from the installed model (a score from another model, or for a photo whose file changed, counts as unscored)",
            always,
            native::scores
        ),
        cmd!(
            "blur.apply",
            "Blur Cull",
            [],
            None,
            "{min, max, brackets?: [{upTo?, min, max}], inRange?: pick | reject | none | keep (pick), outOfRange?: pick | reject | none | keep (reject), ids?, scope?} → {inRange, outOfRange, notScored, changed, byBracket?: [{inRange, outOfRange}], unknownShutter?: {inRange, outOfRange}} — flag the photos whose subject blur score is within min..max (0..100, inclusive) one way and the others another, in one undo step; photos without a score are left alone. Shutter `brackets` (fastest first) give each range of shutter speeds its own window, so a slow pan can be kept softer than a fast frame: `upTo` is the bracket's slowest speed (\"1/60\", \"2\" or a number of seconds), each bracket starts just slower than the one before, and only the last may leave `upTo` out to take every slower speed; min..max is then the window for photos without a known shutter speed (or slower than every bracket)",
            always,
            native::apply
        ),
    ]
}
