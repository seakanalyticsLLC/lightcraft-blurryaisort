//! Blur Cull: score how blurred each photo's *subject* is with the optional BlurNet model ([`lightcraft_blur`]), then
//! flag the photos whose score falls inside (or outside) a window the user picks — a sharp-only cull, or keeping the
//! intentionally soft shots as well.
//!
//! - **The model** is one file the user trains (`tools/blurcull`) and installs (`blur.model.install`); it lives at
//!   [`crate::config::default_blur_model_path`]. Nothing works without it and nothing else needs it.
//! - **Scanning** runs on one worker thread, fed a few photos at a time by `blur.pump` (the UI calls it every frame;
//!   `blur.scan {wait: true}` pumps until done). Its row in the activity stack shows progress, ✕ stops it. Each photo is
//!   read from its raw's embedded preview when that is big enough, else rendered with default settings, uncropped and
//!   upright: the frame its focus point and face regions are given in.
//! - **Scores** are kept beside the library in [`STORE_FILE`], not in the catalog: a catalog opened by a build without
//!   Blur Cull is unchanged. A score is used only while the photo's content and the model are the ones that made it.
//! - **Applying** the window (`blur.apply`) flags photos in one undo step; photos without a usable score are left alone.

use std::collections::{BTreeMap, HashMap, VecDeque};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, mpsc};
use std::time::{Duration, Instant, SystemTime};

use lightcraft_blur::{Model, SubjectScore};
use lightcraft_catalog::{Flag, MediaKind, Op, PhotoId};
use lightcraft_develop::DevelopSettings;
use lightcraft_meta::RegionKind;
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};

use crate::activity::{Cancel, TaskGuard};
use crate::cmd::bad;
use crate::media::{PreviewLoader, RenderJob, content_key};
use crate::{AiModel, EngineError, Result, Session};

/// Scores beside the library.
pub(crate) const STORE_FILE: &str = "blur-scores.json";
const STORE_VERSION: u32 = 1;
const MAX_STORE_BYTES: u64 = 256 << 20;
/// Photos handed to the worker ahead of the one it is on.
const IN_FLIGHT: usize = 3;
/// Results are written to disk at least this often while a scan runs.
const SAVE_EVERY: Duration = Duration::from_secs(5);
/// Score histogram bins (0..100).
pub(crate) const BINS: usize = 20;
/// The activity stack row's kind.
pub(crate) const ACTIVITY_KIND: &str = "blur";

/// One photo's reading.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub(crate) struct Entry {
    /// The photo's content key when it was read ([`content_key`]).
    pub(crate) k: String,
    /// The model that read it ([`lightcraft_blur::ModelInfo::id`]).
    pub(crate) m: String,
    /// Subject blur 0..100; `None` = nothing in the photo to judge.
    pub(crate) s: Option<f64>,
    #[serde(default)]
    pub(crate) b: Option<f64>,
    #[serde(default)]
    pub(crate) from: Option<lightcraft_blur::Subject>,
}

#[derive(Default, Serialize, Deserialize)]
struct StoreFile {
    version: u32,
    photos: BTreeMap<u64, Entry>,
}

/// The scores of the open library.
#[derive(Default)]
pub(crate) struct Store {
    path: Option<PathBuf>,
    pub(crate) photos: HashMap<u64, Entry>,
    dirty: bool,
}

impl Store {
    /// The scores at `path` (none, or an unreadable file: start empty; the file is replaced on the next save).
    fn open(path: Option<PathBuf>) -> Store {
        let photos = path
            .as_deref()
            .and_then(|p| match read_store(p) {
                Ok(v) => Some(v),
                Err(e) => {
                    if p.exists() {
                        log::warn!("Blur Cull scores {}: {e}; starting afresh", p.display());
                    }
                    None
                }
            })
            .unwrap_or_default();
        Store { path, photos, dirty: false }
    }

    pub(crate) fn insert(&mut self, id: PhotoId, e: Entry) {
        self.photos.insert(id.0, e);
        self.dirty = true;
    }

    pub(crate) fn save(&mut self) {
        let Some(path) = self.path.clone().filter(|_| self.dirty) else { return };
        let file = StoreFile { version: STORE_VERSION, photos: self.photos.iter().map(|(k, v)| (*k, v.clone())).collect() };
        let bytes = match serde_json::to_vec(&file) {
            Ok(b) => b,
            Err(e) => return log::warn!("Blur Cull scores: {e}"),
        };
        match lightcraft_catalog::safe_file::write_atomic(&path, &bytes) {
            Ok(()) => self.dirty = false,
            // kept in memory and tried again at the next save
            Err(e) => log::warn!("Blur Cull scores {}: {e}", path.display()),
        }
    }
}

fn read_store(path: &Path) -> std::result::Result<HashMap<u64, Entry>, String> {
    let bytes = lightcraft_blur::weights::read_capped(path, MAX_STORE_BYTES).map_err(|e| e.to_string())?;
    let f: StoreFile = serde_json::from_slice(&bytes).map_err(|e| e.to_string())?;
    if f.version != STORE_VERSION {
        return Err(format!("version {} (this build reads {STORE_VERSION})", f.version));
    }
    Ok(f.photos.into_iter().collect())
}

/// What the worker needs for one photo.
struct Item {
    id: PhotoId,
    content: String,
    job: RenderJob,
    preview: Option<(String, PreviewLoader)>,
    focus: Option<lightcraft_blur::Rect>,
    regions: Vec<lightcraft_blur::Rect>,
}

struct Done {
    id: PhotoId,
    content: String,
    result: std::result::Result<SubjectScore, String>,
}

struct Job {
    identity: Arc<()>,
    cancel: Arc<AtomicBool>,
    todo: VecDeque<PhotoId>,
    in_flight: usize,
    tx: mpsc::Sender<Item>,
    rx: mpsc::Receiver<Done>,
    guard: TaskGuard,
    model_id: String,
    edge: usize,
    total: u64,
    done: u64,
    scored: u64,
    no_detail: u64,
    failed: Vec<Value>,
    saved_at: Instant,
    started: Instant,
}

impl Drop for Job {
    fn drop(&mut self) {
        // the worker stops before its next photo (and when it can't hand a result over)
        self.cancel.store(true, Ordering::Relaxed);
    }
}

type Stamp = (u64, Option<SystemTime>);

/// Blur Cull's state in the session.
#[derive(Default)]
pub(crate) struct BlurCull {
    /// Where the model is installed (`None`: this host keeps no models; `blur.*` says so).
    pub(crate) model_path: Option<PathBuf>,
    model: Option<(Stamp, Arc<Model>)>,
    pub(crate) store: Store,
    library: Option<Arc<()>>,
    job: Option<Job>,
    last: Option<Value>,
}

/// Follow the session's library: another one (or none) drops a running job and loads that library's scores.
pub(crate) fn follow_library(s: &mut Session) {
    if s.blur.library.as_ref().is_some_and(|l| Arc::ptr_eq(l, &s.library_identity)) {
        return;
    }
    s.blur.store.save();
    s.blur.job = None;
    s.blur.last = None;
    s.blur.library = Some(Arc::clone(&s.library_identity));
    let path = s.library.as_ref().filter(|l| l.on_disk).map(|l| l.dir.join(STORE_FILE));
    s.blur.store = Store::open(path);
}

fn missing(cmd: &str) -> EngineError {
    EngineError::ModelMissing {
        model: AiModel::BlurCull,
        msg: bad(cmd, "no Blur Cull model is installed: train one with tools/blurcull (see its README), then Install Model…").to_string(),
    }
}

/// The installed model, loaded (again when its file changed).
pub(crate) fn model(s: &mut Session, cmd: &str) -> Result<Arc<Model>> {
    let path = s.blur.model_path.clone().ok_or_else(|| missing(cmd))?;
    let Ok(meta) = std::fs::metadata(&path) else {
        s.blur.model = None;
        return Err(missing(cmd));
    };
    let stamp = (meta.len(), meta.modified().ok());
    if let Some((st, m)) = &s.blur.model
        && *st == stamp
    {
        return Ok(Arc::clone(m));
    }
    let m = Arc::new(Model::load(&path).map_err(|e| bad(cmd, format!("the installed Blur Cull model can't be used: {e}")))?);
    s.blur.model = Some((stamp, Arc::clone(&m)));
    Ok(m)
}

/// The id of the model scores must come from to count, if one is installed and loads.
fn current_model_id(s: &mut Session) -> Option<String> {
    model(s, "blur").ok().map(|m| m.info.id.clone())
}

/// The photo's usable score: same content, and from the installed model (any model when none is installed).
pub(crate) fn fresh<'a>(s: &'a Session, id: PhotoId, model_id: Option<&str>) -> Option<&'a Entry> {
    let ph = s.catalog.photo(id)?;
    let e = s.blur.store.photos.get(&id.0)?;
    (e.k == content_key(ph) && model_id.is_none_or(|m| e.m == m)).then_some(e)
}

fn rect(r: &lightcraft_meta::Rect) -> lightcraft_blur::Rect {
    lightcraft_blur::Rect { x0: r.x0, y0: r.y0, x1: r.x1, y1: r.y1 }
}

fn prepare(s: &mut Session, id: PhotoId, edge: usize) -> Option<Item> {
    let photo = s.catalog.photo(id)?.clone();
    if photo.kind == MediaKind::Video {
        return None;
    }
    let job = s.preview_job(id, edge, edge, false, &DevelopSettings::default())?;
    let preview = s.scan_preview_of(&photo);
    let focus = photo.meta.focus.as_ref().map(|f| rect(&f.rect));
    let regions = photo.meta.regions.iter().filter(|r| matches!(r.kind, RegionKind::Face | RegionKind::Pet)).map(|r| rect(&r.rect)).collect();
    Some(Item { id, content: content_key(&photo), job, preview, focus, regions })
}

fn analyse(model: &Model, item: Item) -> std::result::Result<SubjectScore, String> {
    let edge = model.info.analysis_long_edge;
    // a raw's embedded preview, when it is (nearly) as big as the analysis size, saves decoding the raw
    let from_preview =
        item.preview.and_then(|(path, load)| load(&path, edge)).filter(|img| img.width.max(img.height).saturating_mul(10) >= edge.saturating_mul(9));
    let image = match from_preview {
        Some(img) => img,
        None => match item.job.run().rendered {
            Ok(r) => r.image,
            Err(_) => return Err("could not render the photo".into()),
        },
    };
    let rgb: Vec<u8> = image.data.iter().flat_map(|px| [px[0], px[1], px[2]]).collect();
    let (rgb, w, h) = lightcraft_blur::downscale_rgb(&rgb, image.width, image.height, edge);
    let map = model.blur_map(&rgb, w, h).map_err(|e| e.to_string())?;
    Ok(map.subject_score(item.focus.as_ref(), &item.regions))
}

/// Scan `ids` (those without a usable score, or all with `rescan`). A running scan with the same model takes them on.
pub(crate) fn start(s: &mut Session, ids: Vec<PhotoId>, rescan: bool) -> Result<Value> {
    const C: &str = "blur.scan";
    follow_library(s);
    let model = model(s, C)?;
    let mid = model.info.id.clone();
    let mut seen = std::collections::HashSet::new();
    let mut fresh_n = 0u64;
    let mut todo: Vec<PhotoId> = Vec::new();
    for id in ids {
        if !seen.insert(id.0) || s.catalog.photo(id).is_none_or(|p| p.kind == MediaKind::Video) {
            continue;
        }
        if !rescan && fresh(s, id, Some(&mid)).is_some() {
            fresh_n += 1;
            continue;
        }
        todo.push(id);
    }
    if let Some(job) = s.blur.job.as_mut()
        && job.model_id == mid
    {
        let queued: std::collections::HashSet<u64> = job.todo.iter().map(|i| i.0).collect();
        let add: Vec<PhotoId> = todo.into_iter().filter(|i| !queued.contains(&i.0)).collect();
        job.total += add.len() as u64;
        job.guard.progress(job.done, job.total);
        let n = add.len();
        job.todo.extend(add);
        return Ok(json!({"queued": n, "alreadyScored": fresh_n, "running": true}));
    }
    s.blur.job = None;
    if todo.is_empty() {
        return Ok(json!({"queued": 0, "alreadyScored": fresh_n, "running": false}));
    }
    let cancel = Arc::new(AtomicBool::new(false));
    let (tx, items) = mpsc::channel::<Item>();
    let (done_tx, rx) = mpsc::channel::<Done>();
    let (worker_cancel, worker_model) = (Arc::clone(&cancel), Arc::clone(&model));
    std::thread::Builder::new()
        .name("blur-cull".into())
        .spawn(move || {
            for item in items {
                if worker_cancel.load(Ordering::Relaxed) {
                    return;
                }
                let (id, content) = (item.id, item.content.clone());
                // one bad file is that file's problem, not the scan's end
                let result = crate::guard::catch("scoring blur", || analyse(&worker_model, item)).and_then(|r| r);
                if done_tx.send(Done { id, content, result }).is_err() {
                    return;
                }
            }
        })
        .map_err(|e| bad(C, format!("could not start the scan: {e}")))?;
    let total = todo.len() as u64;
    let guard = s.activity.start(ACTIVITY_KIND, "Scoring blur", Cancel::Flag(Arc::clone(&cancel)));
    guard.progress(0, total);
    let now = Instant::now();
    let edge = model.info.analysis_long_edge;
    s.blur.job = Some(Job {
        identity: Arc::clone(&s.library_identity),
        cancel,
        todo: todo.into(),
        in_flight: 0,
        tx,
        rx,
        guard,
        model_id: mid,
        edge,
        total,
        done: 0,
        scored: 0,
        no_detail: 0,
        failed: Vec::new(),
        saved_at: now,
        started: now,
    });
    pump(s, false);
    Ok(json!({"queued": total, "alreadyScored": fresh_n, "running": true}))
}

/// Collect results and feed the worker. `paused`: hand it nothing new (the user is dragging or typing).
/// Returns whether a scan is still running.
pub(crate) fn pump(s: &mut Session, paused: bool) -> bool {
    follow_library(s);
    let Some(mut job) = s.blur.job.take() else { return false };
    if !Arc::ptr_eq(&job.identity, &s.library_identity) {
        return false;
    }
    if job.guard.is_cancelled() {
        s.blur.store.save();
        s.blur.last = Some(summary(&job, true));
        return false;
    }
    while let Ok(d) = job.rx.try_recv() {
        job.in_flight = job.in_flight.saturating_sub(1);
        job.done += 1;
        let name = s.catalog.photo(d.id).map(|p| p.file_name.clone()).unwrap_or_default();
        match d.result {
            Ok(r) => {
                // a photo replaced or relinked meanwhile is read again next time
                if s.catalog.photo(d.id).is_some_and(|p| content_key(p) == d.content) {
                    if r.score.is_some() {
                        job.scored += 1;
                    } else {
                        job.no_detail += 1;
                    }
                    s.blur.store.insert(d.id, Entry { k: d.content, m: job.model_id.clone(), s: r.score, b: r.background, from: r.subject });
                }
            }
            Err(e) => {
                if job.failed.len() < 200 {
                    job.failed.push(json!({"id": d.id.0, "file": name, "error": e}));
                }
            }
        }
        job.guard.detail(&name);
    }
    while !paused && job.in_flight < IN_FLIGHT {
        let Some(id) = job.todo.pop_front() else { break };
        match prepare(s, id, job.edge) {
            Some(item) => {
                if job.tx.send(item).is_err() {
                    // the worker is gone: nothing more will come back
                    job.failed.push(json!({"id": id.0, "error": "the scan stopped unexpectedly"}));
                    job.todo.clear();
                    job.in_flight = 0;
                    break;
                }
                job.in_flight += 1;
            }
            None => {
                job.done += 1;
                if job.failed.len() < 200 {
                    job.failed.push(json!({"id": id.0, "error": "can't be read"}));
                }
            }
        }
    }
    job.guard.progress(job.done, job.total);
    let finished = job.todo.is_empty() && job.in_flight == 0;
    if finished || job.saved_at.elapsed() >= SAVE_EVERY {
        s.blur.store.save();
        job.saved_at = Instant::now();
    }
    if finished {
        s.blur.last = Some(summary(&job, false));
        return false;
    }
    s.blur.job = Some(job);
    true
}

fn summary(job: &Job, cancelled: bool) -> Value {
    json!({
        "total": job.total, "done": job.done, "scored": job.scored, "noDetail": job.no_detail,
        "failed": job.failed, "cancelled": cancelled, "seconds": job.started.elapsed().as_secs_f64(),
    })
}

/// Progress of the running scan (or the last one).
pub(crate) fn progress(s: &Session) -> Value {
    match &s.blur.job {
        Some(j) => json!({"active": true, "done": j.done, "total": j.total, "scored": j.scored, "noDetail": j.no_detail, "failed": j.failed.len()}),
        None => json!({"active": false, "last": s.blur.last}),
    }
}

/// Model status for `blur.model.status`.
pub(crate) fn status(s: &mut Session) -> Value {
    follow_library(s);
    let path = s.blur.model_path.clone();
    let exists = path.as_deref().is_some_and(Path::exists);
    let (info, error) = if exists {
        match model(s, "blur.model.status") {
            Ok(m) => (serde_json::to_value(&m.info).unwrap_or(Value::Null), Value::Null),
            Err(e) => (Value::Null, json!(e.to_string())),
        }
    } else {
        (Value::Null, Value::Null)
    };
    json!({
        "available": true,
        "installed": !info.is_null(),
        "path": path,
        "model": info,
        "error": error,
        "scores": s.blur.store.photos.len(),
        "scan": progress(s),
    })
}

/// Check a model file and copy it into place.
pub(crate) fn install(s: &mut Session, src: &Path) -> Result<Value> {
    const C: &str = "blur.model.install";
    let dst = s.blur.model_path.clone().ok_or_else(|| bad(C, "this build has nowhere to keep models"))?;
    let bytes =
        lightcraft_blur::weights::read_capped(src, lightcraft_blur::MAX_MODEL_BYTES).map_err(|e| bad(C, format!("{}: {e}", src.display())))?;
    let m = Model::from_bytes(&bytes).map_err(|e| bad(C, format!("{}: {e}", src.display())))?;
    if let Some(dir) = dst.parent() {
        std::fs::create_dir_all(dir).map_err(|e| bad(C, format!("{}: {e}", dir.display())))?;
    }
    lightcraft_catalog::safe_file::write_atomic(&dst, &bytes).map_err(|e| bad(C, format!("{}: {e}", dst.display())))?;
    s.blur.model = None;
    s.blur.job = None;
    let untrained = m.info.step == 0;
    Ok(json!({"installed": m.info, "path": dst, "untrained": untrained}))
}

pub(crate) fn remove(s: &mut Session) -> Result<Value> {
    const C: &str = "blur.model.remove";
    let path = s.blur.model_path.clone().ok_or_else(|| bad(C, "this build has nowhere to keep models"))?;
    s.blur.job = None;
    s.blur.model = None;
    match std::fs::remove_file(&path) {
        Ok(()) => Ok(json!({"removed": true})),
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(json!({"removed": false})),
        Err(e) => Err(bad(C, format!("{}: {e}", path.display()))),
    }
}

/// Scores of `ids` (listed when `list`) and their histogram; with `window`, how many fall inside and outside it.
pub(crate) fn scores(s: &mut Session, ids: &[PhotoId], list: bool, window: Option<(f64, f64)>) -> Value {
    follow_library(s);
    let mid = current_model_id(s);
    let mut hist = [0u64; BINS];
    let (mut scored, mut no_detail, mut unscored) = (0u64, 0u64, 0u64);
    let (mut inside, mut outside) = (0u64, 0u64);
    let mut photos = Vec::with_capacity(if list { ids.len() } else { 0 });
    for id in ids {
        match fresh(s, *id, mid.as_deref()) {
            Some(e) => {
                if let Some(v) = e.s {
                    scored += 1;
                    if let Some((lo, hi)) = window {
                        if (lo..=hi).contains(&v) {
                            inside += 1;
                        } else {
                            outside += 1;
                        }
                    }
                    let bin = ((v / 100.0 * BINS as f64).floor().max(0.0) as usize).min(BINS - 1);
                    if let Some(h) = hist.get_mut(bin) {
                        *h += 1;
                    }
                } else {
                    no_detail += 1;
                }
                if list {
                    photos.push(json!({"id": id.0, "score": e.s, "background": e.b, "subject": e.from}));
                }
            }
            None => {
                unscored += 1;
                if list {
                    photos.push(json!({"id": id.0, "score": Value::Null, "unscored": true}));
                }
            }
        }
    }
    let mut v = json!({"histogram": hist, "scored": scored, "noDetail": no_detail, "unscored": unscored, "total": ids.len(), "model": mid});
    if list {
        v["photos"] = Value::Array(photos);
    }
    if window.is_some() {
        v["inRange"] = json!(inside);
        v["outOfRange"] = json!(outside);
    }
    v
}

/// What to do with photos inside / outside the window.
#[derive(Clone, Copy, PartialEq, Eq)]
pub(crate) enum Action {
    Flag(Flag),
    Keep,
}

impl Action {
    pub(crate) fn parse(v: Option<&str>, default: Action) -> Option<Action> {
        Some(match v {
            None => default,
            Some("pick") => Action::Flag(Flag::Pick),
            Some("reject") => Action::Flag(Flag::Reject),
            Some("none" | "unflag") => Action::Flag(Flag::None),
            Some("keep" | "leave") => Action::Keep,
            Some(_) => return None,
        })
    }
}

/// Flag `ids` by whether their score is within `min..=max`: one undo step.
pub(crate) fn apply(s: &mut Session, ids: &[PhotoId], min: f64, max: f64, inside: Action, outside: Action) -> Result<Value> {
    follow_library(s);
    let mid = current_model_id(s);
    let (mut n_in, mut n_out, mut skipped) = (0u64, 0u64, 0u64);
    let mut ops = Vec::new();
    for id in ids {
        let Some(score) = fresh(s, *id, mid.as_deref()).and_then(|e| e.s) else {
            skipped += 1;
            continue;
        };
        let within = (min..=max).contains(&score);
        if within {
            n_in += 1;
        } else {
            n_out += 1;
        }
        let Action::Flag(f) = (if within { inside } else { outside }) else { continue };
        if s.catalog.photo(*id).is_some_and(|p| p.flag != f) {
            ops.push(Op::SetFlag { id: *id, flag: f });
        }
    }
    let changed = ops.len();
    if !ops.is_empty() {
        s.commit("Blur Cull", Op::Batch { ops })?;
    }
    Ok(json!({"inRange": n_in, "outOfRange": n_out, "notScored": skipped, "changed": changed}))
}

#[cfg(test)]
pub(crate) fn set_score(s: &mut Session, id: PhotoId, score: Option<f64>, model_id: &str) {
    follow_library(s);
    let Some(k) = s.catalog.photo(id).map(|p| content_key(p)) else { return };
    s.blur.store.insert(id, Entry { k, m: model_id.into(), s: score, b: None, from: Some(lightcraft_blur::Subject::Auto) });
}

#[cfg(test)]
pub(crate) fn store_roundtrip(path: &Path, photos: &[(u64, Entry)]) -> HashMap<u64, Entry> {
    let mut st = Store { path: Some(path.to_path_buf()), photos: photos.iter().cloned().collect(), dirty: true };
    st.save();
    Store::open(Some(path.to_path_buf())).photos
}
