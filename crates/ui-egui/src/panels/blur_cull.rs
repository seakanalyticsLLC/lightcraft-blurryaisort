//! Photo ▸ Blur Cull…: score how blurred each photo's subject is with the optional Blur Cull model (`blur.*`), pick a
//! window of blur on the histogram, and flag the photos inside it one way and the rest another. A window starting at
//! 0 is a sharp-only cull; widening it keeps intentionally soft shots too.
//!
//! The dialog asks the engine for the scores' histogram and the counts every frame (`blur.scores` with
//! `photos: false`, cheap), feeds the scan while it is open, and the background pump ([`pump`]) keeps it going after
//! the dialog closes. The scan's progress is also its row in the activity stack (✕ there stops it).
//!
//! **By Shutter Speed** splits the window into shutter brackets ([`Brackets`]): each range of shutter speeds keeps its
//! own window (a slow pan is never as crisp as a 1/1000 frame), plus one for photos whose shutter speed isn't known.
//! Clicking a bracket's row puts its photos on the histogram and its window under the handles. The brackets are kept
//! with the UI state (`ui.json`) for next time.

use egui::{Align2, CornerRadius, Pos2, Rect, Sense, Stroke, StrokeKind, Ui, pos2, vec2};
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};

use crate::LightcraftApp;
use crate::theme::Tokens;

/// How often a running scan is asked for its results from the background pump.
const BUSY_SECS: f64 = 0.25;
/// How often an idle pump looks (a scan started elsewhere, e.g. over the control channel).
const IDLE_SECS: f64 = 1.0;

/// The dialog's actions, as `blur.apply` names them (key, label).
pub const INSIDE: [(&str, &str); 3] = [("Pick", "pick"), ("Unflag", "none"), ("Leave", "keep")];
pub const OUTSIDE: [(&str, &str); 3] = [("Reject", "reject"), ("Unflag", "none"), ("Leave", "keep")];

/// Shutter speeds a bracket can end at, fastest first (third stops), as cameras write them.
pub const SHUTTER_STOPS: [&str; 55] = [
    "1/8000", "1/6400", "1/5000", "1/4000", "1/3200", "1/2500", "1/2000", "1/1600", "1/1250", "1/1000", "1/800", "1/640", "1/500", "1/400", "1/320",
    "1/250", "1/200", "1/160", "1/125", "1/100", "1/80", "1/60", "1/50", "1/40", "1/30", "1/25", "1/20", "1/15", "1/13", "1/10", "1/8", "1/6", "1/5",
    "1/4", "0.3\"", "0.4\"", "0.5\"", "0.6\"", "0.8\"", "1\"", "1.3\"", "1.6\"", "2\"", "2.5\"", "3.2\"", "4\"", "5\"", "6\"", "8\"", "10\"", "13\"",
    "15\"", "20\"", "25\"", "30\"",
];
/// Most brackets (the last, open-ended one included).
pub const MAX_BRACKETS: usize = 8;

/// One shutter bracket: the speeds slower than the bracket before it, down to `up_to` (`None`: every slower speed),
/// keep a subject blur within `min..=max`.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct Bracket {
    #[serde(default)]
    pub up_to: Option<String>,
    pub min: f32,
    pub max: f32,
}

/// Photo ▸ Blur Cull…'s shutter brackets.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default, rename_all = "camelCase")]
pub struct Brackets {
    /// Cull by shutter speed.
    pub on: bool,
    /// Fastest first; only the last has no `up_to` ([`Brackets::normalized`]).
    pub rows: Vec<Bracket>,
    /// The bracket on the histogram (`None`: the photos without a known shutter speed, whose window is the dialog's
    /// own `min..max`).
    #[serde(skip)]
    pub active: Option<usize>,
}

impl Default for Brackets {
    fn default() -> Self {
        let row = |up: Option<&str>, max: f32| Bracket { up_to: up.map(str::to_string), min: 0.0, max };
        Brackets {
            on: false,
            rows: vec![row(Some("1/500"), 15.0), row(Some("1/60"), 25.0), row(Some("1/15"), 40.0), row(None, 60.0)],
            active: Some(0),
        }
    }
}

fn seconds(up_to: &str) -> Option<f64> {
    lightcraft_catalog::parse_shutter_seconds(up_to)
}

impl Brackets {
    /// Saved brackets as written in `ui.json`; anything that isn't is the defaults, and they're [`Self::normalized`].
    pub fn lenient<'de, D: serde::Deserializer<'de>>(d: D) -> Result<Brackets, D::Error> {
        let value = Value::deserialize(d)?;
        Ok(serde_json::from_value::<Brackets>(value).unwrap_or_default().normalized())
    }

    /// Brackets the dialog can show: limits that parse and get slower down the list, one open-ended bracket last, at
    /// most [`MAX_BRACKETS`], and windows within 0..100 with min ≤ max.
    pub fn normalized(mut self) -> Brackets {
        let window = |b: &Bracket| {
            let f = |v: f32, d: f32| if v.is_finite() { v.clamp(0.0, 100.0) } else { d };
            let (lo, hi) = (f(b.min, 0.0), f(b.max, 100.0));
            (lo.min(hi), hi.max(lo))
        };
        let mut rows: Vec<Bracket> = Vec::new();
        let mut last = 0.0f64;
        for b in self.rows.iter().filter(|b| b.up_to.is_some()) {
            let Some(t) = b.up_to.as_deref().and_then(seconds) else { continue };
            if t > last && rows.len() + 1 < MAX_BRACKETS {
                last = t;
                let (min, max) = window(b);
                rows.push(Bracket { up_to: b.up_to.clone(), min, max });
            }
        }
        let (min, max) = self.rows.iter().find(|b| b.up_to.is_none()).map_or((0.0, 60.0), window);
        rows.push(Bracket { up_to: None, min, max });
        self.rows = rows;
        self.active = self.active.filter(|i| *i < self.rows.len());
        self
    }

    /// `blur.apply` / `blur.scores` `brackets`.
    pub fn param(&self) -> Value {
        self.rows.iter().map(|b| json!({"upTo": b.up_to, "min": b.min, "max": b.max})).collect()
    }

    /// The speeds bracket `i` covers, in words (`None`: the photos without a known shutter speed).
    pub fn label(&self, i: Option<usize>) -> String {
        let tr = crate::i18n::tr;
        let Some(i) = i else { return tr("Unknown shutter speed").to_string() };
        let prev = i.checked_sub(1).and_then(|p| self.rows.get(p)).and_then(|b| b.up_to.clone());
        match (self.rows.get(i).and_then(|b| b.up_to.clone()), prev) {
            (Some(up), None) => crate::i18n::tr_format!("{up} and faster", up = up),
            (Some(up), Some(prev)) => format!("{prev} – {up}"),
            (None, Some(prev)) => crate::i18n::tr_format!("Slower than {prev}", prev = prev),
            (None, None) => tr("Any shutter speed").to_string(),
        }
    }

    /// The limits bracket `i` may end at: slower than the bracket before it, faster than the one after.
    fn choices(&self, i: usize) -> Vec<&'static str> {
        let limit = |j: Option<usize>| j.and_then(|j| self.rows.get(j)).and_then(|b| b.up_to.as_deref()).and_then(seconds);
        let (lo, hi) = (limit(i.checked_sub(1)), limit(Some(i + 1)));
        SHUTTER_STOPS
            .iter()
            .copied()
            .filter(|s| seconds(s).is_some_and(|t| lo.is_none_or(|lo| t > lo * (1.0 + 1e-6)) && hi.is_none_or(|hi| t < hi * (1.0 - 1e-6))))
            .collect()
    }

    /// A new bracket ending a stop slower than the last limited one, before the open-ended one; `None` when there's no
    /// room (the most brackets, or no slower speed left).
    pub fn add(&mut self) -> Option<usize> {
        if self.rows.len() >= MAX_BRACKETS {
            return None;
        }
        let at = self.rows.len().saturating_sub(1);
        let last = at.checked_sub(1).and_then(|j| self.rows.get(j)).and_then(|b| b.up_to.as_deref()).and_then(seconds);
        let up = match last {
            None => "1/250",
            // a stop is three steps of the list
            Some(t) => {
                let next = SHUTTER_STOPS.iter().position(|s| seconds(s).is_some_and(|v| v > t * (1.0 + 1e-6)))?;
                SHUTTER_STOPS.get(next + 2).or_else(|| SHUTTER_STOPS.last()).copied()?
            }
        };
        let (min, max) = self.rows.get(at).map_or((0.0, 40.0), |b| (b.min, b.max));
        self.rows.insert(at, Bracket { up_to: Some(up.to_string()), min, max });
        self.active = Some(at);
        Some(at)
    }

    /// Remove bracket `i` (never the open-ended last one): the bracket after it takes its speeds.
    pub fn remove(&mut self, i: usize) {
        if i + 1 >= self.rows.len() {
            return;
        }
        self.rows.remove(i);
        self.active = match self.active {
            Some(a) if a > i => Some(a - 1),
            Some(a) if a == i => Some(i.min(self.rows.len().saturating_sub(1))),
            other => other,
        };
    }
}

/// Called every frame: keeps a running scan fed and its results collected (nothing new while the user drags, types
/// or scrolls).
pub fn pump(app: &mut LightcraftApp, ctx: &egui::Context) {
    let now = ctx.input(|i| i.time);
    if now < app.caches.blur_next_pump {
        return;
    }
    let pace = crate::panels::faces::background_pace(app, ctx, now);
    let active = app.session.execute("blur.pump", &json!({"pace": pace})).is_ok_and(|v| v["active"] == true);
    app.caches.blur_next_pump = now + if active { BUSY_SECS } else { IDLE_SECS };
    if active {
        ctx.request_repaint_after(std::time::Duration::from_secs_f64(BUSY_SECS));
    }
}

fn scope(selection: bool) -> &'static str {
    if selection { "selection" } else { "view" }
}

/// The dialog's body.
#[allow(clippy::too_many_arguments)]
pub fn body(
    app: &mut LightcraftApp,
    ui: &mut Ui,
    selection: &mut bool,
    min: &mut f32,
    max: &mut f32,
    inside: &mut String,
    outside: &mut String,
    brackets: &mut Brackets,
) {
    let t = Tokens::get(ui.ctx());
    let tr = crate::i18n::tr;
    let status = app.session.execute("blur.model.status", &json!({})).unwrap_or_default();
    let model = &status["model"];

    // the model
    if status["installed"] != true {
        let msg = if status["available"] == false {
            tr("Blur Cull runs in the desktop app.").to_string()
        } else if let Some(e) = status["error"].as_str() {
            crate::i18n::tr_format!("The installed Blur Cull model can't be used: {e}", e = e)
        } else {
            tr(
                "No Blur Cull model is installed. Train one on your own photos with tools/blurcull (see its README), then install the blurnet.safetensors file it exports.",
            )
            .to_string()
        };
        ui.label(egui::RichText::new(msg).color(t.text_label));
        if status["available"] != false {
            let r = crate::button::Button::new(tr("Install Model…")).widget("button:blurInstall").show(ui);
            if r.clicked()
                && let Err(e) = app.run("blur.chooseModel", json!({}))
            {
                app.ui.status = e;
            }
        }
        return;
    }
    let trained = model["step"].as_u64().unwrap_or(0);
    let about = if trained == 0 {
        tr("Model: untrained (a plumbing test): its scores mean nothing.").to_string()
    } else {
        let created = model["created"].as_str().unwrap_or("").chars().take(10).collect::<String>();
        let mut about = crate::i18n::tr_format!("Model trained {created}, {trained} steps", created = created, trained = trained);
        // the user's own note on the model, as they wrote it
        if let Some(note) = model["note"].as_str().filter(|n| !n.is_empty()) {
            about = format!("{about} · {note}");
        }
        about
    };
    ui.horizontal(|ui| {
        ui.label(egui::RichText::new(about).color(if trained == 0 { t.caution } else { t.text_dim }).size(11.5));
        if ui.small_button(tr("Replace…")).clicked()
            && let Err(e) = app.run("blur.chooseModel", json!({}))
        {
            app.ui.status = e;
        }
    });

    // which photos
    let n_sel = app.session.selection.ids.len();
    let n_view = app.session.visible_cloned().len();
    let sel_label = crate::i18n::tr_format!("Selected ({n})", n = n_sel);
    let view_label = crate::i18n::tr_format!("All in View ({n})", n = n_view);
    if let Some(i) = crate::widgets::segmented(
        ui,
        "blurScope",
        &[(sel_label.as_str(), "selection"), (view_label.as_str(), "view")],
        Some(usize::from(!*selection)),
        2,
    ) {
        *selection = i == 0;
    }

    // their scores, the scan
    let mut q = json!({"scope": scope(*selection), "photos": false, "min": *min, "max": *max});
    if brackets.on {
        q["brackets"] = brackets.param();
    }
    let sc = app.session.execute("blur.scores", &q).unwrap_or_default();
    let scan = &status["scan"];
    let running = scan["active"] == true;
    if running {
        let _ = app.session.execute("blur.pump", &json!({}));
        ui.ctx().request_repaint_after(std::time::Duration::from_millis(200));
    }
    let unscored = sc["unscored"].as_u64().unwrap_or(0);
    ui.horizontal(|ui| {
        if running {
            let (done, total) = (scan["done"].as_u64().unwrap_or(0), scan["total"].as_u64().unwrap_or(0));
            ui.add(
                egui::ProgressBar::new(if total > 0 { done as f32 / total as f32 } else { 0.0 }).desired_width(200.0).text(crate::i18n::tr_format!(
                    "Scoring {done} of {total}",
                    done = done,
                    total = total
                )),
            );
        } else if unscored > 0 {
            let label = crate::i18n::tr_format!("Score {n} Photos", n = unscored);
            let r = crate::button::Button::new(&label).kind(crate::button::Kind::Primary).widget("button:blurScan").show(ui);
            if r.clicked()
                && let Err(e) = app.run("blur.scan", json!({"scope": scope(*selection)}))
            {
                app.ui.status = e;
            }
        } else {
            ui.label(egui::RichText::new(tr("All scored.")).color(t.text_dim));
            if ui.small_button(tr("Score Again")).clicked()
                && let Err(e) = app.run("blur.scan", json!({"scope": scope(*selection), "rescan": true}))
            {
                app.ui.status = e;
            }
        }
    });

    // the window(s)
    ui.add_space(4.0);
    ui.horizontal(|ui| {
        ui.label(egui::RichText::new(tr("Subject blur to keep:")).color(t.text_label));
        ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
            let r = ui.checkbox(&mut brackets.on, tr("By Shutter Speed"));
            crate::widgets::register(ui.ctx(), "check:blurByShutter", r.rect);
            r.on_hover_text(tr("Give each range of shutter speeds its own window: slow pans can be kept softer than fast frames."));
        });
    });
    if brackets.on {
        bracket_table(ui, brackets, *min, *max, &sc);
    }
    let shown = if brackets.on { group(&sc, brackets.active) } else { &sc };
    let hist: Vec<u64> = shown["histogram"].as_array().map(|a| a.iter().map(|v| v.as_u64().unwrap_or(0)).collect()).unwrap_or_default();
    if brackets.on {
        let which = brackets.label(brackets.active);
        ui.label(egui::RichText::new(crate::i18n::tr_format!("Window for {which}:", which = which)).color(t.text_dim).size(11.5));
    }
    // the handles move the shown bracket's window (or the one window)
    let active_row = if brackets.on { brackets.active.and_then(|i| brackets.rows.get_mut(i)) } else { None };
    let (lo, hi) = match active_row {
        Some(b) => (&mut b.min, &mut b.max),
        None => (&mut *min, &mut *max),
    };
    range_histogram(ui, &hist, lo, hi);
    ui.horizontal(|ui| {
        let r = ui.add(egui::DragValue::new(lo).range(0.0..=100.0).speed(0.5).fixed_decimals(0).prefix(format!("{} ", tr("From"))));
        crate::widgets::register(ui.ctx(), "field:blurMin", r.rect);
        let r = ui.add(egui::DragValue::new(hi).range(0.0..=100.0).speed(0.5).fixed_decimals(0).prefix(format!("{} ", tr("to"))));
        crate::widgets::register(ui.ctx(), "field:blurMax", r.rect);
        if *lo > *hi {
            std::mem::swap(lo, hi);
        }
        ui.label(egui::RichText::new(tr("0 = tack sharp · 100 = unrecognisable")).color(t.text_dim).size(11.0));
    });
    ui.horizontal(|ui| {
        for (label, a, b) in [("Sharp Only", 0.0, 20.0), ("Allow Soft", 0.0, 40.0), ("Keep Artistic Blur", 0.0, 75.0)] {
            if ui.small_button(tr(label)).clicked() {
                (*lo, *hi) = (a, b);
            }
        }
    });
    // kept for next time (the row on the histogram isn't)
    let keep = Brackets { active: None, ..brackets.clone() };
    if app.ui.blur_brackets != keep {
        app.ui.blur_brackets = keep;
    }

    // what happens
    ui.add_space(4.0);
    let pick = |ui: &mut Ui, id: &str, title: &str, items: &[(&str, &str); 3], value: &mut String| {
        ui.label(egui::RichText::new(tr(title)).color(t.text_label));
        let active = items.iter().position(|(_, k)| *k == value.as_str());
        if let Some(i) = crate::widgets::segmented(ui, id, items, active, 3)
            && let Some((_, k)) = items.get(i)
        {
            *value = (*k).to_string();
        }
    };
    let (n_in, n_out) = (sc["inRange"].as_u64().unwrap_or(0), sc["outOfRange"].as_u64().unwrap_or(0));
    pick(ui, "blurInside", &crate::i18n::tr_format!("Inside the range ({n} photos):", n = n_in), &INSIDE, inside);
    pick(ui, "blurOutside", &crate::i18n::tr_format!("Outside the range ({n} photos):", n = n_out), &OUTSIDE, outside);
    let left = unscored + sc["noDetail"].as_u64().unwrap_or(0);
    if left > 0 {
        ui.label(
            egui::RichText::new(crate::i18n::tr_format!("{n} photos without a score are left as they are.", n = left)).color(t.text_dim).size(11.5),
        );
    }
}

/// `blur.scores`' counts for bracket `i` (`None`: the photos without a known shutter speed).
fn group(sc: &Value, i: Option<usize>) -> &Value {
    match i {
        Some(i) => &sc["brackets"][i],
        None => &sc["unknownShutter"],
    }
}

/// The shutter brackets: one row each (its speeds, its window, how many of its photos fall inside and outside), and
/// one for the photos without a known shutter speed. A row's radio button puts it on the histogram; a bracket's limit
/// is chosen from the speeds between its neighbours'.
fn bracket_table(ui: &mut Ui, brackets: &mut Brackets, min: f32, max: f32, sc: &Value) {
    let t = Tokens::get(ui.ctx());
    let tr = crate::i18n::tr;
    let counts = |v: &Value| {
        let scored = v["scored"].as_u64().unwrap_or(0);
        if scored == 0 {
            tr("no scored photos").to_string()
        } else {
            crate::i18n::tr_format!("{n} in · {m} out", n = v["inRange"].as_u64().unwrap_or(0), m = v["outOfRange"].as_u64().unwrap_or(0))
        }
    };
    let n = brackets.rows.len();
    let mut remove = None;
    egui::Grid::new("blurBrackets").num_columns(4).spacing(vec2(10.0, 4.0)).show(ui, |ui| {
        for row in (0..n).map(Some).chain([None]) {
            ui.horizontal(|ui| {
                let r = crate::widgets::named(ui.radio(brackets.active == row, ""), &brackets.label(row));
                let id = row.map_or_else(|| "row:blurBracketUnknown".to_string(), |i| format!("row:blurBracket{i}"));
                crate::widgets::register(ui.ctx(), id, r.rect);
                if r.clicked() {
                    brackets.active = row;
                }
                let limited = row.and_then(|i| brackets.rows.get(i)).and_then(|b| b.up_to.clone());
                match (row, limited) {
                    (Some(i), Some(up)) => {
                        let prev = i.checked_sub(1).and_then(|p| brackets.rows.get(p)).and_then(|b| b.up_to.clone());
                        if let Some(prev) = &prev {
                            ui.label(format!("{prev} –"));
                        }
                        let choices = brackets.choices(i);
                        let c = crate::widgets::combo_box(("blurBracketLimit", i)).width(78.0).selected_text(up.as_str()).show_ui(ui, |ui| {
                            for s in choices {
                                if ui.selectable_label(s == up, s).clicked()
                                    && let Some(b) = brackets.rows.get_mut(i)
                                {
                                    b.up_to = Some(s.to_string());
                                    brackets.active = Some(i);
                                }
                            }
                        });
                        crate::widgets::register(ui.ctx(), format!("combo:blurBracketLimit{i}"), c.response.rect);
                        if prev.is_none() {
                            ui.label(tr("and faster"));
                        }
                    }
                    _ => {
                        let r = ui.add(egui::Label::new(brackets.label(row)).sense(Sense::click()));
                        if r.clicked() {
                            brackets.active = row;
                        }
                    }
                }
            });
            let (lo, hi) = row.and_then(|i| brackets.rows.get(i)).map_or((min, max), |b| (b.min, b.max));
            ui.label(egui::RichText::new(format!("{lo:.0} – {hi:.0}")).color(if brackets.active == row { t.accent } else { t.text }));
            ui.label(egui::RichText::new(counts(group(sc, row))).color(t.text_dim).size(11.5));
            if let Some(i) = row.filter(|i| i + 1 < n) {
                let r = ui.small_button("−").on_hover_text(tr("Remove this bracket"));
                crate::widgets::register(ui.ctx(), format!("button:blurBracketRemove{i}"), r.rect);
                if r.clicked() {
                    remove = Some(i);
                }
            } else {
                ui.label("");
            }
            ui.end_row();
        }
    });
    if let Some(i) = remove {
        brackets.remove(i);
    }
    ui.horizontal(|ui| {
        let room = brackets.rows.len() < MAX_BRACKETS;
        let r = ui.add_enabled(room, egui::Button::new(tr("Add Bracket")).small());
        crate::widgets::register(ui.ctx(), "button:blurBracketAdd", r.rect);
        if r.clicked() {
            brackets.add();
        }
    });
}

/// The scores' histogram with the window drawn over it; drag either edge (or click to move the nearer one).
fn range_histogram(ui: &mut Ui, hist: &[u64], min: &mut f32, max: &mut f32) {
    let t = Tokens::get(ui.ctx());
    let w = ui.available_width().max(200.0);
    let (rect, resp) = ui.allocate_exact_size(vec2(w, 100.0), Sense::click_and_drag());
    crate::widgets::register(ui.ctx(), "image:blurHistogram", rect);
    let plot = Rect::from_min_max(rect.min + vec2(8.0, 4.0), pos2(rect.max.x - 8.0, rect.max.y - 30.0));
    let to_x = |v: f32| plot.left() + v.clamp(0.0, 100.0) / 100.0 * plot.width();
    let to_v = |x: f32| ((x - plot.left()) / plot.width().max(1.0) * 100.0).clamp(0.0, 100.0).round();

    // dragging: the handle nearer the pointer when the drag starts (ties: by side) moves
    let which_id = resp.id.with("handle");
    if let Some(p) = resp.interact_pointer_pos() {
        // decided the moment the button goes down (a click's press frame comes before egui calls it a click)
        if ui.input(|i| i.pointer.any_pressed()) || resp.drag_started() {
            let (dmin, dmax) = ((p.x - to_x(*min)).abs(), (p.x - to_x(*max)).abs());
            let pick_min = if (dmin - dmax).abs() < 1.0 { p.x < to_x(*min) } else { dmin < dmax };
            ui.ctx().data_mut(|d| d.insert_temp(which_id, pick_min));
        }
        let pick_min = ui.ctx().data(|d| d.get_temp::<bool>(which_id)).unwrap_or(true);
        let v = to_v(p.x);
        if pick_min {
            *min = v.min(*max);
        } else {
            *max = v.max(*min);
        }
    }

    let painter = ui.painter_at(rect);
    painter.rect(plot.expand(2.0), CornerRadius::same(3), t.inset, Stroke::new(1.0, t.divider), StrokeKind::Inside);
    let top = hist.iter().copied().max().unwrap_or(0).max(1) as f32;
    let n = hist.len().max(1) as f32;
    for (i, c) in hist.iter().enumerate() {
        let (lo, hi) = (i as f32 * 100.0 / n, (i + 1) as f32 * 100.0 / n);
        let inside = hi > *min && lo < *max;
        let h = (*c as f32 / top).sqrt() * plot.height(); // sqrt: a few photos still show next to many
        let bar = Rect::from_min_max(pos2(to_x(lo) + 1.0, plot.bottom() - h), pos2(to_x(hi) - 1.0, plot.bottom()));
        painter.rect_filled(bar, CornerRadius::same(1), if inside { t.accent } else { t.text_disabled });
    }
    // the window
    let win = Rect::from_min_max(pos2(to_x(*min), plot.top()), pos2(to_x(*max), plot.bottom()));
    painter.rect_filled(win, CornerRadius::ZERO, t.accent.gamma_multiply(0.12));
    for (v, name) in [(*min, "slider:blurMin"), (*max, "slider:blurMax")] {
        let x = to_x(v);
        painter.line_segment([pos2(x, plot.top()), pos2(x, plot.bottom() + 4.0)], Stroke::new(2.0, t.accent));
        let knob = Pos2::new(x, plot.bottom() + 9.0);
        painter.circle(knob, 5.5, t.thumb, Stroke::new(1.0, t.accent));
        crate::widgets::register(ui.ctx(), name, Rect::from_center_size(knob, vec2(14.0, 14.0)));
    }
    let f = t.font(10.5);
    painter.text(pos2(plot.left(), rect.bottom() - 1.0), Align2::LEFT_BOTTOM, crate::i18n::tr("Sharp"), f.clone(), t.text_dim);
    painter.text(pos2(plot.right(), rect.bottom() - 1.0), Align2::RIGHT_BOTTOM, crate::i18n::tr("Blurry"), f.clone(), t.text_dim);
    painter.text(pos2(plot.center().x, rect.bottom() - 1.0), Align2::CENTER_BOTTOM, format!("{:.0} – {:.0}", min, max), f, t.text);
    if hist.iter().all(|c| *c == 0) {
        painter.text(plot.center(), Align2::CENTER_CENTER, crate::i18n::tr("No scores yet"), t.font(12.0), t.text_dim);
    }
    crate::widgets::named(resp, crate::i18n::tr("Subject blur to keep"));
}

/// Whether Apply Flags can do anything: some of the chosen photos have a score.
pub fn can_apply(app: &mut LightcraftApp, selection: bool) -> bool {
    app.session.execute("blur.scores", &json!({"scope": scope(selection), "photos": false})).is_ok_and(|v| v["scored"].as_u64().unwrap_or(0) > 0)
}

/// The dialog's OK: flag the photos.
pub fn apply(
    app: &mut LightcraftApp,
    selection: bool,
    min: f32,
    max: f32,
    inside: &str,
    outside: &str,
    brackets: &Brackets,
) -> Result<Value, String> {
    let mut p = json!({"scope": scope(selection), "min": min, "max": max, "inRange": inside, "outOfRange": outside});
    if brackets.on {
        p["brackets"] = brackets.param();
    }
    let r = app.run("blur.apply", p)?;
    app.ui.status = crate::i18n::tr_format!(
        "Blur Cull: {inside} inside the range, {outside} outside, {changed} flags changed",
        inside = r["inRange"],
        outside = r["outOfRange"],
        changed = r["changed"]
    );
    Ok(r)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn limits(b: &Brackets) -> Vec<Option<&str>> {
        b.rows.iter().map(|r| r.up_to.as_deref()).collect()
    }

    #[test]
    fn saved_brackets_are_read_leniently_and_put_in_order() {
        #[derive(Deserialize)]
        struct Saved {
            #[serde(deserialize_with = "Brackets::lenient")]
            b: Brackets,
        }
        let read = |v: Value| serde_json::from_value::<Saved>(json!({"b": v})).unwrap().b;
        // not brackets at all: the defaults
        for junk in [json!(7), json!({"rows": "x"}), json!({"on": "yes"})] {
            assert_eq!(read(junk).rows, Brackets::default().rows);
        }
        // limits that don't parse or don't get slower are dropped, the open-ended one goes last, windows are fixed up
        let b = read(json!({"on": true, "rows": [
            {"min": 0, "max": 70},
            {"upTo": "1/60", "min": 30, "max": 10},
            {"upTo": "soon", "min": 0, "max": 10},
            {"upTo": "1/500", "min": 0, "max": 10},
            {"upTo": "2\"", "min": -5, "max": 500},
        ]}));
        assert!(b.on);
        assert_eq!(limits(&b), [Some("1/60"), Some("2\""), None]);
        assert_eq!(
            [(b.rows[0].min, b.rows[0].max), (b.rows[1].min, b.rows[1].max), (b.rows[2].min, b.rows[2].max)],
            [(10.0, 30.0), (0.0, 100.0), (0.0, 70.0)]
        );
        // no open-ended bracket saved: one is added
        assert_eq!(limits(&read(json!({"rows": [{"upTo": "1/60", "min": 0, "max": 10}]}))), [Some("1/60"), None]);
        // too many: the fastest are kept
        let many: Vec<Value> = SHUTTER_STOPS.iter().map(|s| json!({"upTo": s, "min": 0, "max": 10})).collect();
        assert_eq!(read(json!({"rows": many})).rows.len(), MAX_BRACKETS);
    }

    #[test]
    fn adding_removing_and_choosing_limits_keep_the_speeds_in_order() {
        let mut b = Brackets::default();
        assert_eq!(b.label(Some(0)), "1/500 and faster");
        assert_eq!(b.label(Some(1)), "1/500 – 1/60");
        assert_eq!(b.label(Some(3)), "Slower than 1/15");
        assert_eq!(b.label(None), "Unknown shutter speed");
        // a limit is chosen between its neighbours'
        let c = b.choices(1);
        assert_eq!((c.first().copied(), c.last().copied()), (Some("1/400"), Some("1/20")));
        assert!(b.choices(0).contains(&"1/8000") && b.choices(0).contains(&"1/500") && !b.choices(0).contains(&"1/60"));
        // adding: a stop slower than the last limit, before the open-ended bracket, with its window
        assert_eq!(b.add(), Some(3));
        assert_eq!(limits(&b), [Some("1/500"), Some("1/60"), Some("1/15"), Some("1/8"), None]);
        assert_eq!(b.rows[3].max, 60.0);
        while b.add().is_some() {}
        assert_eq!(b.rows.len(), MAX_BRACKETS);
        assert_eq!(b.rows.last().and_then(|r| r.up_to.clone()), None);
        // removing: never the open-ended one; the row on the histogram follows
        b.active = Some(2);
        b.remove(0);
        assert_eq!(b.active, Some(1));
        let n = b.rows.len();
        b.remove(n - 1);
        assert_eq!(b.rows.len(), n);
        // down to the open-ended one alone: any shutter speed
        while b.rows.len() > 1 {
            b.remove(0);
        }
        assert_eq!(b.label(Some(0)), "Any shutter speed");
        assert_eq!(b.add(), Some(0));
        assert_eq!(limits(&b), [Some("1/250"), None]);
        // the slowest speed taken: nothing slower to add
        let mut slow = Brackets { rows: vec![Bracket { up_to: Some("30\"".into()), min: 0.0, max: 10.0 }], ..Brackets::default() }.normalized();
        assert_eq!(slow.add(), None);
        // what's sent: fastest first, the open-ended one without upTo
        assert_eq!(Brackets::default().param()[0], json!({"upTo": "1/500", "min": 0.0, "max": 15.0}));
        assert_eq!(Brackets::default().param()[3]["upTo"], Value::Null);
    }
}
