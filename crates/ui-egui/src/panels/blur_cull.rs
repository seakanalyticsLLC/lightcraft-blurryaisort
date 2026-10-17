//! Photo ▸ Blur Cull…: score how blurred each photo's subject is with the optional Blur Cull model (`blur.*`), pick a
//! window of blur on the histogram, and flag the photos inside it one way and the rest another. A window starting at
//! 0 is a sharp-only cull; widening it keeps intentionally soft shots too.
//!
//! The dialog asks the engine for the scores' histogram and the counts every frame (`blur.scores` with
//! `photos: false`, cheap), feeds the scan while it is open, and the background pump ([`pump`]) keeps it going after
//! the dialog closes. The scan's progress is also its row in the activity stack (✕ there stops it).

use egui::{Align2, CornerRadius, Pos2, Rect, Sense, Stroke, StrokeKind, Ui, pos2, vec2};
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
pub fn body(app: &mut LightcraftApp, ui: &mut Ui, selection: &mut bool, min: &mut f32, max: &mut f32, inside: &mut String, outside: &mut String) {
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
    let q = json!({"scope": scope(*selection), "photos": false, "min": *min, "max": *max});
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

    // the window
    ui.add_space(4.0);
    ui.label(egui::RichText::new(tr("Subject blur to keep:")).color(t.text_label));
    let hist: Vec<u64> = sc["histogram"].as_array().map(|a| a.iter().map(|v| v.as_u64().unwrap_or(0)).collect()).unwrap_or_default();
    range_histogram(ui, &hist, min, max);
    ui.horizontal(|ui| {
        let r = ui.add(egui::DragValue::new(min).range(0.0..=100.0).speed(0.5).fixed_decimals(0).prefix(format!("{} ", tr("From"))));
        crate::widgets::register(ui.ctx(), "field:blurMin", r.rect);
        let r = ui.add(egui::DragValue::new(max).range(0.0..=100.0).speed(0.5).fixed_decimals(0).prefix(format!("{} ", tr("to"))));
        crate::widgets::register(ui.ctx(), "field:blurMax", r.rect);
        if *min > *max {
            std::mem::swap(min, max);
        }
        ui.label(egui::RichText::new(tr("0 = tack sharp · 100 = unrecognisable")).color(t.text_dim).size(11.0));
    });
    ui.horizontal(|ui| {
        for (label, lo, hi) in [("Sharp Only", 0.0, 20.0), ("Allow Soft", 0.0, 40.0), ("Keep Artistic Blur", 0.0, 75.0)] {
            if ui.small_button(tr(label)).clicked() {
                (*min, *max) = (lo, hi);
            }
        }
    });

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
pub fn apply(app: &mut LightcraftApp, selection: bool, min: f32, max: f32, inside: &str, outside: &str) -> Result<Value, String> {
    let r = app.run("blur.apply", json!({"scope": scope(selection), "min": min, "max": max, "inRange": inside, "outOfRange": outside}))?;
    app.ui.status = crate::i18n::tr_format!(
        "Blur Cull: {inside} inside the range, {outside} outside, {changed} flags changed",
        inside = r["inRange"],
        outside = r["outOfRange"],
        changed = r["changed"]
    );
    Ok(r)
}
