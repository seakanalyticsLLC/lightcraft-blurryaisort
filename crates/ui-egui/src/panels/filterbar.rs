//! The Library Filter above the grid, as in Lightroom: a header whose Attribute, Metadata and None
//! choose the sections below it (`view.filterSections`), then the Attribute row — rating
//! (≥ / = / ≤), flags, colour labels, kind and edited state — and the Metadata columns
//! ([`super::filter_columns`]). Every control drives `library.filter`. In the header, "Save as Smart
//! Album…" keeps the current view as a live album, "Clear" resets the filter and Presets applies or
//! saves one.

use egui::{Color32, Rect, Sense, Stroke, StrokeKind, vec2};
use lightcraft_catalog::{ColorLabel, Flag, MediaKind, RatingOp};
use serde_json::{Value, json};

use crate::LightcraftApp;
use crate::icons::{Icon, paint};
use crate::theme::Tokens;
use crate::widgets::register;

pub const HEIGHT: f32 = 40.0;
/// The header's height (points).
const HEADER: f32 = 30.0;
/// Bar width (points) below which the Attribute controls take two rows.
const WIDE: f32 = 760.0;

/// Colour-label swatches (UI colours, our own choice).
pub use crate::theme::label_color;

fn filter(app: &mut LightcraftApp, patch: Value) {
    let _ = app.run("library.filter", patch);
}

/// A small label before a group of controls.
fn caption(ui: &mut egui::Ui, text: &str) {
    let t = Tokens::get(ui.ctx());
    ui.label(egui::RichText::new(crate::i18n::tr(text)).font(t.font(12.0)).color(t.text_dim));
}

/// A square toggle with an icon or a colour swatch.
fn toggle(ui: &mut egui::Ui, id: &str, on: bool, tip: &str, draw: impl FnOnce(&egui::Painter, Rect, Color32)) -> egui::Response {
    let t = Tokens::get(ui.ctx());
    let (r, resp) = ui.allocate_exact_size(vec2(26.0, 26.0), Sense::click());
    crate::widgets::announce_selected(&resp, crate::i18n::tr(tip), on);
    register(ui.ctx(), format!("filter:{id}"), r);
    let p = ui.painter();
    if on {
        p.rect_filled(r, 5.0, t.tool_active);
        p.rect_stroke(r, 5.0, Stroke::new(1.0, t.accent), StrokeKind::Inside);
    } else if resp.hovered() {
        p.rect_filled(r, 5.0, t.hover);
    }
    draw(
        p,
        r.shrink(5.0),
        if on {
            t.text
        } else if resp.hovered() {
            t.text_label
        } else {
            t.icon
        },
    );
    resp.on_hover_text(crate::i18n::tr(tip))
}

/// A dropdown showing `current`; `items` are (label, filter patch, selected).
fn picker(app: &mut LightcraftApp, ui: &mut egui::Ui, id: &str, current: &str, active: bool, items: Vec<(String, Value, bool)>) {
    let t = Tokens::get(ui.ctx());
    let r = crate::widgets::dropdown(ui, &format!("filter-{id}"), crate::i18n::tr(current), t.font(12.5), if active { t.text } else { t.text_label });
    if active {
        ui.painter().rect_stroke(r.rect.expand(2.0), 4.0, Stroke::new(1.0, t.accent.gamma_multiply(0.7)), StrokeKind::Outside);
    }
    let mut chosen = None;
    egui::Popup::menu(&r).show(|ui| {
        egui::ScrollArea::vertical().max_height(320.0).show(ui, |ui| {
            for (label, patch, sel) in &items {
                if ui.selectable_label(*sel, crate::i18n::tr(label)).clicked() {
                    chosen = Some(patch.clone());
                }
            }
        });
    });
    if let Some(p) = chosen {
        filter(app, p);
    }
}

/// The Library Filter: its header, and the sections it shows.
pub fn show(app: &mut LightcraftApp, ui: &mut egui::Ui) {
    let top = ui.cursor().top();
    header(app, ui);
    if app.ui.filter_attribute {
        attribute(app, ui);
    }
    if app.ui.filter_metadata {
        super::filter_columns::show(app, ui);
    }
    let bar = Rect::from_min_max(egui::pos2(ui.max_rect().left(), top), egui::pos2(ui.max_rect().right(), ui.cursor().top()));
    register(ui.ctx(), "filterbar", bar);
}

/// The header: "Library Filter", the section switches, and the save / clear / presets actions.
fn header(app: &mut LightcraftApp, ui: &mut egui::Ui) {
    let t = Tokens::get(ui.ctx());
    let (bar, _) = ui.allocate_exact_size(vec2(ui.available_width(), HEADER), Sense::hover());
    ui.painter().rect_filled(bar, 0.0, t.chrome);
    ui.painter().line_segment([bar.left_bottom(), bar.right_bottom()], Stroke::new(1.0, t.divider));
    register(ui.ctx(), "filter:header", bar);
    let mut ui = ui.new_child(egui::UiBuilder::new().max_rect(bar.shrink2(vec2(16.0, 2.0))).layout(egui::Layout::left_to_right(egui::Align::Center)));
    ui.spacing_mut().item_spacing.x = 4.0;
    ui.label(egui::RichText::new(format!("{} :", crate::i18n::tr("Library Filter"))).font(t.font(12.5)).color(t.text_dim));
    ui.add_space(12.0);
    let (attribute, metadata) = (app.ui.filter_attribute, app.ui.filter_metadata);
    for (id, label, on, params) in [
        ("button:filterAttribute", "Attribute", attribute, json!({"attribute": !attribute})),
        ("button:filterMetadata", "Metadata", metadata, json!({"metadata": !metadata})),
        ("button:filterNone", "None", !attribute && !metadata, json!({"attribute": false, "metadata": false})),
    ] {
        let color = if on { t.text } else { t.text_dim };
        let r = ui.add(egui::Button::selectable(on, egui::RichText::new(crate::i18n::tr(label)).font(t.font(12.5)).color(color)));
        register(ui.ctx(), id, r.rect);
        if r.clicked() {
            let _ = app.run("view.filterSections", params);
        }
    }
    // save / clear / presets, at the right
    let mut ui = ui.new_child(egui::UiBuilder::new().max_rect(ui.max_rect()).layout(egui::Layout::right_to_left(egui::Align::Center)));
    ui.spacing_mut().item_spacing.x = 4.0;
    let ui = &mut ui;
    let filtering = app.session.filter != Default::default();
    // filter presets: apply one, save the current filter, delete
    let presets = crate::widgets::dropdown(ui, "filterPresets", crate::i18n::tr("Presets"), t.font(12.5), t.text);
    register(ui.ctx(), "button:filterPresets", presets.rect);
    egui::Popup::menu(&presets).show(|ui| {
        let names: Vec<String> = app.session.filter_presets.iter().map(|f| f.name.clone()).collect();
        for n in &names {
            let r = ui.button(n);
            if r.clicked() {
                let _ = app.run("filter.applyPreset", json!({"name": n}));
            }
            r.context_menu(|ui| {
                if ui.button(crate::i18n::tr("Delete Preset")).clicked() {
                    let _ = app.run("filter.deletePreset", json!({"name": n}));
                }
            });
        }
        if names.is_empty() {
            ui.label(egui::RichText::new(crate::i18n::tr("No filter presets yet")).color(t.text_dim));
        }
        ui.separator();
        if ui.add_enabled(filtering, egui::Button::new(crate::i18n::tr("Save Current Filter as Preset…"))).clicked() {
            crate::panels::dialogs::prompt(app, "Save Filter Preset", "Preset name", "", "filter.savePreset", json!({}), "name");
        }
    });
    let clear = ui.add_enabled(filtering, egui::Button::new(egui::RichText::new(crate::i18n::tr("Clear")).font(t.font(12.5))));
    register(ui.ctx(), "button:filterClear", clear.rect);
    if clear.clicked() {
        app.ui.search.clear();
        let _ = app.run("library.clearFilter", json!({}));
    }
    let save = ui.add_enabled(filtering, egui::Button::new(egui::RichText::new(crate::i18n::tr("Save as Smart Album…")).font(t.font(12.5))));
    register(ui.ctx(), "button:filterSave", save.rect);
    if save.clicked() {
        let _ = app.run("dialog.newSmartAlbum", json!({}));
    }
}

/// The Attribute row: rating, flags, colour labels, kind and edited state.
fn attribute(app: &mut LightcraftApp, ui: &mut egui::Ui) {
    let t = Tokens::get(ui.ctx());
    // one row when there is room, else kind and edited go to a second row
    let two_rows = ui.available_width() < WIDE;
    let (bar, _) = ui.allocate_exact_size(vec2(ui.available_width(), if two_rows { 2.0 * HEIGHT - 8.0 } else { HEIGHT }), Sense::hover());
    ui.painter().rect_filled(bar, 0.0, t.chrome);
    ui.painter().line_segment([bar.left_bottom(), bar.right_bottom()], Stroke::new(1.0, t.divider));
    register(ui.ctx(), "filter:attribute", bar);
    let f = app.session.filter.clone();
    let row = |ui: &mut egui::Ui, i: f32| {
        let r = Rect::from_min_size(bar.min + vec2(16.0, 4.0 + i * (HEIGHT - 8.0)), vec2(bar.width() - 32.0, HEIGHT - 8.0));
        let mut c = ui.new_child(egui::UiBuilder::new().max_rect(r).layout(egui::Layout::left_to_right(egui::Align::Center)));
        c.spacing_mut().item_spacing.x = 4.0;
        c
    };
    let mut row1 = row(ui, 0.0);
    let mut row2 = row(ui, 1.0);
    let ui = &mut row1;

    // rating: operator + stars
    caption(ui, "Rating");
    let op_label = match f.rating_op {
        RatingOp::AtLeast => "≥",
        RatingOp::Exactly => "=",
        RatingOp::AtMost => "≤",
    };
    picker(
        app,
        ui,
        "ratingOp",
        op_label,
        false,
        [("≥  at least", "atLeast", RatingOp::AtLeast), ("=  exactly", "exactly", RatingOp::Exactly), ("≤  at most", "atMost", RatingOp::AtMost)]
            .into_iter()
            .map(|(l, k, op)| (l.to_string(), json!({"ratingOp": k}), f.rating_op == op))
            .collect(),
    );
    for i in 1..=5u8 {
        let (r, resp) = ui.allocate_exact_size(vec2(17.0, 26.0), Sense::click());
        crate::widgets::announce(&resp, format!("{i} ★"));
        register(ui.ctx(), format!("filter:star{i}"), r);
        let filled = i <= f.rating;
        let c = if filled {
            t.star
        } else if resp.hovered() {
            t.text_label
        } else {
            t.icon
        };
        paint(ui.painter(), Rect::from_center_size(r.center(), vec2(14.0, 14.0)), if filled { Icon::StarFilled } else { Icon::Star }, c);
        if resp.clicked() {
            filter(app, json!({"rating": if f.rating == i { 0 } else { i }}));
        }
    }
    ui.add_space(14.0);

    // flags (one at a time; click again to clear)
    caption(ui, "Flag");
    for (id, icon, flag, tip) in [
        ("pick", Icon::FlagPick, Flag::Pick, "Picked"),
        ("reject", Icon::FlagReject, Flag::Reject, "Rejected"),
        ("none", Icon::Circle, Flag::None, "Unflagged"),
    ] {
        let on = f.flag == Some(flag);
        let color = match flag {
            Flag::Pick if on => t.pick,
            Flag::Reject if on => t.reject,
            _ => Color32::TRANSPARENT,
        };
        let resp = toggle(ui, id, on, tip, |p, r, c| paint(p, r, icon, if color == Color32::TRANSPARENT { c } else { color }));
        if resp.clicked() {
            filter(app, json!({"flag": if on { Value::Null } else { json!(id) }}));
        }
    }
    ui.add_space(14.0);

    // colour labels
    caption(ui, "Label");
    // several labels can be on at once (any of them matches)
    let mut chosen: Vec<ColorLabel> = f.labels.clone();
    if let Some(l) = f.label
        && !chosen.contains(&l)
    {
        chosen.push(l);
    }
    for l in ColorLabel::ALL {
        let on = chosen.contains(&l);
        let name = format!("{l:?}").to_lowercase();
        let tip = crate::i18n::tr_format!("{} label (click more labels to show any of them)", crate::i18n::color_label(&app.session.catalog, l));
        let resp = toggle(ui, &format!("label-{name}"), on, &tip, |p, r, _| {
            p.circle_filled(r.center(), 5.5, label_color(l));
        });
        if resp.clicked() {
            let mut next = chosen.clone();
            if on {
                next.retain(|x| *x != l);
            } else {
                next.push(l);
            }
            next.sort();
            let names: Vec<String> = next.iter().map(|x| format!("{x:?}").to_lowercase()).collect();
            filter(app, json!({"label": Value::Null, "labels": names}));
        }
    }
    ui.add_space(14.0);

    // kind + edited
    let ui: &mut egui::Ui = if two_rows { &mut row2 } else { &mut row1 };
    let kind_label = match f.kind {
        None => "All types",
        Some(MediaKind::Image) => "Photos",
        Some(MediaKind::Raw) => "Raw",
        Some(MediaKind::Video) => "Videos",
    };
    let kind_label = match f.merged.as_deref() {
        Some("hdr") => "HDR",
        Some("panorama") => "Panoramas",
        Some("hdrPanorama") => "HDR Panoramas",
        Some(_) => "Merged",
        None => kind_label,
    };
    picker(
        app,
        ui,
        "kind",
        kind_label,
        f.kind.is_some() || f.merged.is_some(),
        [
            ("All types", Value::Null, None),
            ("Photos", json!("image"), Some(MediaKind::Image)),
            ("Raw", json!("raw"), Some(MediaKind::Raw)),
            ("Videos", json!("video"), Some(MediaKind::Video)),
        ]
        .into_iter()
        .map(|(l, v, k)| (l.to_string(), json!({"kind": v, "merged": null}), f.kind == k && f.merged.is_none()))
        .chain(
            [("HDR", "hdr"), ("Panoramas", "panorama"), ("HDR Panoramas", "hdrPanorama")]
                .into_iter()
                .map(|(l, m)| (l.to_string(), json!({"kind": null, "merged": m}), f.merged.as_deref() == Some(m))),
        )
        .collect(),
    );
    let edited_label = match f.edited {
        None => "Edited or not",
        Some(true) => "Edited",
        Some(false) => "Unedited",
    };
    picker(
        app,
        ui,
        "edited",
        edited_label,
        f.edited.is_some(),
        [("Edited or not", Value::Null, None), ("Edited", json!(true), Some(true)), ("Unedited", json!(false), Some(false))]
            .into_iter()
            .map(|(l, v, e)| (l.to_string(), json!({"edited": v}), f.edited == e))
            .collect(),
    );
}
