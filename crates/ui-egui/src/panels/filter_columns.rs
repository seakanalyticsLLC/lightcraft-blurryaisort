//! The Library Filter's Metadata section: columns of the values a field takes among the photos in
//! view (Date, ISO Speed, Shutter Speed, Aperture by default), as in Lightroom. Each column starts
//! with "All (3 ISO Speeds)" and lists only values a photo in view has, with how many photos have
//! each; columns narrow from left to right, so a column lists what the ones to its left let through.
//!
//! A click chooses a value (again: back to All), Ctrl/⌘-click adds or takes out one more, Shift-click
//! a run of them; every choice goes through `library.filter`'s `columns`. The column's name opens the
//! list of fields it can show; its ≡ menu adds a column to the left or right, removes it, shows a date
//! as a year ▸ month ▸ day tree or a flat list, and sorts ascending or descending
//! (`view.filterColumns`). The values come from [`lightcraft_engine::Session::facets`].

use egui::{Rect, Sense, Stroke, pos2, vec2};
use lightcraft_catalog::facets::{self, DEVELOP_GROUP, FIELDS};
use lightcraft_catalog::{ColorLabel, ColumnFilter, CopyrightStatus, FacetValue};
use serde_json::json;

use crate::LightcraftApp;
use crate::i18n::{tr, tr_format};
use crate::icons::{Icon, paint};
use crate::state::FilterColumn;
use crate::theme::Tokens;
use crate::widgets::register;

/// Height of the section (points).
pub const HEIGHT: f32 = 168.0;
/// A column's header and rows (points).
const HEADER_H: f32 = 24.0;
const ROW_H: f32 = 19.0;
/// Columns narrower than this scroll sideways instead.
const MIN_WIDTH: f32 = 150.0;
/// Indent of a level of the date tree.
const INDENT: f32 = 14.0;

/// A value of `field` as a column lists it (and a chip names it), in the UI language; `short` for a
/// row of the date tree, under its year or month.
pub(crate) fn value_label(catalog: &lightcraft_catalog::Catalog, field: &str, v: &str, short: bool) -> String {
    if v.is_empty() {
        return tr("Unknown").to_string();
    }
    match field {
        "date" => crate::i18n::date_group_label(v, short && v.len() > 4),
        "month" => crate::i18n::date_group_label(v, false),
        "day" => {
            // `MM-DD`: the day and its month, whatever the year
            let day = v.get(3..5).and_then(|d| d.parse::<u32>().ok());
            match (v.get(..2), day) {
                (Some(m), Some(d)) => format!("{d} {}", crate::i18n::date_group_label(&format!("2000-{m}"), true)),
                _ => v.to_string(),
            }
        }
        "fileType" => tr(v).to_string(),
        "flag" => tr(match v {
            "pick" => "Picked",
            "reject" => "Rejected",
            _ => "Unflagged",
        })
        .to_string(),
        "rating" => match v.parse::<usize>() {
            Ok(0) => tr("Unrated").to_string(),
            Ok(n) => "★".repeat(n.min(5)),
            Err(_) => v.to_string(),
        },
        "label" => ColorLabel::parse(v).map_or_else(|| v.to_string(), |l| crate::i18n::color_label(catalog, l)),
        "edit" => tr(if v == "edited" { "Edited" } else { "Unedited" }).to_string(),
        "focalLength" => format!("{v} mm"),
        "shutterSpeed" => tr_format!("{} sec", v),
        "aperture" => format!("f / {v}"),
        "iso" => format!("ISO {v}"),
        "gps" => tr(if v == "yes" { "Has GPS" } else { "No GPS" }).to_string(),
        "copyrightStatus" => CopyrightStatus::parse(v).map_or_else(|| v.to_string(), |c| tr(c.label()).to_string()),
        "aspect" | "treatment" => crate::i18n::choice_text(if field == "aspect" { "aspect" } else { "treatment" }, v).to_string(),
        _ => v.to_string(),
    }
}

/// A field's name in the UI language (its id when no column shows it).
pub(crate) fn field_label(field: &str) -> &str {
    facets::field(field).map_or(field, |f| tr(f.label))
}

/// The "All" row of a column of `field` with `n` values: "All (3 ISO Speeds)".
pub(crate) fn all_label(field: &str, n: usize) -> String {
    let f = facets::field(field);
    let name = if n == 1 { f.map_or(field, |f| tr(f.label)) } else { f.map_or(field, |f| tr(f.plural)) };
    tr_format!("All ({} {})", n, name)
}

/// Choose `values` in the column of `field` (none: its All).
fn choose(app: &mut LightcraftApp, field: &str, values: Vec<String>) {
    let mut cols: Vec<ColumnFilter> = app.session.filter.columns.iter().filter(|c| c.field != field).cloned().collect();
    if !values.is_empty() {
        cols.push(ColumnFilter { field: field.to_string(), values });
    }
    let _ = app.run("library.filter", json!({"columns": cols}));
}

/// Lay the columns out as `cols` (`view.filterColumns`).
fn lay_out(app: &mut LightcraftApp, cols: &[FilterColumn]) {
    let _ = app.run("view.filterColumns", json!({"columns": crate::menus::columns_json(cols)}));
}

/// A row of a column: a value (or a year / month of the date tree), its label and photo count, how
/// deep in the tree it is, and whether it opens (`Some(open)`) to rows below it.
struct Row {
    key: String,
    label: String,
    count: usize,
    depth: u8,
    opens: Option<bool>,
}

/// The rows of a column of `field` over `values` (in their order): flat, or for a date shown as a
/// tree, each year and month with the photos under it, and the months and days of the open ones.
fn rows(app: &LightcraftApp, field: &str, flat: bool, values: &[FacetValue]) -> Vec<Row> {
    let cat = &app.session.catalog;
    let row = |key: &str, count: usize, depth: u8, opens: Option<bool>, short: bool| Row {
        key: key.to_string(),
        label: value_label(cat, field, key, short),
        count,
        depth,
        opens,
    };
    if flat || !facets::hierarchical(field) {
        return values.iter().map(|v| row(&v.value, v.count, 0, None, false)).collect();
    }
    // the tree: a day is `YYYY-MM-DD`; a chosen year or month no photo has any more is listed as it is
    let part = |v: &str, n: usize| v.get(..n).filter(|_| v.len() == 10).map(str::to_string);
    let mut out: Vec<Row> = Vec::new();
    let mut i = 0;
    while let Some(v) = values.get(i) {
        let Some(year) = part(&v.value, 4) else {
            out.push(row(&v.value, v.count, 0, None, false));
            i += 1;
            continue;
        };
        let in_year: Vec<&FacetValue> =
            values.get(i..).unwrap_or_default().iter().take_while(|x| part(&x.value, 4).as_deref() == Some(year.as_str())).collect();
        i += in_year.len();
        let open = app.ui.filter_open.contains(&year);
        out.push(row(&year, in_year.iter().map(|x| x.count).sum(), 0, Some(open), false));
        if !open {
            continue;
        }
        let mut j = 0;
        while let Some(d) = in_year.get(j) {
            let month = part(&d.value, 7).unwrap_or_default();
            let in_month: Vec<&&FacetValue> =
                in_year.get(j..).unwrap_or_default().iter().take_while(|x| part(&x.value, 7).as_deref() == Some(month.as_str())).collect();
            j += in_month.len().max(1);
            let open = app.ui.filter_open.contains(&month);
            out.push(row(&month, in_month.iter().map(|x| x.count).sum(), 1, Some(open), true));
            if open {
                out.extend(in_month.iter().map(|x| row(&x.value, x.count, 2, None, true)));
            }
        }
    }
    out
}

/// The first field no column shows (what a new column lists).
fn unused_field(cols: &[FilterColumn]) -> Option<&'static str> {
    FIELDS.iter().map(|f| f.id).find(|id| cols.iter().all(|c| c.field != *id))
}

/// Draw the Metadata section across the width of `ui`.
pub fn show(app: &mut LightcraftApp, ui: &mut egui::Ui) {
    let t = Tokens::get(ui.ctx());
    let width = ui.available_width();
    let (area, _) = ui.allocate_exact_size(vec2(width, HEIGHT), Sense::hover());
    ui.painter().rect_filled(area, 0.0, t.chrome);
    ui.painter().line_segment([area.left_bottom(), area.right_bottom()], Stroke::new(1.0, t.divider));
    register(ui.ctx(), "filter:metadata", area);
    // (columns of fields this build doesn't know — a settings file from a newer one — are left out)
    let mut cols: Vec<FilterColumn> = app.ui.filter_columns.iter().filter(|c| facets::field(&c.field).is_some()).cloned().collect();
    if cols.is_empty() {
        cols = FilterColumn::defaults();
    }
    let asked: Vec<(String, bool)> = cols.iter().map(|c| (c.field.clone(), c.descending)).collect();
    let found = app.session.facets(&asked);
    let col_w = (width / cols.len() as f32).max(MIN_WIDTH);
    let mut child = ui.new_child(egui::UiBuilder::new().max_rect(area).layout(egui::Layout::left_to_right(egui::Align::Min)));
    egui::ScrollArea::horizontal().id_salt("filter-columns").auto_shrink([false, false]).show(&mut child, |ui| {
        ui.spacing_mut().item_spacing = vec2(0.0, 0.0);
        for (i, col) in cols.iter().enumerate() {
            let (r, _) = ui.allocate_exact_size(vec2(col_w, HEIGHT - 1.0), Sense::hover());
            let facet = found.get(i).filter(|f| f.field == col.field);
            column(app, ui, r, i, &cols, facet);
        }
    });
}

/// One column at `r`: its header and its rows.
fn column(app: &mut LightcraftApp, ui: &mut egui::Ui, r: Rect, i: usize, cols: &[FilterColumn], facet: Option<&lightcraft_engine::Facet>) {
    let t = Tokens::get(ui.ctx());
    let Some(col) = cols.get(i) else { return };
    let field = col.field.as_str();
    if i > 0 {
        ui.painter().line_segment([r.left_top(), r.left_bottom()], Stroke::new(1.0, t.divider));
    }
    // the header: the field's name (its menu of fields) and ≡ (the column's menu)
    let header = Rect::from_min_size(r.min, vec2(r.width(), HEADER_H));
    ui.painter().rect_filled(header, 0.0, t.hover);
    let lines = Rect::from_center_size(pos2(header.right() - 14.0, header.center().y), vec2(18.0, 18.0));
    let name_rect = Rect::from_min_max(header.min, pos2(lines.left() - 2.0, header.bottom()));
    let name = ui.interact(name_rect, ui.id().with(("filtercol-name", field)), Sense::click());
    crate::widgets::announce(&name, field_label(field));
    register(ui.ctx(), format!("filtercol:{field}:header"), name_rect);
    let g = crate::widgets::one_line(
        ui.painter(),
        field_label(field),
        t.semibold(12.5),
        if name.hovered() { t.text } else { t.text_label },
        name_rect.width() - 28.0,
    );
    let at = pos2(name_rect.left() + 8.0, header.center().y - g.size().y / 2.0);
    let text_w = g.size().x;
    ui.painter().galley(at, g, t.text_label);
    paint(ui.painter(), Rect::from_center_size(pos2(at.x + text_w + 9.0, header.center().y), vec2(11.0, 11.0)), Icon::ChevronDown, t.text_dim);
    egui::Popup::menu(&name).show(|ui| field_menu(app, ui, i, cols));
    let menu = ui.interact(lines, ui.id().with(("filtercol-menu", field)), Sense::click());
    crate::widgets::announce(&menu, tr("Column options"));
    register(ui.ctx(), format!("filtercol:{field}:menu"), lines);
    let c = if menu.hovered() { t.text } else { t.icon };
    for k in [-4.0, 0.0, 4.0] {
        let y = lines.center().y + k;
        ui.painter().line_segment([pos2(lines.left() + 3.0, y), pos2(lines.right() - 3.0, y)], Stroke::new(1.3, c));
    }
    egui::Popup::menu(&menu).show(|ui| column_menu(app, ui, i, cols));

    // the rows: All, then the values
    let body = Rect::from_min_max(pos2(r.left(), header.bottom()), r.max);
    let values: &[FacetValue] = facet.map_or(&[], |f| f.values.as_slice());
    let chosen: Vec<String> = app.session.filter.columns.iter().find(|c| c.field == field).map(|c| c.values.clone()).unwrap_or_default();
    let listed = values.iter().filter(|v| v.count > 0).count();
    let all = Row { key: String::new(), label: all_label(field, listed), count: facet.map_or(0, |f| f.photos), depth: 0, opens: None };
    let rows = rows(app, field, col.flat, values);
    let order: Vec<String> = rows.iter().map(|r| r.key.clone()).collect();
    let mut child = ui.new_child(egui::UiBuilder::new().max_rect(body).layout(egui::Layout::top_down(egui::Align::Min)));
    egui::ScrollArea::vertical().id_salt(("filtercol-rows", field)).auto_shrink([false, false]).show(&mut child, |ui| {
        ui.spacing_mut().item_spacing = vec2(0.0, 0.0);
        if row(app, ui, field, &all, chosen.is_empty(), true) {
            choose(app, field, Vec::new());
        }
        for x in &rows {
            if row(app, ui, field, x, chosen.contains(&x.key), false) {
                click(app, ui, field, &x.key, &chosen, &order);
            }
        }
    });
}

/// Draw one row; whether it was clicked (to choose it: a click on its triangle opens or folds it).
fn row(app: &mut LightcraftApp, ui: &mut egui::Ui, field: &str, x: &Row, selected: bool, all: bool) -> bool {
    let t = Tokens::get(ui.ctx());
    let w = ui.available_width();
    let (r, resp) = ui.allocate_exact_size(vec2(w, ROW_H), Sense::click());
    crate::widgets::announce_selected(&resp, &x.label, selected);
    let id = if all { format!("filtercol:{field}:all") } else { format!("filtercol:{field}:{}", x.key) };
    register(ui.ctx(), id, r);
    if selected {
        ui.painter().rect_filled(r, 0.0, t.tool_active);
    } else if resp.hovered() {
        ui.painter().rect_filled(r, 0.0, t.hover);
    }
    let left = r.left() + 8.0 + f32::from(x.depth) * INDENT;
    let tri = Rect::from_center_size(pos2(left + 5.0, r.center().y), vec2(14.0, ROW_H));
    if let Some(open) = x.opens {
        register(ui.ctx(), format!("filtercol:{field}:open:{}", x.key), tri);
        paint(
            ui.painter(),
            Rect::from_center_size(tri.center(), vec2(10.0, 10.0)),
            if open { Icon::ChevronDown } else { Icon::ChevronRight },
            t.text_dim,
        );
    }
    let text_left = left + if x.opens.is_some() || x.depth > 0 { 14.0 } else { 0.0 };
    let count = format!("{}", x.count);
    let cg = ui.painter().layout_no_wrap(count, t.font(11.5), t.text_dim);
    let cw = cg.size().x;
    ui.painter().galley(pos2(r.right() - 10.0 - cw, r.center().y - cg.size().y / 2.0), cg, t.text_dim);
    let color = if selected { t.text } else { t.text_label };
    let g = crate::widgets::one_line(ui.painter(), &x.label, t.font(12.0), color, (r.right() - 18.0 - cw - text_left).max(10.0));
    ui.painter().galley(pos2(text_left, r.center().y - g.size().y / 2.0), g, color);
    if !resp.clicked() {
        return false;
    }
    if x.opens.is_some() && resp.interact_pointer_pos().is_some_and(|p| tri.contains(p)) {
        if !app.ui.filter_open.remove(&x.key) {
            app.ui.filter_open.insert(x.key.clone());
        }
        return false;
    }
    true
}

/// A click on the row of `key`: alone, choose it (again: back to All); with Ctrl/⌘, add it or take it
/// out; with Shift, the run of rows from the one clicked last.
fn click(app: &mut LightcraftApp, ui: &egui::Ui, field: &str, key: &str, chosen: &[String], order: &[String]) {
    let mods = ui.input(|i| i.modifiers);
    let anchor_id = egui::Id::new(("filtercol-anchor", field.to_string()));
    let anchor: Option<String> = ui.data(|d| d.get_temp(anchor_id));
    let next: Vec<String> = if mods.shift
        && let Some(a) = anchor.as_deref()
        && let (Some(from), Some(to)) = (order.iter().position(|k| k == a), order.iter().position(|k| k == key))
    {
        let (lo, hi) = (from.min(to), from.max(to));
        let run = order.get(lo..=hi).unwrap_or_default().iter().cloned();
        if mods.command {
            let mut v = chosen.to_vec();
            v.extend(run.filter(|k| !chosen.contains(k)));
            v
        } else {
            run.collect()
        }
    } else if mods.command {
        if chosen.iter().any(|k| k == key) {
            chosen.iter().filter(|k| *k != key).cloned().collect()
        } else {
            chosen.iter().cloned().chain(std::iter::once(key.to_string())).collect()
        }
    } else if chosen.len() == 1 && chosen.first().is_some_and(|k| k == key) {
        Vec::new()
    } else {
        vec![key.to_string()]
    };
    if !mods.shift {
        ui.data_mut(|d| d.insert_temp(anchor_id, key.to_string()));
    }
    choose(app, field, next);
}

/// The fields column `i` can show, grouped as Lightroom lists them (the Develop ones in a submenu); a
/// field another column shows can't be chosen twice.
fn field_menu(app: &mut LightcraftApp, ui: &mut egui::Ui, i: usize, cols: &[FilterColumn]) {
    let current = cols.get(i).map(|c| c.field.clone()).unwrap_or_default();
    let mut chosen: Option<&'static str> = None;
    let mut item = |ui: &mut egui::Ui, f: &'static facets::Field| {
        let elsewhere = cols.iter().enumerate().any(|(j, c)| j != i && c.field == f.id);
        let r = ui.add_enabled(!elsewhere, egui::Button::selectable(current == f.id, tr(f.label)));
        register(ui.ctx(), format!("filtercol:field:{}", f.id), r.rect);
        if r.clicked() && current != f.id {
            chosen = Some(f.id);
        }
    };
    let mut group = None;
    for f in FIELDS.iter().filter(|f| f.group != DEVELOP_GROUP) {
        if group.is_some_and(|g| g != f.group) {
            ui.separator();
        }
        group = Some(f.group);
        item(ui, f);
    }
    ui.separator();
    ui.menu_button(tr("Develop"), |ui| {
        for f in FIELDS.iter().filter(|f| f.group == DEVELOP_GROUP) {
            item(ui, f);
        }
    });
    if let Some(f) = chosen {
        let mut next = cols.to_vec();
        if let Some(c) = next.get_mut(i) {
            *c = FilterColumn::of(f);
        }
        lay_out(app, &next);
        ui.close();
    }
}

/// Column `i`'s ≡ menu: add a column to its left or right, remove it, a date as a tree or a flat
/// list, and the order of its values.
fn column_menu(app: &mut LightcraftApp, ui: &mut egui::Ui, i: usize, cols: &[FilterColumn]) {
    let t = Tokens::get(ui.ctx());
    let Some(col) = cols.get(i) else { return };
    let caption = |ui: &mut egui::Ui, text: &str| {
        ui.label(egui::RichText::new(format!("{}:", tr(text))).color(t.text_dim));
    };
    let mut next: Option<Vec<FilterColumn>> = None;
    caption(ui, "Add Column");
    let new = unused_field(cols);
    for (label, at) in [("Left", i), ("Right", i + 1)] {
        let r = ui.add_enabled(new.is_some(), egui::Button::new(format!("   {}", tr(label))));
        register(ui.ctx(), format!("filtercol:{}:add{label}", col.field), r.rect);
        if r.clicked()
            && let Some(f) = new
        {
            let mut v = cols.to_vec();
            v.insert(at.min(v.len()), FilterColumn::of(f));
            next = Some(v);
        }
    }
    let r = ui.add_enabled(cols.len() > 1, egui::Button::new(tr("Remove this Column")));
    register(ui.ctx(), format!("filtercol:{}:remove", col.field), r.rect);
    if r.clicked() {
        let mut v = cols.to_vec();
        v.remove(i);
        next = Some(v);
    }
    ui.separator();
    caption(ui, "View");
    let tree = facets::hierarchical(&col.field);
    for (label, flat) in [("Hierarchical", false), ("Flat", true)] {
        let on = if tree { col.flat == flat } else { flat };
        let r = ui.add_enabled(tree, egui::Button::selectable(on, format!("   {}", tr(label))));
        register(ui.ctx(), format!("filtercol:{}:{}", col.field, label.to_lowercase()), r.rect);
        if r.clicked() && col.flat != flat {
            let mut v = cols.to_vec();
            if let Some(c) = v.get_mut(i) {
                c.flat = flat;
            }
            next = Some(v);
        }
    }
    ui.separator();
    caption(ui, "Sort");
    for (label, descending) in [("Ascending", false), ("Descending", true)] {
        let r = ui.add(egui::Button::selectable(col.descending == descending, format!("   {}", tr(label))));
        register(ui.ctx(), format!("filtercol:{}:{}", col.field, label.to_lowercase()), r.rect);
        if r.clicked() && col.descending != descending {
            let mut v = cols.to_vec();
            if let Some(c) = v.get_mut(i) {
                c.descending = descending;
            }
            next = Some(v);
        }
    }
    if let Some(v) = next {
        lay_out(app, &v);
        ui.close();
    }
}

/// What a chip of the metadata columns reads in the UI language ("ISO Speed: ISO 100 or ISO 200"),
/// from the column it lets go of; `None` for a chip of anything else.
pub(crate) fn chip_label(chip_field: &str, filter: &lightcraft_catalog::Filter, catalog: &lightcraft_catalog::Catalog) -> Option<String> {
    let c = filter.columns.iter().find(|c| c.field == chip_field)?;
    let values: Vec<String> = c.values.iter().map(|v| value_label(catalog, &c.field, v, false)).collect();
    Some(format!("{}: {}", field_label(&c.field), values.join(&format!(" {} ", tr("or")))))
}
