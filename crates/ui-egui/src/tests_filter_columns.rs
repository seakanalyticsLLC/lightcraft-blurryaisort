//! Headless tests of the Library Filter's header and Metadata columns: Attribute / Metadata / None,
//! the columns' values and counts, choosing values (click, Ctrl/⌘-click, Shift-click, All), columns
//! narrowing from left to right, the date tree, the column menus, chips, and bad requests.

use std::time::Duration;

use lightcraft_catalog::{ColumnFilter, Op, PhotoId};
use serde_json::{Value, json};

use crate::headless::Headless;
use crate::{LightcraftApp, Services};

const T: Duration = Duration::from_secs(20);
const SETTLE: Duration = Duration::from_secs(120);
const ISOS: [u32; 3] = [100, 400, 1600];
const SHUTTERS: [&str; 2] = ["1/250", "1/60"];

/// The demo library in Photo Grid with the Library Filter open; every photo given an ISO (100,
/// 400, 1600 in turn), a shutter speed (1/250, 1/60 in turn) and a capture date (the first half in
/// August 2016, the rest in January 2017). Returns the photos in library order.
fn grid() -> (Headless, Vec<PhotoId>) {
    let mut app = LightcraftApp::new(lightcraft_engine::Session::with_demo(), Services { png: None, ..Default::default() });
    app.ui.settings.gpu = false;
    let mut h = Headless::new(app, [1500.0, 950.0], 1.0);
    let ids = h.app.session.visible_cloned();
    let half = ids.len() / 2;
    for (i, id) in ids.iter().enumerate() {
        let p = h.app.session.catalog.photo(*id).expect("photo").clone();
        let mut meta = p.meta.clone();
        meta.iso = Some(ISOS[i % 3]);
        meta.shutter = SHUTTERS[i % 2].into();
        h.app.session.catalog.apply(Op::SetMeta { id: *id, meta: Box::new(meta) }).expect("set meta");
        let captured = Some(if i < half { format!("2016-08-{:02}T10:00:00", 1 + i % 3) } else { "2017-01-05T10:00:00".into() });
        h.app.session.catalog.apply(Op::SetCaptured { id: *id, captured }).expect("set date");
    }
    exec(&mut h, "view.photoGrid", json!({}));
    exec(&mut h, "view.filterBar", json!({}));
    h.settle(SETTLE);
    (h, ids)
}

fn exec(h: &mut Headless, cmd: &str, params: Value) -> Value {
    let r = h.request("engine.execute", json!({"command": cmd, "params": params}), T);
    assert_eq!(r["ok"], true, "{cmd}: {r}");
    r["result"].clone()
}

fn click(h: &mut Headless, id: &str, extra: Value) {
    let mut p = json!({"id": id});
    if let (Some(o), Some(e)) = (p.as_object_mut(), extra.as_object()) {
        o.extend(e.clone());
    }
    let r = h.request("ui.clickWidget", p, T);
    assert_eq!(r["ok"], true, "{id}: {r}");
    for _ in 0..3 {
        h.step();
    }
}

fn shown(h: &Headless, id: &str) -> bool {
    h.app.widgets.iter().any(|(w, _)| w == id)
}

fn chosen(h: &Headless, field: &str) -> Vec<String> {
    h.app.session.filter.columns.iter().find(|c| c.field == field).map(|c| c.values.clone()).unwrap_or_default()
}

/// How many photos of `ids` have ISO `iso` (by the order they were given ISOs in).
fn with_iso(ids: &[PhotoId], iso: u32) -> usize {
    (0..ids.len()).filter(|i| ISOS[i % 3] == iso).count()
}

#[test]
fn the_columns_list_each_value_once_with_its_count() {
    let (mut h, ids) = grid();
    for w in [
        "filter:header",
        "filter:attribute",
        "filter:metadata",
        "filtercol:date:header",
        "filtercol:iso:all",
        "filtercol:iso:100",
        "filtercol:shutterSpeed:1/60",
        "filtercol:aperture:menu",
    ] {
        assert!(shown(&h, w), "{w} is on screen");
    }
    assert!(!shown(&h, "filter-camera") && !shown(&h, "dropdown:filter-camera"), "the Camera picker left the Attribute row");
    let facets = exec(&mut h, "library.facets", json!({"columns": [{"field": "iso"}, {"field": "shutterSpeed", "descending": true}]}));
    let iso: Vec<(String, u64)> =
        facets[0]["values"].as_array().unwrap().iter().map(|v| (v["value"].as_str().unwrap().to_string(), v["count"].as_u64().unwrap())).collect();
    let want: Vec<(String, u64)> = ISOS.iter().map(|i| (i.to_string(), with_iso(&ids, *i) as u64)).collect();
    assert_eq!(iso, want, "every ISO in view, ascending, with its photos");
    assert_eq!(facets[0]["photos"], ids.len());
    let shutters: Vec<&str> = facets[1]["values"].as_array().unwrap().iter().map(|v| v["value"].as_str().unwrap()).collect();
    assert_eq!(shutters, ["1/60", "1/250"], "descending: the longest first");
    // the All row's wording
    assert_eq!(crate::panels::filter_columns::all_label("iso", 3), "All (3 ISO Speeds)");
    assert_eq!(crate::panels::filter_columns::all_label("date", 1), "All (1 Date)");
    h.settle(SETTLE);
}

#[test]
fn a_click_chooses_ctrl_adds_shift_takes_a_run_and_all_clears() {
    let (mut h, ids) = grid();
    let all = ids.len();
    click(&mut h, "filtercol:iso:100", json!({}));
    assert_eq!(chosen(&h, "iso"), ["100"]);
    assert_eq!(h.app.session.visible_cloned().len(), with_iso(&ids, 100));
    click(&mut h, "filtercol:iso:100", json!({}));
    assert!(chosen(&h, "iso").is_empty(), "a click on the one chosen goes back to All");
    assert_eq!(h.app.session.visible_cloned().len(), all);
    click(&mut h, "filtercol:iso:100", json!({}));
    click(&mut h, "filtercol:iso:1600", json!({"cmd": true}));
    assert_eq!(chosen(&h, "iso"), ["100", "1600"], "Ctrl/⌘-click adds");
    assert_eq!(h.app.session.visible_cloned().len(), with_iso(&ids, 100) + with_iso(&ids, 1600));
    click(&mut h, "filtercol:iso:100", json!({"cmd": true}));
    assert_eq!(chosen(&h, "iso"), ["1600"], "and takes out");
    click(&mut h, "filtercol:iso:100", json!({}));
    click(&mut h, "filtercol:iso:1600", json!({"shift": true}));
    assert_eq!(chosen(&h, "iso"), ["100", "1600", "400"], "Shift-click takes the run between");
    click(&mut h, "filtercol:iso:all", json!({}));
    assert!(h.app.session.filter.columns.is_empty(), "All lets go of the column");
    h.settle(SETTLE);
}

#[test]
fn columns_narrow_from_left_to_right() {
    let (mut h, ids) = grid();
    // ISO is left of Shutter Speed: choosing an ISO narrows the shutter column, not the other way
    click(&mut h, "filtercol:shutterSpeed:1/250", json!({}));
    let f = h.app.session.facets(&[("iso".into(), false), ("shutterSpeed".into(), false)]);
    let iso_total: usize = f[0].values.iter().map(|v| v.count).sum();
    assert_eq!(iso_total, ids.len(), "a column to the right doesn't narrow the ISO column");
    click(&mut h, "filtercol:shutterSpeed:1/250", json!({}));
    click(&mut h, "filtercol:iso:400", json!({}));
    let f = h.app.session.facets(&[("iso".into(), false), ("shutterSpeed".into(), false)]);
    let n400 = with_iso(&ids, 400);
    assert_eq!(f[1].photos, n400, "the shutter column looks only at the ISO 400 photos");
    assert_eq!(f[1].values.iter().map(|v| v.count).sum::<usize>(), n400);
    // a value chosen to the right that the left takes away is still listed, with no photos
    exec(&mut h, "library.filter", json!({"columns": [{"field": "iso", "values": ["400"]}, {"field": "shutterSpeed", "values": ["30"]}]}));
    let f = h.app.session.facets(&[("iso".into(), false), ("shutterSpeed".into(), false)]);
    assert!(f[1].values.iter().any(|v| v.value == "30" && v.count == 0), "{:?}", f[1].values);
    assert!(h.app.session.visible_cloned().is_empty());
    h.settle(SETTLE);
}

#[test]
fn the_date_tree_opens_years_and_months_and_a_year_takes_in_its_days() {
    let (mut h, ids) = grid();
    assert!(shown(&h, "filtercol:date:2016") && shown(&h, "filtercol:date:2017"), "years");
    assert!(!shown(&h, "filtercol:date:2016-08"), "months only once the year is open");
    click(&mut h, "filtercol:date:open:2016", json!({}));
    assert!(shown(&h, "filtercol:date:2016-08"));
    assert!(h.app.session.filter.columns.is_empty(), "opening isn't choosing");
    click(&mut h, "filtercol:date:open:2016-08", json!({}));
    assert!(shown(&h, "filtercol:date:2016-08-01"));
    click(&mut h, "filtercol:date:2016", json!({}));
    assert_eq!(h.app.session.visible_cloned().len(), ids.len() / 2, "a year shows all its days");
    // Flat: every day in a list
    click(&mut h, "filtercol:date:menu", json!({}));
    click(&mut h, "filtercol:date:flat", json!({}));
    assert!(h.app.ui.filter_columns[0].flat);
    assert!(shown(&h, "filtercol:date:2017-01-05") && !shown(&h, "filtercol:date:open:2017"));
    h.settle(SETTLE);
}

#[test]
fn the_column_menus_add_remove_sort_and_change_fields() {
    let (mut h, _) = grid();
    let fields = |h: &Headless| h.app.ui.filter_columns.iter().map(|c| c.field.clone()).collect::<Vec<_>>();
    assert_eq!(fields(&h), ["date", "iso", "shutterSpeed", "aperture"], "the default columns");
    click(&mut h, "filtercol:iso:menu", json!({}));
    click(&mut h, "filtercol:iso:addRight", json!({}));
    assert_eq!(fields(&h), ["date", "iso", "month", "shutterSpeed", "aperture"], "a new column right of ISO, of a field not shown");
    click(&mut h, "filtercol:month:menu", json!({}));
    click(&mut h, "filtercol:month:remove", json!({}));
    assert_eq!(fields(&h).len(), 4);
    click(&mut h, "filtercol:iso:menu", json!({}));
    click(&mut h, "filtercol:iso:descending", json!({}));
    assert!(h.app.ui.filter_columns[1].descending);
    // the name opens the fields: choosing another one lets go of what was chosen in the column
    click(&mut h, "filtercol:aperture:header", json!({}));
    assert!(shown(&h, "filtercol:field:camera"));
    click(&mut h, "filtercol:aperture:header", json!({}));
    exec(&mut h, "library.filter", json!({"columns": [{"field": "aperture", "values": ["2.8"]}]}));
    exec(&mut h, "view.filterColumns", json!({"columns": [{"field": "date"}, {"field": "iso"}, {"field": "shutterSpeed"}, {"field": "camera"}]}));
    assert!(h.app.session.filter.columns.is_empty(), "the aperture column went, and its choice with it");
    h.settle(SETTLE);
}

#[test]
fn attribute_metadata_and_none_choose_the_sections() {
    let (mut h, _) = grid();
    click(&mut h, "button:filterAttribute", json!({}));
    assert!(!h.app.ui.filter_attribute && h.app.ui.filter_metadata);
    assert!(!shown(&h, "filter:attribute") && shown(&h, "filter:metadata"));
    click(&mut h, "button:filterNone", json!({}));
    assert!(!h.app.ui.filter_attribute && !h.app.ui.filter_metadata);
    assert!(shown(&h, "filter:header") && !shown(&h, "filter:metadata"), "None leaves the header only");
    click(&mut h, "button:filterMetadata", json!({}));
    click(&mut h, "button:filterAttribute", json!({}));
    assert!(h.app.ui.filter_attribute && h.app.ui.filter_metadata);
    h.settle(SETTLE);
}

#[test]
fn a_column_choice_has_a_chip_that_lets_it_go() {
    let (mut h, _) = grid();
    exec(&mut h, "library.filter", json!({"columns": [{"field": "iso", "values": ["100", "400"]}, {"field": "shutterSpeed", "values": ["1/60"]}]}));
    let chips = lightcraft_engine::filter_chips(&h.app.session.filter, &h.app.session.catalog);
    let iso = chips.iter().find(|c| c.label.starts_with("ISO Speed")).expect("an ISO chip");
    assert_eq!(crate::panels::chips::display_label(iso, &h.app.session.filter, &h.app.session.catalog), "ISO Speed: ISO 100 or ISO 400");
    exec(&mut h, "library.filter", iso.clear.clone());
    assert_eq!(
        h.app.session.filter.columns,
        vec![ColumnFilter { field: "shutterSpeed".into(), values: vec!["1/60".into()] }],
        "only the ISO choice went"
    );
    h.settle(SETTLE);
}

#[test]
fn bad_column_requests_change_nothing() {
    let (mut h, _) = grid();
    let before = (h.app.ui.filter_columns.clone(), h.app.session.filter.clone());
    for (cmd, params) in [
        ("view.filterColumns", json!({"columns": []})),
        ("view.filterColumns", json!({"columns": [{"field": "nope"}]})),
        ("view.filterColumns", json!({"columns": [{"field": "iso"}, {"field": "iso"}]})),
        ("view.filterColumns", json!({"columns": [{"field": "iso", "sort": "sideways"}]})),
        ("view.filterSections", json!({"attribute": "yes"})),
        ("library.filter", json!({"columns": [{"field": "nope", "values": ["1"]}]})),
        ("library.filter", json!({"columns": [{"field": "iso", "values": ["1"]}, {"field": "iso", "values": ["2"]}]})),
        ("library.facets", json!({"columns": [{"field": "nope"}]})),
        ("library.facets", json!({})),
    ] {
        let r = h.request("engine.execute", json!({"command": cmd, "params": params}), T);
        assert_eq!(r["ok"], false, "{cmd} {params}: {r}");
        assert_eq!((h.app.ui.filter_columns.clone(), h.app.session.filter.clone()), before, "{cmd} {params}");
    }
    h.settle(SETTLE);
}
