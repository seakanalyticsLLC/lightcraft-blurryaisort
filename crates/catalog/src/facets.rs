//! The library filter's metadata columns (the Library Filter's Metadata section, as in Lightroom):
//! a column lists the values one field takes among the photos in view, with how many photos have
//! each, and choosing values shows the photos that have any of them. Columns narrow from left to
//! right: each lists the values among the photos the columns to its left let through.
//!
//! [`FIELDS`] are the fields a column can show; [`values_of`] what a photo has for one;
//! [`tally`] a column's values and counts; [`ColumnFilter`] a column's choice in a
//! [`crate::Filter`], matched by [`column_matches`].

use std::cmp::Ordering;

use serde::{Deserialize, Serialize};

use crate::{ColorLabel, Flag, MediaKind, Photo};

/// One metadata column's choice: the photos whose `field` has any of `values` (an empty list, the
/// column's "All", doesn't filter). A `date` value may be a year (`2016`) or a month (`2016-08`):
/// it takes in every day under it.
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(default)]
pub struct ColumnFilter {
    pub field: String,
    pub values: Vec<String>,
}

/// How a field's values sort.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Order {
    /// Case-insensitive text.
    Text,
    /// As numbers (shutter speeds by their length in seconds).
    Number,
    /// Dates, as written (`2016-08-27` sorts as text).
    Date,
    /// A fixed rank (flags, labels, yes / no).
    Rank,
}

/// A field a metadata column can show.
#[derive(Debug)]
pub struct Field {
    /// The id commands use (`library.filter`'s `columns`, `library.facets`).
    pub id: &'static str,
    /// The column's name.
    pub label: &'static str,
    /// Its name for several values: "All (3 ISO Speeds)".
    pub plural: &'static str,
    /// The group it is listed in on the column's field menu (a separator between groups; the last
    /// group is the Develop submenu).
    pub group: u8,
    order: Order,
}

/// The group of [`FIELDS`] listed in the field menu's Develop submenu.
pub const DEVELOP_GROUP: u8 = 6;

/// Every field a column can show, in the order of the column's field menu.
pub const FIELDS: &[Field] = &[
    Field { id: "date", label: "Date", plural: "Dates", group: 0, order: Order::Date },
    Field { id: "month", label: "Month", plural: "Months", group: 0, order: Order::Date },
    Field { id: "day", label: "Day", plural: "Days", group: 0, order: Order::Date },
    Field { id: "fileType", label: "File Type", plural: "File Types", group: 1, order: Order::Text },
    Field { id: "flag", label: "Flag", plural: "Flags", group: 2, order: Order::Rank },
    Field { id: "rating", label: "Rating", plural: "Ratings", group: 2, order: Order::Number },
    Field { id: "label", label: "Label", plural: "Labels", group: 2, order: Order::Rank },
    Field { id: "keyword", label: "Keyword", plural: "Keywords", group: 2, order: Order::Text },
    Field { id: "edit", label: "Edit", plural: "Edit States", group: 2, order: Order::Rank },
    Field { id: "camera", label: "Camera", plural: "Cameras", group: 3, order: Order::Text },
    Field { id: "lens", label: "Lens", plural: "Lenses", group: 3, order: Order::Text },
    Field { id: "focalLength", label: "Focal Length", plural: "Focal Lengths", group: 3, order: Order::Number },
    Field { id: "shutterSpeed", label: "Shutter Speed", plural: "Shutter Speeds", group: 3, order: Order::Number },
    Field { id: "aperture", label: "Aperture", plural: "Apertures", group: 3, order: Order::Number },
    Field { id: "iso", label: "ISO Speed", plural: "ISO Speeds", group: 3, order: Order::Number },
    Field { id: "gps", label: "GPS Data", plural: "GPS States", group: 3, order: Order::Rank },
    Field { id: "location", label: "Location", plural: "Locations", group: 4, order: Order::Text },
    Field { id: "city", label: "City", plural: "Cities", group: 4, order: Order::Text },
    Field { id: "state", label: "State / Province", plural: "States / Provinces", group: 4, order: Order::Text },
    Field { id: "country", label: "Country / Region", plural: "Countries / Regions", group: 4, order: Order::Text },
    Field { id: "creator", label: "Creator", plural: "Creators", group: 5, order: Order::Text },
    Field { id: "copyrightStatus", label: "Copyright Status", plural: "Copyright Statuses", group: 5, order: Order::Rank },
    Field { id: "aspect", label: "Aspect Ratio", plural: "Aspect Ratios", group: DEVELOP_GROUP, order: Order::Rank },
    Field { id: "treatment", label: "Treatment", plural: "Treatments", group: DEVELOP_GROUP, order: Order::Rank },
];

/// The field `id` names, if a column can show it.
pub fn field(id: &str) -> Option<&'static Field> {
    FIELDS.iter().find(|f| f.id == id)
}

/// Whether a column of `field` can show its values as a tree (a date's years, months and days).
pub fn hierarchical(field: &str) -> bool {
    field == "date"
}

/// A number for a column value: one decimal at most (`5.6`, `400`), the way the Info panel shows it.
fn number(v: f32) -> String {
    if !v.is_finite() {
        return String::new();
    }
    let r = (f64::from(v) * 10.0).round() / 10.0;
    if r.fract() == 0.0 { format!("{r:.0}") } else { format!("{r:.1}") }
}

/// A file's type as a column lists it: `Raw`, `DNG`, `Video`, else its format (`JPEG`, `TIFF`, `HEIF`…).
fn file_type(p: &Photo) -> String {
    let format = p.format.trim().to_ascii_uppercase();
    match p.kind {
        MediaKind::Video => "Video".into(),
        _ if format == "DNG" => "DNG".into(),
        MediaKind::Raw => "Raw".into(),
        MediaKind::Image => match format.as_str() {
            "JPG" | "JPE" => "JPEG".into(),
            "TIF" => "TIFF".into(),
            "HEIC" | "HIF" => "HEIF".into(),
            _ => format,
        },
    }
}

/// What photo `p` has for `field`: one value for most fields, any number for keywords, `""` where
/// nothing is recorded (listed as Unknown). Nothing for a field no column shows.
pub fn values_of(field: &str, p: &Photo) -> Vec<String> {
    let m = &p.meta;
    let captured = p.captured.as_deref().unwrap_or("");
    let text = |s: &str| vec![s.trim().to_string()];
    match field {
        "date" => vec![captured.get(..10).unwrap_or("").to_string()],
        "month" => vec![captured.get(..7).unwrap_or("").to_string()],
        "day" => vec![captured.get(5..10).unwrap_or("").to_string()],
        "fileType" => vec![file_type(p)],
        "flag" => vec![format!("{:?}", p.flag).to_lowercase()],
        "rating" => vec![p.rating.min(5).to_string()],
        "label" => vec![p.label.map(|l| format!("{l:?}").to_lowercase()).unwrap_or_default()],
        "keyword" => {
            let mut v: Vec<String> = m.keywords.iter().map(|k| k.trim().to_string()).filter(|k| !k.is_empty()).collect();
            v.sort();
            v.dedup();
            if v.is_empty() { vec![String::new()] } else { v }
        }
        "edit" => vec![if p.is_edited() { "edited" } else { "unedited" }.into()],
        "camera" => text(&m.camera),
        "lens" => text(&m.lens),
        "focalLength" => vec![m.focal_mm.map(number).unwrap_or_default()],
        "shutterSpeed" => text(&m.shutter),
        "aperture" => vec![m.aperture.map(number).unwrap_or_default()],
        "iso" => vec![m.iso.map(|i| i.to_string()).unwrap_or_default()],
        "gps" => vec![if m.gps.is_some() { "yes" } else { "no" }.into()],
        "location" => text(&m.location),
        "city" => text(&m.city),
        "state" => text(&m.state),
        "country" => text(&m.country),
        "creator" => text(&m.creator),
        "copyrightStatus" => vec![m.copyright_status.id().into()],
        "aspect" => vec![crate::rules::aspect(p.shown_size()).unwrap_or("").into()],
        "treatment" => vec![if p.develop.treatment == lightcraft_develop::Treatment::Bw { "monochrome" } else { "color" }.into()],
        _ => Vec::new(),
    }
}

/// Whether one of photo `p`'s values for the column's field is among those chosen. A column with
/// nothing chosen, or of a field no column shows (a filter saved by a newer LightCraft), doesn't
/// filter.
pub fn column_matches(c: &ColumnFilter, p: &Photo) -> bool {
    if c.values.is_empty() || field(&c.field).is_none() {
        return true;
    }
    let have = values_of(&c.field, p);
    c.values.iter().any(|want| {
        have.iter().any(|v| {
            v == want
                // a year or a month of the date tree takes in its days
                || (c.field == "date" && !want.is_empty() && v.starts_with(want.as_str()) && v.get(want.len()..).is_some_and(|rest| rest.starts_with('-')))
        })
    })
}

/// One value of a column, and how many photos have it.
#[derive(Clone, Debug, PartialEq, Eq, Serialize)]
pub struct FacetValue {
    pub value: String,
    pub count: usize,
}

/// The rank of a fixed-order value (flags, labels, yes / no…).
fn rank(field: &str, v: &str) -> usize {
    let order: &[&str] = match field {
        "flag" => &["pick", "none", "reject"],
        "label" => &["red", "yellow", "green", "blue", "purple"],
        "edit" => &["edited", "unedited"],
        "gps" => &["yes", "no"],
        "copyrightStatus" => &["copyrighted", "publicDomain", "unknown"],
        "aspect" => &["landscape", "portrait", "square"],
        "treatment" => &["color", "monochrome"],
        _ => &[],
    };
    order.iter().position(|x| *x == v).unwrap_or(order.len())
}

/// A value as a number, to sort by: a shutter speed by its seconds.
fn as_number(field: &str, v: &str) -> f64 {
    let n = if field == "shutterSpeed" { crate::parse_shutter_seconds(v) } else { v.trim().parse::<f64>().ok() };
    n.filter(|n| n.is_finite()).unwrap_or(f64::MAX)
}

/// How two values of `field` sort, ascending; Unknown (`""`) last.
fn compare(field: &str, a: &str, b: &str) -> Ordering {
    match (a.is_empty(), b.is_empty()) {
        (true, true) => return Ordering::Equal,
        (true, false) => return Ordering::Greater,
        (false, true) => return Ordering::Less,
        _ => {}
    }
    let order = self::field(field).map_or(Order::Text, |f| f.order);
    let by = match order {
        Order::Text => crate::query::cmp_name(a, b),
        Order::Date => a.cmp(b),
        Order::Number => as_number(field, a).total_cmp(&as_number(field, b)),
        Order::Rank => rank(field, a).cmp(&rank(field, b)),
    };
    by.then_with(|| a.cmp(b))
}

/// The values `field` takes among `photos` and how many photos have each, ascending (or
/// descending), Unknown last either way. A photo is counted once per value (a photo with two
/// keywords counts under both).
pub fn tally<'a>(field: &str, photos: impl IntoIterator<Item = &'a Photo>, descending: bool) -> Vec<FacetValue> {
    let mut counts: std::collections::HashMap<String, usize> = std::collections::HashMap::new();
    for p in photos {
        for v in values_of(field, p) {
            *counts.entry(v).or_default() += 1;
        }
    }
    let mut out: Vec<FacetValue> = counts.into_iter().map(|(value, count)| FacetValue { value, count }).collect();
    out.sort_by(|a, b| {
        let o = compare(field, &a.value, &b.value);
        // Unknown stays last when the rest is reversed
        if descending && !a.value.is_empty() && !b.value.is_empty() { o.reverse() } else { o }
    });
    out
}

/// A colour label's id in a `label` column (`red`…); `""` is no label.
pub fn label_value(l: Option<ColorLabel>) -> String {
    l.map(|l| format!("{l:?}").to_lowercase()).unwrap_or_default()
}

/// A flag's id in a `flag` column.
pub fn flag_value(f: Flag) -> String {
    format!("{f:?}").to_lowercase()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Meta, PhotoId, Source};

    fn photo(id: u64, f: impl FnOnce(&mut Photo)) -> Photo {
        let mut p = Photo::new(PhotoId(id), Source::Demo { scene: 0 }, &format!("{id}.jpg"), "JPEG", 6000, 4000, "2026-09-20T10:00:00");
        p.meta = Meta::default();
        f(&mut p);
        p
    }

    #[test]
    fn every_field_is_listed_once_and_has_values() {
        let p = photo(1, |_| {});
        for (i, f) in FIELDS.iter().enumerate() {
            assert!(FIELDS.iter().skip(i + 1).all(|g| g.id != f.id), "{} twice", f.id);
            assert!(!values_of(f.id, &p).is_empty(), "{} gives a value (Unknown at least)", f.id);
        }
        assert!(values_of("nope", &p).is_empty());
        assert!(field("iso").is_some() && field("nope").is_none());
    }

    #[test]
    fn values_read_the_photo_metadata() {
        let p = photo(1, |p| {
            p.captured = Some("2016-08-27T14:03:00".into());
            p.meta.iso = Some(400);
            p.meta.aperture = Some(5.6);
            p.meta.focal_mm = Some(400.0);
            p.meta.shutter = " 1/250 ".into();
            p.meta.keywords = vec!["air show".into(), "b".into(), "air show".into()];
            p.format = "jpg".into();
        });
        assert_eq!(values_of("date", &p), ["2016-08-27"]);
        assert_eq!(values_of("month", &p), ["2016-08"]);
        assert_eq!(values_of("day", &p), ["08-27"]);
        assert_eq!(values_of("iso", &p), ["400"]);
        assert_eq!(values_of("aperture", &p), ["5.6"]);
        assert_eq!(values_of("focalLength", &p), ["400"]);
        assert_eq!(values_of("shutterSpeed", &p), ["1/250"]);
        assert_eq!(values_of("keyword", &p), ["air show", "b"], "each keyword once");
        assert_eq!(values_of("fileType", &p), ["JPEG"]);
        assert_eq!(values_of("camera", &p), [""], "nothing recorded is Unknown");
        // a short or missing capture time is an unknown date, never a panic
        let q = photo(2, |p| p.captured = Some("2016".into()));
        assert_eq!(
            (values_of("date", &q), values_of("month", &q), values_of("day", &q)),
            (vec![String::new()], vec![String::new()], vec![String::new()])
        );
        // NaN or infinite numbers are unknown
        let r = photo(3, |p| p.meta.aperture = Some(f32::NAN));
        assert_eq!(values_of("aperture", &r), [""]);
    }

    #[test]
    fn tally_counts_and_sorts_numbers_as_numbers_with_unknown_last() {
        let ps: Vec<Photo> = [Some(100), Some(1600), Some(200), None, Some(100)]
            .into_iter()
            .enumerate()
            .map(|(i, iso)| photo(i as u64 + 1, |p| p.meta.iso = iso))
            .collect();
        let t = tally("iso", &ps, false);
        let got: Vec<(&str, usize)> = t.iter().map(|v| (v.value.as_str(), v.count)).collect();
        assert_eq!(got, [("100", 2), ("200", 1), ("1600", 1), ("", 1)]);
        let t = tally("iso", &ps, true);
        let got: Vec<&str> = t.iter().map(|v| v.value.as_str()).collect();
        assert_eq!(got, ["1600", "200", "100", ""], "descending keeps Unknown last");
        // shutter speeds by their length
        let ps: Vec<Photo> =
            ["1/1000", "1/60", "2", "1/250"].into_iter().enumerate().map(|(i, s)| photo(i as u64 + 1, |p| p.meta.shutter = s.into())).collect();
        let t: Vec<String> = tally("shutterSpeed", &ps, false).into_iter().map(|v| v.value).collect();
        assert_eq!(t, ["1/1000", "1/250", "1/60", "2"]);
    }

    #[test]
    fn a_column_matches_any_chosen_value_and_dates_take_in_their_days() {
        let p = photo(1, |p| {
            p.captured = Some("2016-08-27T14:03:00".into());
            p.meta.iso = Some(400);
        });
        let col = |field: &str, values: &[&str]| ColumnFilter { field: field.into(), values: values.iter().map(|s| s.to_string()).collect() };
        assert!(column_matches(&col("iso", &[]), &p), "All doesn't filter");
        assert!(column_matches(&col("iso", &["100", "400"]), &p));
        assert!(!column_matches(&col("iso", &["100"]), &p));
        assert!(column_matches(&col("date", &["2016"]), &p), "a year takes in its days");
        assert!(column_matches(&col("date", &["2016-08"]), &p), "a month too");
        assert!(!column_matches(&col("date", &["2016-0"]), &p), "but only whole parts of the date");
        assert!(!column_matches(&col("date", &["2017"]), &p));
        assert!(column_matches(&col("camera", &[""]), &p), "Unknown is a value");
        assert!(column_matches(&col("nope", &["x"]), &p), "a field no column shows doesn't filter");
    }
}
