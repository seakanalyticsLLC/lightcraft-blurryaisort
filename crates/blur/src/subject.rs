//! From a blur map to one number per photo: the subject's blur score. The Python twin is
//! `tools/blurcull/subject.py`; keep both in step (the constants and the percentile rule).
//!
//! Where is the subject?
//! 1. the camera's focus point, when the file records one;
//! 2. otherwise face / pet regions;
//! 3. otherwise "auto": the sharpest part of the frame that has detail. If anything is in focus, that's almost always
//!    what the photographer meant: a sharp subject on a creamy background scores sharp, a missed-focus or shaken shot
//!    has no sharp part and scores blurry.
//!
//! Inside the subject area a low percentile of the blur over cells the network is confident about is taken, so a few
//! soft cells at the subject's edge don't drag it down.

use serde::Serialize;

use crate::log2_to_score;

/// Cells below this confidence (probability) are not judged.
pub const CONF_MIN: f32 = 0.5;
/// Auto subject: the sharpest 15 % of judged cells.
pub const AUTO_PERCENTILE: f64 = 15.0;
/// Focus point / faces: a robust low percentile inside the region.
pub const REGION_PERCENTILE: f64 = 35.0;
/// Fewest judged cells that make a reading.
pub const MIN_CELLS: usize = 4;

/// A normalized rectangle on the upright, uncropped photo (y down, 0..1).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Rect {
    pub x0: f64,
    pub y0: f64,
    pub x1: f64,
    pub y1: f64,
}

/// The network's output for one photo: `blur` = log2(1 + σ) per cell, `conf` = probability there's detail to judge.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct BlurMap {
    pub cols: usize,
    pub rows: usize,
    pub blur: Vec<f32>,
    pub conf: Vec<f32>,
}

/// How the subject was found.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, serde::Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Subject {
    Focus,
    Faces,
    Auto,
}

#[derive(Clone, Copy, Debug, PartialEq, Serialize)]
pub struct SubjectScore {
    /// 0 (tack sharp) .. 100; `None` when nothing in the photo has enough detail to judge (fog, a blank wall).
    pub score: Option<f64>,
    /// Typical blur of the rest of the frame, 0..100 (the median of judged cells).
    pub background: Option<f64>,
    pub subject: Option<Subject>,
    /// Judged cells the reading is based on.
    pub cells: usize,
}

/// numpy's default (linear) percentile of unsorted values; `None` for fewer than [`MIN_CELLS`].
fn percentile(mut v: Vec<f32>, q: f64) -> Option<f64> {
    if v.len() < MIN_CELLS {
        return None;
    }
    v.sort_by(f32::total_cmp);
    let pos = (q / 100.0).clamp(0.0, 1.0) * (v.len() - 1) as f64;
    let lo = pos.floor() as usize;
    let hi = (lo + 1).min(v.len() - 1);
    let (a, b) = (f64::from(*v.get(lo)?), f64::from(*v.get(hi)?));
    Some(a + (b - a) * (pos - lo as f64))
}

impl BlurMap {
    fn judged(&self, i: usize) -> bool {
        self.conf.get(i).is_some_and(|c| *c >= CONF_MIN)
    }

    /// Cell mask for `r` grown by `grow` of its size on each side (focus boxes are small).
    fn region(&self, r: &Rect, grow: f64) -> Vec<bool> {
        let (gw, gh) = ((r.x1 - r.x0).max(0.0) * grow, (r.y1 - r.y0).max(0.0) * grow);
        let clamp = |v: f64| if v.is_finite() { v.clamp(0.0, 1.0) } else { 0.0 };
        let (x0, x1) = (clamp(r.x0 - gw), clamp(r.x1 + gw));
        let (y0, y1) = (clamp(r.y0 - gh), clamp(r.y1 + gh));
        let (wc, hc) = (self.cols as f64, self.rows as f64);
        let cx0 = ((x0 * wc).floor() as usize).min(self.cols);
        let cx1 = ((x1 * wc).ceil() as usize).max(cx0 + 1).min(self.cols);
        let cy0 = ((y0 * hc).floor() as usize).min(self.rows);
        let cy1 = ((y1 * hc).ceil() as usize).max(cy0 + 1).min(self.rows);
        let mut m = vec![false; self.cols * self.rows];
        for y in cy0..cy1 {
            for x in cx0..cx1 {
                if let Some(c) = m.get_mut(y * self.cols + x) {
                    *c = true;
                }
            }
        }
        m
    }

    fn masked(&self, m: &[bool]) -> Vec<f32> {
        self.blur.iter().enumerate().filter(|(i, _)| m.get(*i).copied().unwrap_or(false) && self.judged(*i)).map(|(_, b)| *b).collect()
    }

    /// The subject's blur score. `focus`: the camera's focus box; `regions`: face / pet boxes.
    pub fn subject_score(&self, focus: Option<&Rect>, regions: &[Rect]) -> SubjectScore {
        let all = vec![true; self.blur.len()];
        let judged = self.masked(&all);
        let n_judged = judged.len();
        let background = percentile(judged, 50.0);

        let mut found: Option<(f64, Subject, usize)> = None;
        if let Some(f) = focus {
            for grow in [0.5, 1.5, 3.0] {
                let v = self.masked(&self.region(f, grow));
                let n = v.len();
                if let Some(b) = percentile(v, REGION_PERCENTILE) {
                    found = Some((b, Subject::Focus, n));
                    break;
                }
            }
        }
        if found.is_none() && !regions.is_empty() {
            let mut m = vec![false; self.blur.len()];
            for r in regions {
                for (a, b) in m.iter_mut().zip(self.region(r, 0.15)) {
                    *a |= b;
                }
            }
            let v = self.masked(&m);
            let n = v.len();
            if let Some(b) = percentile(v, REGION_PERCENTILE) {
                found = Some((b, Subject::Faces, n));
            }
        }
        if found.is_none()
            && let Some(b) = percentile(self.masked(&all), AUTO_PERCENTILE)
        {
            found = Some((b, Subject::Auto, n_judged));
        }
        SubjectScore {
            score: found.map(|f| round1(log2_to_score(f.0))),
            background: background.map(|b| round1(log2_to_score(b))),
            subject: found.map(|f| f.1),
            cells: found.map(|f| f.2).unwrap_or(0),
        }
    }
}

fn round1(v: f64) -> f64 {
    (v * 10.0).round() / 10.0
}

#[cfg(test)]
mod tests {
    use super::*;

    fn map(cols: usize, rows: usize, f: impl Fn(usize, usize) -> (f32, f32)) -> BlurMap {
        let mut m = BlurMap { cols, rows, ..Default::default() };
        for y in 0..rows {
            for x in 0..cols {
                let (b, c) = f(x, y);
                m.blur.push(b);
                m.conf.push(c);
            }
        }
        m
    }

    #[test]
    fn percentile_matches_numpy() {
        // np.percentile([1,2,3,4,5,6,7,8,9,10], 15) == 2.35; 35 -> 4.15; 50 -> 5.5
        let v: Vec<f32> = (1..=10).map(|v| v as f32).collect();
        assert!((percentile(v.clone(), 15.0).unwrap() - 2.35).abs() < 1e-9);
        assert!((percentile(v.clone(), 35.0).unwrap() - 4.15).abs() < 1e-9);
        assert!((percentile(v, 50.0).unwrap() - 5.5).abs() < 1e-9);
        assert!(percentile(vec![1.0, 2.0], 50.0).is_none());
    }

    #[test]
    fn sharp_subject_on_soft_background_scores_sharp() {
        // left quarter sharp (0), rest blurred (3.0 = sigma 7)
        let m = map(20, 12, |x, _| (if x < 5 { 0.0 } else { 3.0 }, 0.9));
        let s = m.subject_score(None, &[]);
        assert_eq!(s.subject, Some(Subject::Auto));
        assert_eq!(s.score, Some(0.0));
        assert!(s.background.unwrap() > 60.0);
    }

    #[test]
    fn focus_point_wins_over_the_sharpest_area() {
        // camera focused on the blurred right side: the photo is a miss even though the left is sharp
        let m = map(20, 12, |x, _| (if x < 5 { 0.0 } else { 3.0 }, 0.9));
        let f = Rect { x0: 0.7, y0: 0.4, x1: 0.75, y1: 0.5 };
        let s = m.subject_score(Some(&f), &[]);
        assert_eq!(s.subject, Some(Subject::Focus));
        assert!(s.score.unwrap() > 60.0);
        // faces come next
        let face = Rect { x0: 0.0, y0: 0.2, x1: 0.2, y1: 0.6 };
        let s = m.subject_score(None, &[face]);
        assert_eq!((s.subject, s.score), (Some(Subject::Faces), Some(0.0)));
    }

    #[test]
    fn a_focus_box_on_a_blank_area_widens_then_falls_back() {
        // detail only in the far corner: the focus box (centre, no detail) grows, then auto takes over
        let m = map(20, 12, |x, y| (1.0, if x > 16 && y > 8 { 0.9 } else { 0.0 }));
        let f = Rect { x0: 0.45, y0: 0.45, x1: 0.55, y1: 0.55 };
        let s = m.subject_score(Some(&f), &[]);
        assert_eq!(s.subject, Some(Subject::Auto));
        assert!(s.score.is_some());
    }

    #[test]
    fn nothing_to_judge_is_none_and_hostile_rects_are_safe() {
        let m = map(8, 6, |_, _| (2.0, 0.1));
        assert_eq!(m.subject_score(None, &[]).score, None);
        let m = map(8, 6, |_, _| (2.0, 0.9));
        for r in [
            Rect { x0: f64::NAN, y0: 0.0, x1: f64::INFINITY, y1: 1.0 },
            Rect { x0: 2.0, y0: 2.0, x1: 3.0, y1: 3.0 },
            Rect { x0: -5.0, y0: -5.0, x1: -4.0, y1: -4.0 },
            Rect { x0: 0.9, y0: 0.9, x1: 0.1, y1: 0.1 },
        ] {
            let _ = m.subject_score(Some(&r), &[r]);
        }
        let empty = BlurMap::default();
        assert_eq!(empty.subject_score(None, &[]).score, None);
    }
}
