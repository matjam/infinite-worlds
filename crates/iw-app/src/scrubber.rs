//! The time scrubber: a slider over the history snapshots on disk.
//!
//! While the slider is being dragged the globe shows a loaded
//! [`iw_store_postcard::HistorySnapshot`] instead of the live simulation
//! snapshot. Thin snapshots carry only elevation, biome, plate id and ice, so
//! layers that need anything else fall back to elevation colouring (see
//! [`crate::layers::snapshot_layer`]).

use std::time::{Duration, Instant};

use iw_render_vulkan::egui;

/// How often the entry list is re-read from disk while the sim is running.
pub const REFRESH_INTERVAL: Duration = Duration::from_millis(750);

/// What the scrubber wants the app to do after this frame.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ScrubAction {
    /// Nothing changed.
    None,
    /// Show this history version.
    Load(u64),
    /// Stop scrubbing and follow the simulation again.
    Live,
}

/// Scrubber state: the entries we know about and where the handle is.
pub struct Scrubber {
    /// `(version, time_myr)` pairs, oldest first.
    pub entries: Vec<(u64, f64)>,
    /// Index into `entries` of the selected snapshot.
    pub index: usize,
    /// True while following the live simulation (the normal state).
    pub live: bool,
    /// Version currently displayed from history, if any.
    pub showing: Option<u64>,
    /// True when the last load substituted the elevation layer.
    pub layer_substituted: bool,
    /// When the entry list was last re-read.
    pub last_refresh: Instant,
}

impl Default for Scrubber {
    fn default() -> Self {
        Scrubber {
            entries: Vec::new(),
            index: 0,
            live: true,
            showing: None,
            layer_substituted: false,
            last_refresh: Instant::now() - REFRESH_INTERVAL,
        }
    }
}

impl Scrubber {
    /// Adopt a freshly listed set of entries, keeping the handle on the same
    /// snapshot version where possible (history eviction shifts indices).
    pub fn set_entries(&mut self, entries: Vec<(u64, f64)>) {
        let want = self.selected_version();
        self.entries = entries;
        self.index = match want {
            Some(v) => nearest_index(&self.entries, v),
            None => self.entries.len().saturating_sub(1),
        };
        if self.live {
            self.index = self.entries.len().saturating_sub(1);
        }
        self.last_refresh = Instant::now();
    }

    /// Whether the entry list is due a refresh.
    pub fn due_refresh(&self) -> bool {
        self.last_refresh.elapsed() >= REFRESH_INTERVAL
    }

    /// Version under the handle, if there is one.
    pub fn selected_version(&self) -> Option<u64> {
        self.entries.get(self.index).map(|(v, _)| *v)
    }

    /// Simulated time under the handle, if there is one.
    pub fn selected_time_myr(&self) -> Option<f64> {
        self.entries.get(self.index).map(|(_, t)| *t)
    }

    /// Return to following the simulation.
    pub fn go_live(&mut self) {
        self.live = true;
        self.showing = None;
        self.layer_substituted = false;
        self.index = self.entries.len().saturating_sub(1);
    }
}

/// Index of the entry with `version`, or the nearest one below it; 0 when the
/// list is empty or every entry is newer.
pub fn nearest_index(entries: &[(u64, f64)], version: u64) -> usize {
    match entries.binary_search_by_key(&version, |(v, _)| *v) {
        Ok(i) => i,
        Err(i) => i.saturating_sub(1),
    }
}

/// Draw the bottom scrubber bar. Takes the frame's root [`egui::Ui`] because
/// panels — unlike windows — are laid out inside a parent `Ui`.
pub fn show(
    root: &mut egui::Ui,
    scrubber: &mut Scrubber,
    total_duration_myr: f64,
    layer_name: &str,
) -> ScrubAction {
    let mut action = ScrubAction::None;
    egui::Panel::bottom("scrubber").show(root, |ui| {
        ui.horizontal(|ui| {
            if scrubber.entries.is_empty() {
                ui.weak("history: waiting for the first snapshot");
                return;
            }
            let last = scrubber.entries.len() - 1;
            let before = scrubber.index;
            let response = ui.add(
                egui::Slider::new(&mut scrubber.index, 0..=last)
                    .show_value(false)
                    .text(""),
            );
            if response.changed() || (response.dragged() && scrubber.index != before) {
                scrubber.live = false;
                if let Some(v) = scrubber.selected_version() {
                    action = ScrubAction::Load(v);
                }
            }
            let t = scrubber.selected_time_myr().unwrap_or(0.0);
            ui.label(format!(
                "{t:.1} Myr  ({:.1} Myr ago)",
                (total_duration_myr - t).max(0.0)
            ));
            ui.weak(format!("{}/{} snapshots", scrubber.index + 1, last + 1));
            if scrubber.live {
                ui.colored_label(egui::Color32::from_rgb(0x70, 0xd0, 0x70), "LIVE");
            } else {
                if ui
                    .button("Live")
                    .on_hover_text("Follow the simulation again")
                    .clicked()
                {
                    scrubber.go_live();
                    action = ScrubAction::Live;
                }
                ui.colored_label(
                    egui::Color32::from_rgb(0xd0, 0xa0, 0x40),
                    "HISTORY (live updates paused)",
                );
                if scrubber.layer_substituted {
                    ui.weak(format!(
                        "{layer_name} is not stored in history - showing elevation"
                    ));
                }
            }
        });
    });
    action
}

#[cfg(test)]
mod tests {
    use super::*;

    fn entries(versions: &[u64]) -> Vec<(u64, f64)> {
        versions.iter().map(|v| (*v, *v as f64 * 0.5)).collect()
    }

    #[test]
    fn nearest_index_finds_exact_and_lower_neighbours() {
        let e = entries(&[4, 9, 12, 30]);
        assert_eq!(nearest_index(&e, 4), 0);
        assert_eq!(nearest_index(&e, 12), 2);
        assert_eq!(nearest_index(&e, 13), 2, "falls back to the older snapshot");
        assert_eq!(nearest_index(&e, 1), 0, "older than everything");
        assert_eq!(nearest_index(&e, 999), 3);
        assert_eq!(nearest_index(&[], 5), 0);
    }

    #[test]
    fn live_scrubber_tracks_the_newest_snapshot() {
        let mut s = Scrubber::default();
        s.set_entries(entries(&[1, 2, 3]));
        assert!(s.live);
        assert_eq!(s.selected_version(), Some(3));
        s.set_entries(entries(&[1, 2, 3, 4]));
        assert_eq!(s.selected_version(), Some(4), "live follows new snapshots");
    }

    #[test]
    fn scrubbed_position_survives_history_eviction() {
        let mut s = Scrubber::default();
        s.set_entries(entries(&[1, 2, 3, 4, 5]));
        s.live = false;
        s.index = 2; // version 3
        assert_eq!(s.selected_version(), Some(3));
        // The oldest two snapshots are evicted; the handle must stay on 3.
        s.set_entries(entries(&[3, 4, 5, 6]));
        assert_eq!(s.selected_version(), Some(3));
        assert!(!s.live);
        // Now version 3 itself is evicted: fall back to the oldest kept one.
        s.set_entries(entries(&[4, 5, 6, 7]));
        assert_eq!(s.selected_version(), Some(4));
    }

    #[test]
    fn going_live_jumps_to_the_end() {
        let mut s = Scrubber::default();
        s.set_entries(entries(&[1, 2, 3]));
        s.live = false;
        s.index = 0;
        s.showing = Some(1);
        s.layer_substituted = true;
        s.go_live();
        assert!(s.live);
        assert_eq!(s.showing, None);
        assert!(!s.layer_substituted);
        assert_eq!(s.selected_version(), Some(3));
    }

    #[test]
    fn an_empty_history_is_harmless() {
        let mut s = Scrubber::default();
        s.set_entries(Vec::new());
        assert_eq!(s.selected_version(), None);
        assert_eq!(s.selected_time_myr(), None);
        assert_eq!(s.index, 0);
        s.go_live();
        assert_eq!(s.index, 0);
    }

    #[test]
    fn refresh_is_rate_limited() {
        let mut s = Scrubber::default();
        assert!(s.due_refresh(), "the first refresh is immediate");
        s.set_entries(entries(&[1]));
        assert!(!s.due_refresh());
    }
}
