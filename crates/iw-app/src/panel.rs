//! The New Planet panel: every DESIGN.md §9.1 tunable, plus the phase re-run
//! buttons.
//!
//! The panel keeps its own editable copy of a [`PlanetConfig`]; nothing it
//! shows affects the running simulation until Apply (or the dice button, which
//! applies immediately). The state <-> config conversion is a pure function
//! pair so it can be round-trip tested without a window.

use iw_core::{Phase, PlanetConfig};
use iw_render_vulkan::egui;

/// Rough total resident bytes per cell across planet state, mesh, snapshots
/// and GPU geometry. Calibrated so the level-9 readout lands inside the
/// 2–4 GB estimate in DESIGN.md §10.
pub const EST_BYTES_PER_CELL: u64 = 900;

/// Human-readable phase names, in schedule order.
pub fn phase_name(phase: Phase) -> &'static str {
    match phase {
        Phase::CrustalFormation => "Crustal formation",
        Phase::Drift => "Continental drift",
        Phase::Refinement => "Refinement",
        Phase::RecentPast => "Recent past",
    }
}

/// What the user asked for this frame.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct PanelAction {
    /// Rebuild the planet from the panel's config.
    pub regenerate: bool,
    /// Re-run this phase (and everything after it) from its checkpoint.
    pub rerun_from: Option<Phase>,
}

/// Editable mirror of a [`PlanetConfig`].
///
/// The seed is held as text so a user can type a full `u64`; everything else
/// is edited in place on the config, which keeps [`PanelState::to_config`]
/// trivial and lossless.
#[derive(Debug, Clone)]
pub struct PanelState {
    /// Seed as typed. Invalid text falls back to the last good seed.
    pub seed_text: String,
    /// Working copy of every other tunable.
    pub config: PlanetConfig,
    /// History budget in MiB (the config stores bytes).
    pub history_cap_mib: u64,
    /// Whether the panel window is open.
    pub open: bool,
}

impl PanelState {
    /// Mirror `config` into a fresh panel state.
    pub fn from_config(config: &PlanetConfig) -> PanelState {
        PanelState {
            seed_text: config.seed.to_string(),
            config: config.clone(),
            history_cap_mib: (config.history_cap_bytes / (1024 * 1024)).max(1),
            open: false,
        }
    }

    /// The config the panel currently describes. Sanitised, so it is always
    /// safe to hand to the simulation.
    pub fn to_config(&self) -> PlanetConfig {
        let mut config = self.config.clone();
        config.seed = self.seed_text.trim().parse().unwrap_or(self.config.seed);
        config.history_cap_bytes = self.history_cap_mib.max(1) * 1024 * 1024;
        config.sanitize();
        config
    }

    /// Adopt `config` (e.g. after the simulation sanitised it).
    pub fn set_config(&mut self, config: &PlanetConfig) {
        *self = PanelState {
            open: self.open,
            ..PanelState::from_config(config)
        };
    }

    /// Replace the seed with a fresh pseudo-random one derived from `entropy`
    /// (wall-clock nanoseconds at the call site) mixed with the current seed,
    /// so repeated clicks never repeat.
    pub fn randomize_seed(&mut self, entropy: u64) {
        let mut x = entropy ^ self.to_config().seed.rotate_left(17) ^ 0x9e37_79b9_7f4a_7c15;
        // splitmix64 finaliser: cheap, no dependency, well distributed.
        x = (x ^ (x >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        x = (x ^ (x >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        x ^= x >> 31;
        self.seed_text = x.to_string();
    }

    /// Estimated resident memory for the current cell budget.
    pub fn est_bytes(&self) -> u64 {
        self.config.cell_budget as u64 * EST_BYTES_PER_CELL
    }
}

/// Format a byte count as MiB/GiB.
pub fn format_bytes(bytes: u64) -> String {
    const MIB: f64 = (1024 * 1024) as f64;
    let mib = bytes as f64 / MIB;
    if mib >= 1024.0 {
        format!("{:.1} GiB", mib / 1024.0)
    } else {
        format!("{mib:.0} MiB")
    }
}

/// Draw the panel. `running` is the config of the planet currently being
/// simulated, `completed` the phases whose checkpoints exist on disk (a phase
/// can be re-run when the *previous* phase has a checkpoint, and the first
/// phase is always re-runnable because it is just a regenerate).
pub fn show(
    ctx: &egui::Context,
    state: &mut PanelState,
    running: &PlanetConfig,
    completed: &[Phase],
) -> PanelAction {
    let mut action = PanelAction::default();
    let mut open = state.open;
    egui::Window::new("New planet")
        .open(&mut open)
        .default_pos([360.0, 40.0])
        // Tall enough to reach the Apply button without scrolling.
        .default_size([340.0, 900.0])
        .vscroll(true)
        .show(ctx, |ui| {
            ui.horizontal(|ui| {
                ui.label("Seed");
                ui.add(
                    egui::TextEdit::singleline(&mut state.seed_text)
                        .desired_width(160.0)
                        .hint_text("u64"),
                );
                if ui
                    .button("\u{1f3b2} Random")
                    .on_hover_text("Randomise the seed and regenerate immediately")
                    .clicked()
                {
                    state.randomize_seed(entropy());
                    action.regenerate = true;
                }
            });
            if state.seed_text.trim().parse::<u64>().is_err() {
                ui.colored_label(
                    egui::Color32::from_rgb(0xd0, 0x80, 0x40),
                    format!("not a number - keeping {}", state.config.seed),
                );
            }

            ui.separator();
            let mut budget = state.config.cell_budget;
            ui.add(
                egui::Slider::new(&mut budget, 10_000..=3_000_000)
                    .logarithmic(true)
                    .text("Cell budget"),
            );
            state.config.cell_budget = budget;
            ui.label(format!(
                "{budget} terrain-sized Voronoi cells  ~{}",
                format_bytes(state.est_bytes())
            ));
            if budget != running.cell_budget {
                ui.label("changing the budget re-tessellates the planet");
            }

            ui.separator();
            ui.label("Phase durations (Myr)");
            let ranges = [0.0..=1000.0, 0.0..=1000.0, 0.0..=500.0, 0.0..=20.0];
            for (i, phase) in Phase::ALL.iter().enumerate() {
                ui.add(
                    egui::Slider::new(&mut state.config.phase_durations_myr[i], ranges[i].clone())
                        .text(phase_name(*phase)),
                );
            }
            ui.label(format!(
                "total {:.0} Myr",
                state.config.total_duration_myr()
            ));

            ui.separator();
            ui.add(
                egui::Slider::new(&mut state.config.water_budget, 0.1..=3.0)
                    .text("Water budget")
                    .suffix(" oceans"),
            );
            ui.add(
                egui::Slider::new(&mut state.config.temperature_offset_c, -20.0..=20.0)
                    .text("Temperature offset")
                    .suffix(" C"),
            );
            ui.add(
                egui::Slider::new(&mut state.config.axial_tilt_deg, 0.0..=45.0)
                    .text("Axial tilt")
                    .suffix(" deg"),
            );
            ui.add(
                egui::Slider::new(&mut state.config.precip_multiplier, 0.25..=4.0)
                    .logarithmic(true)
                    .text("Precipitation")
                    .suffix("x"),
            );
            ui.add(
                egui::Slider::new(&mut state.config.tectonic_vigor, 0.25..=2.0)
                    .text("Tectonic vigour")
                    .suffix("x"),
            );
            ui.add(egui::Slider::new(&mut state.config.hotspot_count, 0..=30).text("Hotspots"));
            ui.add(egui::Slider::new(&mut state.config.craton_count, 4..=30).text("Cratons"));
            ui.add(
                egui::Slider::new(&mut state.config.glacial_intensity, 0.0..=2.0)
                    .text("Glacial intensity")
                    .suffix("x"),
            );
            ui.add(
                egui::Slider::new(&mut state.history_cap_mib, 64..=8192)
                    .logarithmic(true)
                    .text("History cap")
                    .suffix(" MiB"),
            );

            ui.separator();
            if ui
                .add(egui::Button::new("Apply & regenerate"))
                .on_hover_text("Throw the planet away and rebuild it from these settings")
                .clicked()
            {
                action.regenerate = true;
            }

            ui.separator();
            ui.label("Re-run from a phase (keeps earlier history):");
            for phase in Phase::ALL {
                let enabled = rerun_available(phase, completed);
                let button = ui.add_enabled(enabled, egui::Button::new(phase_name(phase)));
                if button.clicked() {
                    action.rerun_from = Some(phase);
                }
                if !enabled {
                    button.on_hover_text("no checkpoint for the preceding phase yet");
                }
            }

            ui.separator();
            ui.collapsing("Running planet", |ui| {
                ui.label(format!("seed {}", running.seed));
                ui.label(format!(
                    "level {} ({} cells)",
                    running.subdivision_level,
                    running.n_cells()
                ));
                ui.label(format!(
                    "phases {:?} Myr, total {:.0}",
                    running.phase_durations_myr,
                    running.total_duration_myr()
                ));
                ui.label(format!(
                    "water {:.2}  temp {:+.1} C  tilt {:.1} deg",
                    running.water_budget, running.temperature_offset_c, running.axial_tilt_deg
                ));
                ui.label(format!(
                    "precip {:.2}x  vigour {:.2}x  glacial {:.2}x",
                    running.precip_multiplier, running.tectonic_vigor, running.glacial_intensity
                ));
                ui.label(format!(
                    "hotspots {}  cratons {}",
                    running.hotspot_count, running.craton_count
                ));
            });
        });
    state.open = open;
    action
}

/// Whether re-running `phase` is possible: the first phase always is (it is a
/// plain regenerate), the others need the preceding phase's checkpoint.
pub fn rerun_available(phase: Phase, completed: &[Phase]) -> bool {
    match Phase::ALL.get(phase.index().wrapping_sub(1)) {
        None => true,
        Some(prev) => completed.contains(prev),
    }
}

/// Wall-clock nanoseconds, used only as dice entropy.
pub fn entropy() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_nanos() as u64)
        .unwrap_or(0x5eed)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample() -> PlanetConfig {
        let mut c = PlanetConfig {
            seed: 18_446_744_073_709_551_557,
            subdivision_level: 7,
            phase_durations_myr: [150.0, 250.0, 60.0, 3.0],
            phase_dt_myr: [1.0, 0.5, 0.25, 0.005],
            water_budget: 1.4,
            temperature_offset_c: -3.5,
            axial_tilt_deg: 31.0,
            precip_multiplier: 2.0,
            tectonic_vigor: 0.75,
            hotspot_count: 12,
            craton_count: 22,
            glacial_intensity: 1.5,
            history_cap_bytes: 512 * 1024 * 1024,
            cell_budget: 60_000,
        };
        c.sanitize();
        c
    }

    #[test]
    fn config_round_trips_through_the_panel() {
        let config = sample();
        let state = PanelState::from_config(&config);
        assert_eq!(state.to_config(), config);

        // ...including the default config, and a maximum-value seed.
        let default = PlanetConfig::default();
        assert_eq!(PanelState::from_config(&default).to_config(), default);
        let mut edge = sample();
        edge.seed = u64::MAX;
        edge.history_cap_bytes = 64 * 1024 * 1024;
        assert_eq!(PanelState::from_config(&edge).to_config(), edge);
    }

    #[test]
    fn edits_reach_the_config() {
        let mut state = PanelState::from_config(&PlanetConfig::default());
        state.seed_text = "1337".to_string();
        state.config.craton_count = 25;
        state.config.water_budget = 0.4;
        state.history_cap_mib = 128;
        let out = state.to_config();
        assert_eq!(out.seed, 1337);
        assert_eq!(out.craton_count, 25);
        assert!((out.water_budget - 0.4).abs() < 1e-9);
        assert_eq!(out.history_cap_bytes, 128 * 1024 * 1024);
    }

    #[test]
    fn a_bad_seed_keeps_the_old_one_and_out_of_range_values_are_clamped() {
        let mut state = PanelState::from_config(&PlanetConfig::default());
        state.seed_text = "not a number".to_string();
        assert_eq!(state.to_config().seed, PlanetConfig::default().seed);
        state.seed_text = "  99  ".to_string();
        assert_eq!(state.to_config().seed, 99);

        state.config.craton_count = 9_999;
        state.config.water_budget = 40.0;
        state.config.subdivision_level = 200;
        let out = state.to_config();
        assert_eq!(out.craton_count, 30);
        assert_eq!(out.water_budget, 3.0);
        assert_eq!(out.subdivision_level, 10);
    }

    #[test]
    fn the_dice_changes_the_seed_every_time() {
        let mut state = PanelState::from_config(&PlanetConfig::default());
        let mut seen = std::collections::HashSet::new();
        seen.insert(state.to_config().seed);
        for i in 0..64 {
            state.randomize_seed(i);
            let seed = state.to_config().seed;
            assert!(seen.insert(seed), "dice repeated {seed} on roll {i}");
            // The text must always parse back to the seed it produced.
            assert_eq!(state.seed_text.parse::<u64>().unwrap(), seed);
        }
    }

    #[test]
    fn set_config_keeps_the_window_open_state() {
        let mut state = PanelState::from_config(&PlanetConfig::default());
        state.open = true;
        let other = sample();
        state.set_config(&other);
        assert!(state.open);
        assert_eq!(state.to_config(), other);
    }

    #[test]
    fn memory_estimate_tracks_the_budget() {
        let mut state = PanelState::from_config(&PlanetConfig::default());
        state.config.cell_budget = 40_000;
        let small = state.est_bytes();
        state.config.cell_budget = 640_000;
        let big = state.est_bytes();
        assert!(big == small * 16, "estimate must scale with the budget");
        assert_eq!(format_bytes(1024 * 1024), "1 MiB");
        assert_eq!(format_bytes(3 * 1024 * 1024 * 1024), "3.0 GiB");
    }

    #[test]
    fn rerun_needs_the_previous_checkpoint() {
        assert!(rerun_available(Phase::CrustalFormation, &[]));
        assert!(!rerun_available(Phase::Drift, &[]));
        assert!(rerun_available(Phase::Drift, &[Phase::CrustalFormation]));
        assert!(!rerun_available(
            Phase::Refinement,
            &[Phase::CrustalFormation]
        ));
        assert!(rerun_available(
            Phase::RecentPast,
            &[Phase::CrustalFormation, Phase::Drift, Phase::Refinement]
        ));
    }
}
