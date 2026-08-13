use serde::{Deserialize, Serialize};

/// Simulation eras, in order. See DESIGN.md §5.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum Phase {
    CrustalFormation,
    Drift,
    Refinement,
    RecentPast,
}

impl Phase {
    pub const ALL: [Phase; 4] = [
        Phase::CrustalFormation,
        Phase::Drift,
        Phase::Refinement,
        Phase::RecentPast,
    ];

    #[inline]
    pub fn index(self) -> usize {
        Phase::ALL.iter().position(|p| *p == self).unwrap()
    }

    pub fn next(self) -> Option<Phase> {
        Phase::ALL.get(self.index() + 1).copied()
    }
}

/// Every user-facing tunable (DESIGN.md §9.1) plus derived helpers.
/// Serialized into every checkpoint so a planet remembers how it was made.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PlanetConfig {
    pub seed: u64,
    /// Legacy Goldberg subdivision level. Unit tests and comparison tooling
    /// only — the pipeline tessellates by [`Self::cell_budget`].
    pub subdivision_level: u8,
    /// Total Voronoi cell budget (docs/voronoi-v2.md): the fidelity knob.
    /// Cell sizes are terrain-driven; this caps how many there are.
    pub cell_budget: u32,
    /// [crustal formation, drift, refinement, recent past] durations in Myr.
    pub phase_durations_myr: [f64; 4],
    /// Timestep per phase in Myr.
    pub phase_dt_myr: [f64; 4],
    /// Total surface water, in units of Earth's ocean volume (1.335e9 km^3),
    /// scaled by planet surface area relative to Earth.
    pub water_budget: f64,
    /// Added uniformly to the temperature field, degrees C.
    pub temperature_offset_c: f32,
    pub axial_tilt_deg: f32,
    pub precip_multiplier: f32,
    /// Scales plate speeds and volcanism rates.
    pub tectonic_vigor: f32,
    pub hotspot_count: u32,
    pub craton_count: u32,
    /// Scales the amplitude of Phase-4 glacial temperature cycles.
    pub glacial_intensity: f32,
    /// Disk budget for history snapshots (time scrubber).
    pub history_cap_bytes: u64,
}

impl Default for PlanetConfig {
    fn default() -> Self {
        PlanetConfig {
            seed: 42,
            subdivision_level: 8,
            cell_budget: 250_000,
            phase_durations_myr: [200.0, 200.0, 75.0, 2.0],
            phase_dt_myr: [1.0, 0.5, 0.25, 0.005],
            // 0.93 Earth oceans, not 1.0: these worlds' post-breakup oceans
            // are younger and shallower than Earth's (fresh ridge crust
            // floats high), so a full Earth ocean drowns the margins and
            // lands ~20% land. 0.93 restores an Earth-like ~26-29%.
            water_budget: 0.93,
            temperature_offset_c: 0.0,
            axial_tilt_deg: 23.4,
            precip_multiplier: 1.0,
            tectonic_vigor: 1.0,
            hotspot_count: 8,
            craton_count: 14,
            glacial_intensity: 1.0,
            history_cap_bytes: 2 * 1024 * 1024 * 1024,
        }
    }
}

impl PlanetConfig {
    #[inline]
    pub fn n_cells(&self) -> usize {
        10 * 4usize.pow(self.subdivision_level as u32) + 2
    }

    #[inline]
    pub fn duration_myr(&self, phase: Phase) -> f64 {
        self.phase_durations_myr[phase.index()]
    }

    #[inline]
    pub fn dt_myr(&self, phase: Phase) -> f64 {
        self.phase_dt_myr[phase.index()]
    }

    /// Number of steps in a phase (at least 1 if the duration is nonzero).
    pub fn steps_in(&self, phase: Phase) -> u64 {
        let d = self.duration_myr(phase);
        if d <= 0.0 {
            0
        } else {
            (d / self.dt_myr(phase)).round().max(1.0) as u64
        }
    }

    /// Simulation start time on the "Myr before present" axis, as elapsed time 0.
    pub fn total_duration_myr(&self) -> f64 {
        self.phase_durations_myr.iter().sum()
    }

    /// Clamp all values into their sane UI ranges (DESIGN.md §9.1).
    pub fn sanitize(&mut self) {
        self.subdivision_level = self.subdivision_level.clamp(4, 10);
        self.cell_budget = self.cell_budget.clamp(10_000, 3_000_000);
        for d in &mut self.phase_durations_myr {
            *d = d.clamp(0.0, 2000.0);
        }
        for (i, dt) in self.phase_dt_myr.iter_mut().enumerate() {
            let lo = if i == 3 { 0.001 } else { 0.05 };
            *dt = dt.clamp(lo, 5.0);
        }
        self.water_budget = self.water_budget.clamp(0.0, 3.0);
        self.temperature_offset_c = self.temperature_offset_c.clamp(-20.0, 20.0);
        self.axial_tilt_deg = self.axial_tilt_deg.clamp(0.0, 45.0);
        self.precip_multiplier = self.precip_multiplier.clamp(0.25, 4.0);
        self.tectonic_vigor = self.tectonic_vigor.clamp(0.25, 2.0);
        self.hotspot_count = self.hotspot_count.min(30);
        self.craton_count = self.craton_count.clamp(4, 30);
        self.glacial_intensity = self.glacial_intensity.clamp(0.0, 2.0);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cell_counts() {
        let c = PlanetConfig {
            subdivision_level: 6,
            ..Default::default()
        };
        assert_eq!(c.n_cells(), 40_962);
        let c = PlanetConfig {
            subdivision_level: 8,
            ..c
        };
        assert_eq!(c.n_cells(), 655_362);
    }

    #[test]
    fn steps() {
        let c = PlanetConfig::default();
        assert_eq!(c.steps_in(Phase::CrustalFormation), 200);
        assert_eq!(c.steps_in(Phase::Drift), 400);
        assert_eq!(c.steps_in(Phase::RecentPast), 400);
    }
}
