//! Diagnostic climate model: annual-mean temperature, prescribed wind belts and
//! advected precipitation (DESIGN.md §7).
//!
//! Nothing here integrates in time. Every field is recomputed from the current
//! `Planet` each step, so the process holds no simulation state: dropping a
//! [`ClimateProcess`] mid-run and stepping a fresh one produces identical
//! output. What it does cache is derived purely from the mesh (wind belts, the
//! advection operator) or invalidated by a hash of the land/sea mask
//! (distance-to-ocean), which is why the cache can never disagree with a cold
//! start.
//!
//! # Model
//!
//! **Temperature** — `T = T_zonal(lat) + maritime + offset + glacial − 6.5 C/km
//! × surface_height_above_sea`, with `T_zonal` running from +28 C at
//! the equator to −25 C at the poles, and `maritime` dragging cells within 8
//! cells of open ocean up to 10% of the way toward +14 C. Ice thickness counts
//! toward surface height, so ice sheets cool themselves.
//!
//! **Seasonality** — `Planet::temperature_c` is the *annual mean*. Summer and
//! winter endpoints are derived, not stored: call [`seasonal_amplitude_c`] or
//! [`seasonal_extremes_c`] (O(n), no simulation state required). Amplitude is
//! `8 C * sin²(lat) * (tilt / 23.4°) * (1 + min(dist_to_ocean, 8) / 4)`.
//!
//! # Glacial cycles (read this if you consume `temperature_c`)
//!
//! Glacial forcing is generated **inside this crate**, not injected by
//! `iw-sim`. During [`Phase::RecentPast`](iw_core::Phase::RecentPast) a global
//! anomaly is added to every cell:
//!
//! ```text
//! dT(t) = -6 C * glacial_intensity * (0.5 - 0.5*cos(2*pi * t_recent / 0.1 Myr))
//! ```
//!
//! where `t_recent = time_myr - sum(phase_durations_myr[0..3])`. The phase
//! therefore opens at an interglacial (0 C), bottoms out at
//! `-6 C * intensity` every 100 kyr, and runs ~10 cycles over the default
//! 2 Myr phase. Outside RecentPast the anomaly is exactly 0. `iw-surface`
//! grows and melts ice against this signal; use [`glacial_forcing_c`] rather
//! than re-deriving it.
//!
//! **Winds** — three belts per hemisphere (trades, westerlies, polar
//! easterlies), Coriolis-deflected, smoothly blended across the 30° and 60°
//! boundaries, 5–9 m/s at belt centres and weak in the horse latitudes. Purely
//! a function of latitude and the local tangent basis: see [`wind_vector`].
//!
//! **Precipitation** — ocean and lake cells evaporate (temperature-dependent,
//! up to 2000 mm/yr, sealed by ice) and land cells transpire a fraction of last
//! step's rainfall. That moisture is injected once and then advected downwind
//! for ~14 sweeps at level 6 (the wind is projected onto neighbour directions
//! and the flux split by the positive components), raining out each sweep in
//! proportion to `base + orographic lift + convergence`. Convergence is
//! strongly positive at the ITCZ and the polar front and negative in the
//! subtropics, which produces the wet equator, dry ~30° belts, wet
//! mid-latitudes, rain shadows behind ranges and dry deep interiors. The sweep
//! count and per-sweep rainout are scaled by cell width, so the pattern is the
//! same in kilometres at every subdivision level. The field is finally rescaled
//! so global precipitation matches global evaporation, multiplied by
//! `config.precip_multiplier`, and blended 20% with the previous step's field
//! for temporal stability.
//!
//! One step's advection only carries moisture `sweeps` cells inland, which is
//! far short of the middle of a real continent; land evapotranspiration is what
//! relays it further, so a continental interior takes several steps to settle
//! and stays an order of magnitude drier than its coasts.

#![warn(missing_docs)]

mod geometry;
mod precip;
mod temperature;

use iw_core::{Planet, Process, StepCtx};
use iw_mesh::Mesh;

pub use geometry::{
    wind_components_m_s, wind_field, wind_vector, POLAR_SPEED_M_S, TRADE_SPEED_M_S,
    WESTERLY_SPEED_M_S,
};
pub use precip::{convergence_rainout, evaporation_mm_yr, EVAPORATION_MAX_MM_YR};
pub use temperature::{
    annual_mean_temperature_c, continentality, distance_to_ocean_cells, glacial_forcing_c,
    seasonal_amplitude_at, seasonal_amplitude_c, seasonal_extremes_c, zonal_mean_c,
    GLACIAL_DEPTH_C, GLACIAL_PERIOD_MYR, LAPSE_RATE_C_PER_M, MAX_OCEAN_DISTANCE_CELLS,
    REFERENCE_TILT_DEG, SEASONAL_BASE_C, T_EQUATOR_C, T_POLE_C,
};

use geometry::MeshCache;

/// The climate [`Process`]: writes `temperature_c`, `precip_mm_yr` and
/// `wind_m_s` every step. Cheap enough to run in every phase (< 10 ms at
/// subdivision level 6).
///
/// Holds only mesh-derived and mask-derived caches; see the crate docs.
#[derive(Default)]
pub struct ClimateProcess {
    /// (mesh fingerprint, advection operator). Keyed on the FINGERPRINT, not
    /// n_cells: era re-tessellation swaps the mesh at the same budget, and a
    /// surviving cache indexes a foreign CSR adjacency.
    geometry: Option<(u64, MeshCache)>,
    /// (land/sea mask hash, mesh fingerprint, distance-to-ocean field). The
    /// Dijkstra walks mesh adjacency, so the mesh identity is part of the key.
    ocean_dist: Option<(u64, u64, Vec<u8>)>,
}

impl ClimateProcess {
    /// New process with empty caches.
    pub fn new() -> ClimateProcess {
        ClimateProcess::default()
    }

    fn geometry(&mut self, mesh: &Mesh) -> &MeshCache {
        let fp = mesh.fingerprint();
        let stale = match &self.geometry {
            Some((f, _)) => *f != fp,
            None => true,
        };
        if stale {
            self.geometry = Some((fp, MeshCache::build(mesh)));
        }
        &self.geometry.as_ref().expect("geometry cache built").1
    }

    fn ocean_dist(&mut self, planet: &Planet, mesh: &Mesh) -> &[u8] {
        let hash = temperature::ocean_mask_hash(planet);
        let fp = mesh.fingerprint();
        let stale = match &self.ocean_dist {
            Some((h, f, _)) => *h != hash || *f != fp,
            None => true,
        };
        if stale {
            self.ocean_dist = Some((hash, fp, distance_to_ocean_cells(planet, mesh)));
        }
        &self.ocean_dist.as_ref().expect("distance cache built").2
    }
}

impl Process for ClimateProcess {
    fn name(&self) -> &'static str {
        "climate"
    }

    /// Diagnostic: the fields depend on the planet's current state, not on
    /// `dt_myr`, and no randomness is consumed.
    fn step(&mut self, planet: &mut Planet, mesh: &Mesh, _dt_myr: f64, _ctx: &mut StepCtx) {
        debug_assert_eq!(planet.n_cells(), mesh.n_cells());
        let dist = self.ocean_dist(planet, mesh);
        temperature::update(planet, mesh, dist);

        let geometry = self.geometry(mesh);
        planet.wind_m_s.copy_from_slice(&geometry.wind);
        precip::update(planet, mesh, geometry);
    }
}
