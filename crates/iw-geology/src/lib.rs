//! Geology: the elevation authority.
//!
//! `iw-geology` owns `Planet::elevation_m` and `Planet::sea_level_m`. No other
//! process may write them — processes change crustal thickness, density and
//! surface load, and elevation follows (DESIGN.md §5).
//!
//! Each step, in this order:
//!
//! 1. [`igneous`] emplacement from the tectonic flags, so the step's elevation
//!    already carries the new rock;
//! 2. [`metamorphism`] of buried strata, every
//!    [`METAMORPHISM_INTERVAL_STEPS`] steps;
//! 3. [`isostasy`]: elevation recomputed from scratch (Airy + flexural
//!    smoothing);
//! 4. [`sea_level`]: hypsometric solve for the water surface.
//!
//! # Statelessness
//!
//! [`GeologyProcess`] is reconstructible from `Planet` alone. Its fields are
//! reused scratch buffers only, and the metamorphism cadence is derived from
//! `planet.step_index` rather than an internal counter, so swapping in a fresh
//! instance mid-run — as checkpoint resume does — changes nothing.
//!
//! # Contracts other work packages depend on
//!
//! - `crust_thickness_m` is the isostatic column: basement plus every solid
//!   layer emplaced into the strata. `StrataColumns` records that material; it
//!   is never added to `crust_thickness_m` a second time. Anything that adds
//!   rock to a column must add the same thickness to `crust_thickness_m`
//!   (geology does this for its own emplacement).
//! - `sediment_m` (loose regolith, surface-owned) and `ice_thickness_m` are
//!   loads *outside* `crust_thickness_m` and are the only fields double-counted
//!   nowhere else.
//! - `elevation_m` is the top of rock + loose sediment, excluding ice.

#![warn(missing_docs)]

pub mod igneous;
pub mod isostasy;
pub mod metamorphism;
pub mod sea_level;

use iw_core::{Planet, Process, StepCtx};
use iw_mesh::Mesh;

/// The metamorphism sweep runs every N steps. Thresholds are unchanged by the
/// cadence — the transition is a fixed point of the rock's own P/T state, not a
/// rate, so sweeping less often only delays a transformation by at most N steps
/// (at Drift resolution, 4 Myr) and never changes its outcome.
pub const METAMORPHISM_INTERVAL_STEPS: u64 = 8;

/// Isostasy, ocean fill, metamorphism and igneous emplacement.
///
/// Runs every step in every phase. See the crate docs for ordering and for the
/// field-ownership contract.
#[derive(Debug, Default)]
pub struct GeologyProcess {
    /// Scratch for the flexure ping-pong buffer.
    elev_scratch: Vec<f32>,
    /// Scratch for the sea-level sort order.
    order: Vec<u32>,
}

impl GeologyProcess {
    /// New process with empty scratch buffers.
    pub fn new() -> Self {
        Self::default()
    }
}

impl Process for GeologyProcess {
    fn name(&self) -> &'static str {
        "geology"
    }

    fn step(&mut self, planet: &mut Planet, mesh: &Mesh, dt_myr: f64, ctx: &mut StepCtx) {
        debug_assert_eq!(planet.n_cells(), mesh.n_cells());

        igneous::emplace(planet, mesh, dt_myr, ctx);

        if planet
            .step_index
            .is_multiple_of(METAMORPHISM_INTERVAL_STEPS)
        {
            metamorphism::sweep(planet);
        }

        isostasy::update_elevation(planet, mesh, &mut self.elev_scratch);

        let water_km3 = sea_level::water_volume_km3(&planet.config, mesh);
        planet.sea_level_m = sea_level::solve_sea_level_m_with(
            &planet.elevation_m,
            &mesh.areas_km2,
            water_km3,
            &mut self.order,
        );
    }
}
