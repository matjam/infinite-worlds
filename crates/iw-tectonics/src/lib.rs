//! Plate tectonics for Infinite Worlds (DESIGN.md §5, IMPLEMENTATION_PLAN.md §3 WP4).
//!
//! # What this process owns
//!
//! `plate_id`, `plates`, `crust_type`, `crust_thickness_m`, `crust_density_kg_m3`,
//! `crust_age_myr`, `tectonic_flags`, `hotspots`, and the creation/destruction of
//! stratigraphic columns (new oceanic crust, arc and plume volcanics, subducted
//! columns). It never writes `elevation_m` or `sea_level_m` — isostasy in
//! `iw-geology` is the elevation authority and derives relief from the thickness
//! and density fields written here.
//!
//! # Model
//!
//! **CrustalFormation.** `config.craton_count` continental discs are seeded by
//! Poisson-disc rejection sampling. Each craton is its own plate with an Euler
//! pole that random-walks slowly; a supercontinent attractor switches on for the
//! last quarter of the phase. Cratons are advected by *reassignment*: the disc
//! centre is rotated analytically each step and the disc is re-rasterized, so
//! cells at the leading edge become continental and cells at the trailing edge
//! revert to ocean. Touching cratons weld (union-find) and thereafter move
//! rigidly together, leaving `SUTURE` flags on the contact line.
//!
//! **Drift / Refinement / RecentPast.** Plates are rigid caps rotating about
//! Euler poles updated each step from slab pull, ridge push, collision
//! resistance and basal drag. Cell fields do not advect; instead plate
//! *boundaries* migrate. Each boundary edge is classified by the sign of the
//! relative velocity along the edge normal, then:
//!
//! - continuous, rate-proportional effects run every step (arc volcanism,
//!   collisional thickening, rift thinning, trench flexure);
//! - discrete cell reassignment is drawn with probability
//!   `|v_rel| * dt / cell_pitch` against `ctx.rng`. That is stateless and
//!   correct in expectation, which is why no cross-step accumulator is needed
//!   and why a fresh `TectonicsProcess` resumed from a checkpoint reproduces a
//!   straight-through run exactly.
//!
//! # Determinism
//!
//! All evolution state lives in `Planet`. `TectonicsProcess` holds only mesh-
//! derived caches and scratch buffers, so two instances given the same `Planet`
//! produce the same next state. Collections built from hash sets are sorted
//! before they are iterated; `rayon` is used only for pure per-cell maps.

#![warn(missing_docs)]

use iw_core::{Phase, Planet, Process, StepCtx};
use iw_mesh::Mesh;

mod boundary;
mod crust;
mod drift;
mod geom;
mod phase1;
mod topology;

pub use phase1::craton_min_separation_m;

// --- crust parameters (SI, DESIGN.md §5/§6) ---

/// Reference oceanic crust thickness, metres.
pub const OCEANIC_THICKNESS_M: f32 = 7_000.0;
/// Density of newly formed oceanic crust, kg/m^3.
pub const OCEANIC_DENSITY_KG_M3: f32 = 3_000.0;
/// Density ceiling for old, cold oceanic crust, kg/m^3.
pub const OCEANIC_DENSITY_MAX_KG_M3: f32 = 3_300.0;
/// `rho(age) = 3000 + AGE_DENSITY_COEFF * sqrt(age_myr)`, kg/m^3.
pub const AGE_DENSITY_COEFF: f32 = 30.0;
/// Continental crust density, kg/m^3.
pub const CONTINENTAL_DENSITY_KG_M3: f32 = 2_700.0;
/// Continental crust thickness at a craton's edge, metres.
pub const CRATON_EDGE_THICKNESS_M: f32 = 35_000.0;
/// Continental crust thickness at a craton's centre, metres.
pub const CRATON_CORE_THICKNESS_M: f32 = 45_000.0;
/// Tibet-scale ceiling on crustal thickening, metres.
pub const MAX_CRUST_THICKNESS_M: f32 = 70_000.0;
/// Crustal thickness at which a stretched continental cell breaks and becomes
/// oceanic, metres.
pub const RIFT_BREAKUP_THICKNESS_M: f32 = 20_000.0;
/// Crust thickness of an actively flexing trench cell, metres.
pub const TRENCH_THICKNESS_M: f32 = 5_000.0;

/// Number of cells a plate must have before it may be rifted apart.
pub const MIN_RIFTABLE_CELLS: usize = 24;
/// Hard ceiling on the number of live plates.
pub const MAX_PLATES: usize = 16;

/// Tectonics as a [`Process`]. Stateless with respect to the simulation: the
/// struct holds only mesh-derived caches and scratch buffers.
pub struct TectonicsProcess {
    cache: Option<MeshCache>,
    scratch: Scratch,
}

impl Default for TectonicsProcess {
    fn default() -> Self {
        Self::new()
    }
}

impl TectonicsProcess {
    /// Create a process instance. Cheap: caches are built on the first step.
    pub fn new() -> TectonicsProcess {
        TectonicsProcess {
            cache: None,
            scratch: Scratch::default(),
        }
    }
}

/// Quantities derived from the mesh alone; rebuilt when the mesh changes.
pub(crate) struct MeshCache {
    pub(crate) n_cells: usize,
    /// Mean centre-to-centre cell spacing, metres.
    pub(crate) pitch_m: f64,
    /// Per-cell area in m^2 (for mass accounting).
    pub(crate) area_m2: Vec<f64>,
    /// Total planet surface area, m^2.
    pub(crate) total_area_m2: f64,
}

impl MeshCache {
    fn build(mesh: &Mesh) -> MeshCache {
        let area_m2: Vec<f64> = mesh.areas_km2.iter().map(|a| *a as f64 * 1.0e6).collect();
        MeshCache {
            n_cells: mesh.n_cells(),
            pitch_m: geom::cell_pitch_m(mesh),
            total_area_m2: area_m2.iter().sum(),
            area_m2,
        }
    }
}

/// Reusable per-step buffers. Contents are meaningless between steps.
#[derive(Default)]
pub(crate) struct Scratch {
    pub(crate) u32a: Vec<u32>,
    pub(crate) u32b: Vec<u32>,
    pub(crate) u16a: Vec<u16>,
    pub(crate) f32a: Vec<f32>,
    pub(crate) f32b: Vec<f32>,
    pub(crate) flags: Vec<bool>,
    pub(crate) cells: Vec<u32>,
}

impl Scratch {
    fn prepare(&mut self, n: usize) {
        self.u32a.clear();
        self.u32a.resize(n, 0);
        self.u32b.clear();
        self.u32b.resize(n, 0);
        self.u16a.clear();
        self.u16a.resize(n, u16::MAX);
        self.f32a.clear();
        self.f32a.resize(n, 0.0);
        self.f32b.clear();
        self.f32b.resize(n, 0.0);
        self.flags.clear();
        self.flags.resize(n, false);
        self.cells.clear();
    }
}

impl Process for TectonicsProcess {
    fn name(&self) -> &'static str {
        "tectonics"
    }

    fn step(&mut self, planet: &mut Planet, mesh: &Mesh, dt_myr: f64, ctx: &mut StepCtx) {
        assert_eq!(
            planet.n_cells(),
            mesh.n_cells(),
            "planet/mesh cell count mismatch"
        );
        if self
            .cache
            .as_ref()
            .is_none_or(|c| c.n_cells != mesh.n_cells())
        {
            self.cache = Some(MeshCache::build(mesh));
        }
        let cache = self.cache.as_ref().expect("cache just built");
        self.scratch.prepare(mesh.n_cells());

        // Only SUTURE survives a step: it records ancient seams that rifting
        // looks for later. Everything else describes this step's activity.
        for f in planet.tectonic_flags.iter_mut() {
            *f &= iw_core::planet::cell_flags::SUTURE;
        }

        match planet.phase {
            Phase::CrustalFormation => {
                phase1::step(planet, mesh, dt_myr, ctx, cache, &mut self.scratch)
            }
            Phase::Drift | Phase::Refinement | Phase::RecentPast => {
                drift::step(planet, mesh, dt_myr, ctx, cache, &mut self.scratch)
            }
        }
    }
}
