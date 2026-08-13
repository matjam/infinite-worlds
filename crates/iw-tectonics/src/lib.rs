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
//! **CrustalFormation.** `config.craton_count` continental nuclei are seeded by
//! Poisson-disc rejection sampling. A craton's outline is a spherical cap
//! deformed by 3D fBm — domain-warped, then radius-modulated — so coastlines are
//! fractal from continent scale down to cell scale and no two cratons look
//! alike; its crustal thickness is a core-to-edge profile with fBm texture (see
//! [`craton`]). Each craton is its own plate with an Euler pole that
//! random-walks slowly; a supercontinent attractor switches on for the last
//! quarter of the phase. Cratons are advected by *reassignment*: the centre is
//! rotated analytically each step and the shape is re-rasterized **in the
//! craton's own frame**, so the outline is rigid under drift — cells at the
//! leading edge become continental and cells at the trailing edge revert to
//! ocean, without the boundary breathing in and out. Touching cratons weld
//! (union-find) and thereafter move rigidly together, leaving `SUTURE` flags on
//! the contact line.
//!
//! **Hand-off.** The initial plate partition is a graph Voronoi under a
//! noise-warped metric, so plate boundaries meander instead of running as
//! great-circle bisectors, and a sparse ridged-noise crease network is flagged
//! `SUTURE` so later rifting has inherited weakness to follow.
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

use iw_core::noise::Noise3;
use iw_core::{Phase, Planet, Process, StepCtx};
use iw_mesh::Mesh;
use rayon::prelude::*;

mod advect;
mod boundary;
mod craton;
mod crust;
mod drift;
mod geom;
mod phase1;
mod topology;

pub use phase1::craton_min_separation_m;

/// Genesis-epoch tessellation density (docs/voronoi-v2.md §2): cells crowd
/// along the supercontinent's coast-to-be (the fractal outline where all the
/// early visual interest lives), stay moderate across its interior, and sprawl
/// over the open proto-ocean. Later epochs re-derive density from the actual
/// terrain; this one comes straight from the genesis shapes.
///
/// Values in (0, 1]; deterministic in `(seed, craton_count)`.
pub fn genesis_density(seed: u64, craton_count: u32) -> impl Fn(glam::Vec3) -> f32 + Sync {
    // Octave pitch fixed: density needs coastline-scale structure, not
    // cell-scale crenellation.
    let genesis = craton::Genesis::new(seed, craton_count as usize, 30_000.0);
    move |dir: glam::Vec3| match genesis.membership(dir) {
        Some((_, f)) => {
            // Rim (f -> 1) densest: that is the coastline and shelf.
            let rim = ((f - 0.55) / 0.45).clamp(0.0, 1.0);
            0.35 + 0.65 * rim * rim
        }
        None => 0.10,
    }
}

// --- crust parameters (SI, DESIGN.md §5/§6) ---

/// Reference oceanic crust thickness, metres.
pub const OCEANIC_THICKNESS_M: f32 = 7_000.0;
/// Peak departure from that reference, metres. A fixed low-frequency noise
/// field keeps the sea floor from being a perfectly uniform sheet: at oceanic
/// density this is only ~50 m of isostatic relief, so it textures the abyssal
/// plains without competing with real bathymetry.
pub const OCEANIC_THICKNESS_NOISE_M: f32 = 550.0;
/// Frequency of that field on the unit sphere (~2,000 km features).
const OCEANIC_NOISE_FREQ: f32 = 5.0;
const OCEANIC_NOISE_OCTAVES: u32 = 5;
/// Density of newly formed oceanic crust, kg/m^3.
pub const OCEANIC_DENSITY_KG_M3: f32 = 3_000.0;
/// Density ceiling for old, cold oceanic crust, kg/m^3.
pub const OCEANIC_DENSITY_MAX_KG_M3: f32 = 3_300.0;
/// `rho(age) = 3000 + AGE_DENSITY_COEFF * sqrt(age_myr)`, kg/m^3.
pub const AGE_DENSITY_COEFF: f32 = 30.0;
/// Continental crust density, kg/m^3.
pub const CONTINENTAL_DENSITY_KG_M3: f32 = 2_700.0;
/// Continental crust thickness at a craton's edge, metres.
///
/// Calibration: 35 km is the isostatic anchor for +800 m, so the old value put
/// the *outermost* ring of every craton a full 800 m above the geoid and the
/// planet had no drowned margin anywhere. 31 km floats at ~-260 m — but once
/// breakup genuinely fragments the supercontinent (rift-pair weld immunity),
/// every fragment has margins on all sides and at 31 km the flooded-shelf
/// share pulled land down to ~19% of the surface. 33 km lifts margins ~250 m:
/// still a drowned outer ring, land back near Earth's ~29%.
pub const CRATON_EDGE_THICKNESS_M: f32 = 33_000.0;
/// Continental crust thickness at a craton's centre, metres.
///
/// Calibration: 45 km floats at +2.6 km, which is a plateau, not a shield. Real
/// cratons are thick *and* low because their depleted lithospheric roots are
/// cold and dense; this model has no root term, so the crust has to be thinner
/// to stand at a shield-like height. 40 km gave +2.1 km cores and a land mean
/// of 1185 m (Earth: 840) while the margins still drowned — the freeboard
/// SPREAD was the problem, not its centre. 38 km (+1.6 km core) with the
/// 33 km edge puts most craton area in the +200..800 m platform band where
/// Earth's continental area actually lives.
pub const CRATON_CORE_THICKNESS_M: f32 = 38_000.0;
/// Tibet-scale ceiling on crustal thickening, metres.
pub const MAX_CRUST_THICKNESS_M: f32 = 70_000.0;
/// Crustal thickness at which a stretched continental cell breaks and becomes
/// oceanic, metres.
pub const RIFT_BREAKUP_THICKNESS_M: f32 = 20_000.0;
/// Crust thickness of an actively flexing trench cell, metres.
pub const TRENCH_THICKNESS_M: f32 = 4_200.0;

/// Number of cells a plate must have before it may be rifted apart.
pub const MIN_RIFTABLE_CELLS: usize = 24;
/// Hard ceiling on the number of live plates.
/// Safety valve only — never the thing that decides how many plates exist.
/// The steady-state count emerges from the rift/weld/absorb dynamics; this
/// bound exists so a pathological run cannot allocate unbounded plate state.
pub const MAX_PLATES: usize = 64;

/// Tectonics as a [`Process`]. Stateless with respect to the simulation: the
/// struct holds only mesh-derived caches and scratch buffers.
pub struct TectonicsProcess {
    /// Keyed by mesh FINGERPRINT (not n_cells: re-tessellation keeps the
    /// budget, so counts collide across distinct meshes).
    cache: Option<(u64, MeshCache)>,
    genesis: Option<craton::Genesis>,
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
            genesis: None,
            scratch: Scratch::default(),
        }
    }
}

/// Quantities derived from the mesh and the planet seed; rebuilt when either
/// changes. Nothing here is simulation *state* — every field is a pure function
/// of `(mesh, seed)`, so rebuilding it in a fresh process instance is exact.
pub(crate) struct MeshCache {
    /// Seed the seed-dependent fields below were built for.
    pub(crate) seed: u64,
    /// Mean centre-to-centre cell spacing, metres.
    pub(crate) pitch_m: f64,
    /// Per-cell area in m^2 (for mass accounting).
    pub(crate) area_m2: Vec<f64>,
    /// Total planet surface area, m^2.
    pub(crate) total_area_m2: f64,
    /// Reference thickness of fresh oceanic crust per cell, metres: the
    /// constant plus a fixed noise field. New sea floor is created at this
    /// thickness and old sea floor relaxes back to it.
    pub(crate) ocean_thickness_m: Vec<f32>,
}

impl MeshCache {
    fn build(mesh: &Mesh, seed: u64) -> MeshCache {
        let area_m2: Vec<f64> = mesh.areas_km2.iter().map(|a| *a as f64 * 1.0e6).collect();
        let noise = Noise3::new(phase1::noise_seed(seed, "tectonics/ocean-floor"));
        let ocean_thickness_m = mesh
            .centers
            .par_iter()
            .map(|d| {
                OCEANIC_THICKNESS_M
                    + OCEANIC_THICKNESS_NOISE_M
                        * noise.fbm(*d * OCEANIC_NOISE_FREQ, OCEANIC_NOISE_OCTAVES, 2.0, 0.5)
            })
            .collect();
        MeshCache {
            seed,
            pitch_m: geom::cell_pitch_m(mesh),
            total_area_m2: area_m2.iter().sum(),
            area_m2,
            ocean_thickness_m,
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
        let seed = planet.config.seed;
        let fp = mesh.fingerprint();
        if self
            .cache
            .as_ref()
            .is_none_or(|(f, c)| *f != fp || c.seed != seed)
        {
            self.cache = Some((fp, MeshCache::build(mesh, seed)));
        }
        let cache = &self.cache.as_ref().expect("cache just built").1;
        self.scratch.prepare(mesh.n_cells());

        // Only SUTURE survives a step: it records ancient seams that rifting
        // looks for later. Everything else describes this step's activity.
        for f in planet.tectonic_flags.iter_mut() {
            *f &= iw_core::planet::cell_flags::SUTURE;
        }

        match planet.phase {
            Phase::CrustalFormation => {
                let count = planet.config.craton_count as usize;
                if self
                    .genesis
                    .as_ref()
                    .is_none_or(|s| !s.matches(seed, count, cache.pitch_m))
                {
                    self.genesis = Some(craton::Genesis::new(seed, count, cache.pitch_m));
                }
                let genesis = self.genesis.as_ref().expect("genesis just built");
                phase1::step(planet, mesh, dt_myr, ctx, cache, genesis, &mut self.scratch)
            }
            Phase::Drift | Phase::Refinement | Phase::RecentPast => {
                drift::step(planet, mesh, dt_myr, ctx, cache, &mut self.scratch)
            }
        }
    }
}
