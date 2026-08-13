//! Surface processes: the machinery that turns tectonic relief into landscape.
//!
//! [`SurfaceProcess`] implements [`iw_core::Process`] and runs the DESIGN.md
//! §5 Phase 3/4 surface engine: depression-filled drainage, stream-power
//! incision with downstream transport, hillslope weathering and creep, ice
//! sheets and valley glaciers, aeolian deflation and wave-cut coasts.
//!
//! # Phase behaviour
//!
//! | Phase | What runs |
//! |---|---|
//! | CrustalFormation | nothing — there is no atmosphere-driven landscape yet |
//! | Drift | nothing — tectonics still dominates at 0.5 Myr steps |
//! | Refinement | drainage, fluvial, hillslope, lithification (coarse: [`REFINEMENT_K_MULT`]) |
//! | RecentPast | the full set, glaciers first so meltwater joins the rivers |
//!
//! # Field ownership
//!
//! Writes `sediment_m`, `ice_thickness_m`, `water_flux_m3_yr`, `lake_depth_m`,
//! the strata columns, and `crust_thickness_m` *only* to keep it consistent
//! with column changes (see [`mass`] for the contract). `elevation_m` is read,
//! never written: iw-geology recomputes it from crust, sediment and ice loads
//! every step, which is also why a step's erosion shows up as surface lowering
//! only at the Airy ratio `(rho_m - rho_c)/rho_m ~ 0.18` — removing a kilometre
//! of granite drops the ground about 180 m and floats the root up the rest.
//!
//! # Determinism and statelessness
//!
//! Every serial pass walks cells in id order or in a total order derived from
//! the state (fill level, ice surface) with ties broken by cell id. `rayon` is
//! used only for per-cell maps with no reduction. Nothing survives between
//! steps except mesh-derived geometry and scratch buffers, so dropping in a
//! fresh [`SurfaceProcess`] mid-run — as checkpoint resume does — continues
//! bit-identically. The pass never draws from `ctx.rng`.
//!
//! # Cost
//!
//! The drainage solve is a serial binary heap plus a serial topological sweep,
//! `O(n log n)`. That is the floor for correct flow routing and it dominates
//! the step. See `tests/perf.rs` for measured numbers.

#![warn(missing_docs)]

pub mod aeolian;
pub mod coastal;
pub mod fluvial;
pub mod geom;
pub mod glacial;
pub mod hillslope;
pub mod hydro;
pub mod mass;

use iw_core::{Phase, Planet, Process, RockType, StepCtx};
use iw_mesh::Mesh;

use crate::geom::Geometry;
use crate::glacial::GlacialScratch;
use crate::hydro::Hydrology;
use crate::mass::MassMover;

/// Stream-power multiplier in Refinement. Phase 3 has to age a landscape over
/// tens of Myr in 0.25 Myr steps and only carries fluvial and hillslope
/// processes, so it runs them harder than the kyr-resolution Phase 4.
pub const REFINEMENT_K_MULT: f32 = 2.0;

/// Myr -> yr.
const YEARS_PER_MYR: f64 = 1.0e6;

/// Shared inputs for one step's passes: the mesh, its SI geometry, the
/// timestep in years, and the ledger-aware mass mover.
pub struct Ctx<'a> {
    /// Cell topology.
    pub mesh: &'a Mesh,
    /// Areas and edge lengths in metres.
    pub geom: &'a Geometry,
    /// Timestep in years.
    pub dt_yr: f64,
    /// All erosion and deposition goes through here.
    pub mover: MassMover<'a>,
}

/// Rivers, hillslopes, glaciers, wind and waves.
///
/// All fields are reusable scratch or mesh-derived caches; see the crate docs
/// on statelessness.
#[derive(Debug, Default)]
pub struct SurfaceProcess {
    geom: Option<Geometry>,
    hydro: Hydrology,
    glacial: GlacialScratch,
    melt_m_yr: Vec<f32>,
    flux_m3: Vec<f64>,
    scratch_a: Vec<f32>,
    scratch_b: Vec<f32>,
}

impl SurfaceProcess {
    /// New process with empty buffers.
    pub fn new() -> SurfaceProcess {
        SurfaceProcess::default()
    }

    /// The drainage solution from the most recent step (tests and debug views).
    pub fn hydrology(&self) -> &Hydrology {
        &self.hydro
    }
}

impl Process for SurfaceProcess {
    fn name(&self) -> &'static str {
        "surface"
    }

    fn step(&mut self, planet: &mut Planet, mesh: &Mesh, dt_myr: f64, ctx: &mut StepCtx) {
        debug_assert_eq!(planet.n_cells(), mesh.n_cells());
        let full = match planet.phase {
            Phase::CrustalFormation | Phase::Drift => return,
            Phase::Refinement => false,
            Phase::RecentPast => true,
        };
        if dt_myr <= 0.0 || planet.n_cells() == 0 {
            return;
        }

        let SurfaceProcess {
            geom,
            hydro,
            glacial,
            melt_m_yr,
            flux_m3,
            scratch_a,
            scratch_b,
        } = self;

        if geom.as_ref().is_none_or(|g| g.n_cells != mesh.n_cells()) {
            *geom = Some(Geometry::build(mesh));
        }
        let geom = geom.as_ref().expect("geometry built above");

        let mut c = Ctx {
            mesh,
            geom,
            dt_yr: dt_myr * YEARS_PER_MYR,
            mover: MassMover {
                area_m2: &geom.area_m2,
                ledger: ctx.ledger,
            },
        };

        if full {
            glacial::run(planet, &mut c, glacial, melt_m_yr);
        } else {
            melt_m_yr.clear();
        }

        hydro.solve(planet, mesh, geom, melt_m_yr);
        hydro.publish_flow_edges(planet, mesh);
        precipitate_evaporites(planet, &mut c, hydro);

        let k_mult = if full { 1.0 } else { REFINEMENT_K_MULT };
        fluvial::run(planet, &mut c, hydro, flux_m3, k_mult);

        hillslope::weather(planet, &mut c, scratch_a);
        hillslope::diffuse(planet, &mut c, scratch_b);

        if full {
            aeolian::run(planet, &mut c, scratch_a, scratch_b);
            coastal::run(planet, &mut c);
        }

        mass::lithify(planet, mesh, geom, hydro);
    }
}

/// Lay down evaporite wherever a closed basin boiled off its inflow this step.
fn precipitate_evaporites(planet: &mut Planet, ctx: &mut Ctx<'_>, hydro: &Hydrology) {
    let time = planet.time_myr;
    for cell in 0..planet.n_cells() as u32 {
        let i = cell as usize;
        let evap = hydro.closed_evap_m3_yr[i];
        if evap <= 0.0 {
            continue;
        }
        let volume = evap * hydro::SALT_YIELD_M3_PER_M3 * ctx.dt_yr;
        let thickness = (volume / ctx.geom.area_m2[i]) as f32;
        ctx.mover
            .precipitate(planet, cell, RockType::Evaporite, thickness, time);
    }
}
