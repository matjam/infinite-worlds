//! Stream-power incision and downstream sediment transport.
//!
//! # Incision
//!
//! ```text
//! E = K_f * erodibility * sqrt(Q) * S      [m/yr]
//! ```
//!
//! the detachment-limited stream-power law with the usual `m = 0.5, n = 1`
//! exponents, using discharge directly instead of drainage area (identical up
//! to the runoff coefficient, and it lets arid catchments stop cutting).
//! [`K_F`] is calibrated on a continental river.
//!
//! # Transport
//!
//! Each cell adds what it eroded to the load coming from upstream and compares
//! the total against a capacity `K_t * Q * S`. Excess drops out as loose
//! regolith — floodplains where the gradient dies, lake floors and, at the
//! coast, deltas. Capacity is zero in standing water, so a river mouth deposits
//! its whole load in the cell it enters and its submarine neighbours.
//!
//! Every cell is visited in a single upstream-first sweep, so all material
//! eroded in a step is deposited in the same step: the ledger balances exactly.

use iw_core::Planet;

use crate::hydro::{Hydrology, NO_DOWNSTREAM};
use crate::Ctx;

/// Stream-power coefficient, calibrated so a large river
/// (`Q = 1e12 m^3/yr`, `S = 0.002`) incising average rock cuts ~1 km/Myr:
/// `1e-3 m/yr = K_F * 1.0 * sqrt(1e12) * 0.002`.
pub const K_F: f32 = 5.0e-7;

/// Share of a cell that the channel and its valley floor actually lower.
///
/// [`K_F`] is calibrated on *channel* incision, but a cell here is 27-110 km
/// across and holds a whole catchment, not a gorge. Cell-mean lowering is the
/// channel rate times the fraction of the cell the river works on. Without
/// this the model removes ~1.2 km/Myr of rock from moderately steep terrain —
/// an order of magnitude above the 50-200 m/Myr that orogens actually shed
/// (`tests/surface.rs::denudation_rate_is_geologically_plausible` locks the
/// resulting rate).
pub const CHANNEL_FRACTION: f32 = 0.1;

/// Transport capacity coefficient: `capacity = K_T * Q * S` in m^3/yr. Set so
/// the calibration river carries ~1e9 m^3/yr, the order of the Ganges load.
pub const K_T: f64 = 0.5;

/// Hard cap on incision per step, metres. Only binds in Refinement, whose
/// 0.25 Myr steps would otherwise let one step cut a canyon.
pub const MAX_INCISION_PER_STEP_M: f32 = 250.0;

/// Fraction of the drop to the receiver a single step may remove. Keeps the
/// explicit scheme from inverting the drainage it just solved.
pub const MAX_INCISION_RELIEF_FRACTION: f32 = 0.3;

/// Ice thicker than this shuts fluvial action off; the glacial pass owns the bed.
pub const ICE_SHUTOFF_M: f32 = 50.0;

/// Share of a river's load that settles in the cell where it enters the sea;
/// the rest spreads over that cell's submarine neighbours (the delta front).
pub const DELTA_MOUTH_FRACTION: f32 = 0.5;

/// Run one incision + transport sweep. `k_mult` scales [`K_F`] (Refinement
/// runs coarse and fast).
pub fn run(
    planet: &mut Planet,
    ctx: &mut Ctx<'_>,
    hydro: &Hydrology,
    flux_m3: &mut Vec<f64>,
    k_mult: f32,
) {
    let n = planet.n_cells();
    flux_m3.clear();
    flux_m3.resize(n, 0.0);
    let sea = planet.sea_level_m;
    let dt_yr = ctx.dt_yr;

    for idx in (0..n).rev() {
        let cell = hydro.order[idx];
        let i = cell as usize;
        let area = ctx.geom.area_m2[i];
        let q = hydro.discharge_m3_yr[i];
        let slope = hydro.slope[i] as f64;
        let down = hydro.downstream[i];
        let land = planet.elevation_m[i] >= sea;

        // --- incision ---
        if land
            && !hydro.is_lake[i]
            && down != NO_DOWNSTREAM
            && planet.ice_thickness_m[i] < ICE_SHUTOFF_M
            && q > 0.0
        {
            let erod = crate::mass::surface_erodibility(planet, cell);
            let rate = (K_F * CHANNEL_FRACTION * k_mult) as f64 * erod as f64 * q.sqrt() * slope;
            let relief = hydro.filled_m[i] - hydro.filled_m[down as usize];
            let cap = MAX_INCISION_PER_STEP_M.min(MAX_INCISION_RELIEF_FRACTION * relief);
            let want = ((rate * dt_yr) as f32).min(cap);
            let removed = ctx.mover.erode(planet, cell, want);
            flux_m3[i] += removed as f64 * area;
        }

        // --- transport ---
        let load = flux_m3[i];
        if load <= 0.0 {
            continue;
        }
        if down == NO_DOWNSTREAM {
            drop_terminal(planet, ctx, cell, load, land);
            continue;
        }
        let capacity = if hydro.is_lake[i] {
            0.0
        } else {
            K_T * q * slope * dt_yr
        };
        if load > capacity {
            let excess = load - capacity;
            ctx.mover
                .deposit_loose(planet, cell, (excess / area) as f32);
            flux_m3[down as usize] += capacity;
        } else {
            flux_m3[down as usize] += load;
        }
    }
}

/// Dump a load that has nowhere left to go. In the sea this is a delta: part
/// settles at the mouth, the rest spreads over the submarine neighbours.
fn drop_terminal(planet: &mut Planet, ctx: &mut Ctx<'_>, cell: u32, load_m3: f64, land: bool) {
    let sea = planet.sea_level_m;
    let area = ctx.geom.area_m2[cell as usize];
    if land {
        ctx.mover
            .deposit_loose(planet, cell, (load_m3 / area) as f32);
        return;
    }
    let front: Vec<u32> = ctx
        .mesh
        .neighbors_of(cell)
        .iter()
        .copied()
        .filter(|&m| planet.elevation_m[m as usize] < sea)
        .collect();
    if front.is_empty() {
        ctx.mover
            .deposit_loose(planet, cell, (load_m3 / area) as f32);
        return;
    }
    let mouth = load_m3 * DELTA_MOUTH_FRACTION as f64;
    ctx.mover.deposit_loose(planet, cell, (mouth / area) as f32);
    let each = (load_m3 - mouth) / front.len() as f64;
    for m in front {
        let a = ctx.geom.area_m2[m as usize];
        ctx.mover.deposit_loose(planet, m, (each / a) as f32);
    }
}
