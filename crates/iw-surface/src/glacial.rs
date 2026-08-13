//! Ice sheets and valley glaciers: mass balance, SIA-lite flow, abrasion and
//! moraine deposition. RecentPast only.
//!
//! # Mass balance
//!
//! `planet.temperature_c` already carries the ~100 kyr glacial oscillation that
//! iw-climate bakes in, so this pass only has to respond to it: snow
//! accumulates below [`SNOW_TEMP_C`] at the local precipitation rate, and melts
//! at a degree-day rate driven by the annual mean plus [`MELT_SEASON_C`] (the
//! melt season is summer, not the annual mean). Meltwater is handed to the
//! drainage solve as extra runoff.
//!
//! # Flow
//!
//! Shallow-ice approximation, linearised: the depth-averaged flux through an
//! edge is `C * H^2 * S` on the *ice surface* gradient (bed elevation plus ice),
//! i.e. a column-mean speed `C * H * S`. [`FLOW_C_PER_YR`] gives ~50 m/yr for a
//! 1 km glacier on a 2% surface slope. Two half-steps per call, each CFL-limited
//! to [`CFL_MAX_FRACTION`] of the source thickness, keep the explicit scheme
//! stable at kyr timesteps.
//!
//! # Erosion
//!
//! Abrasion and plucking scale with the sliding speed. Rock type matters less
//! than under running water ([`ROCK_HARDNESS_SCALE`] damps the contrast). The
//! product is carried down the ice-flow graph within the same step and dropped
//! where the ice ablates — terminal and lateral moraines — so the pass keeps no
//! state between steps.
//!
//! Nothing stops the bed being cut below sea level. Where an ice stream reaches
//! the coast that is exactly right: the overdeepened trough is a fjord once the
//! ice goes.

use iw_core::Planet;

use crate::Ctx;

/// Mean temperature below which precipitation falls and stays as snow, °C.
pub const SNOW_TEMP_C: f32 = -2.0;
/// Width of the rain/snow transition below [`SNOW_TEMP_C`], °C.
pub const SNOW_RAMP_C: f32 = 2.0;
/// Share of snowfall that survives sublimation and wind scour.
pub const ACCUMULATION_EFFICIENCY: f32 = 0.5;
/// Degree-day melt factor, m of ice per °C above freezing per year.
pub const MELT_M_PER_C_YR: f32 = 0.5;
/// Warmth of the melt season above the annual mean, °C.
///
/// `planet.temperature_c` is an annual mean, but ablation is a *summer*
/// process: a degree-day sum is positive wherever the warmest weeks clear 0 °C,
/// which on Earth happens well below a 0 °C annual mean. Driving ablation
/// straight off the annual mean instead gave every cell below -2 °C an
/// unopposed accumulation term, so ice grew to its cap anywhere it could form
/// and nothing ever melted it back. Adding the melt-season offset before the
/// max puts the permanent-ice line near a -8 °C annual mean, which is about
/// where Earth's is; `iw-climate`'s per-cell seasonal amplitude is not visible
/// from this crate, so it is a constant.
pub const MELT_SEASON_C: f32 = 9.0;
/// Melt-season offset used instead of [`MELT_SEASON_C`] for ice standing over
/// water, °C.
///
/// Ice that has flowed off the coast is floating, and the sea beneath it is a
/// heat source that never drops below its freezing point — shelves melt from
/// below and calve regardless of the air above them. This model has no calving
/// and no basal term, so without something here a single polar ice cap flows
/// out across every high-latitude ocean cell and stops only when it meets air
/// warm enough to melt it: the observed failure mode was a Southern-Ocean ice
/// belt reaching 50 deg. Doubling the effective melt season over water pins the
/// shelf edge near 74 deg, which is about where Earth's perennial sea ice sits.
pub const OCEAN_MELT_SEASON_C: f32 = 16.0;
/// Ice thickness cap, metres (Antarctic interior scale).
pub const MAX_ICE_M: f32 = 3000.0;

/// SIA-lite flow coefficient, 1/yr. `u = C * H * S`.
pub const FLOW_C_PER_YR: f32 = 2.5;
/// Ice thinner than this does not flow.
pub const MIN_FLOW_ICE_M: f32 = 1.0;
/// Largest share of a cell's ice one sub-pass may export (CFL guard).
pub const CFL_MAX_FRACTION: f32 = 0.4;
/// Largest share of an ice-surface difference one sub-pass may move.
pub const MAX_FLOW_RELIEF_FRACTION: f32 = 0.25;
/// Hexagonal cells: shared edge length is ~1/sqrt(3) of the centre spacing.
pub const EDGE_LENGTH_FACTOR: f32 = 0.577;

/// Glacial erosion per unit of sliding, dimensionless: 50 m/yr of sliding cuts
/// 0.5 mm/yr, the low end of measured Alpine rates.
pub const K_GLACIAL: f32 = 1.0e-5;
/// Rock-type contrast is damped under ice relative to running water.
pub const ROCK_HARDNESS_SCALE: f32 = 0.5;
/// Ice at or below this thickness counts as the terminus for moraine dumping.
pub const TERMINUS_ICE_M: f32 = 10.0;

/// Scratch buffers for the glacial pass, sized to the mesh.
#[derive(Debug, Default, Clone)]
pub struct GlacialScratch {
    surface_m: Vec<f32>,
    delta: Vec<f32>,
    slide_m_yr: Vec<f32>,
    ice_down: Vec<u32>,
    load_m3: Vec<f64>,
    glaciated: Vec<u32>,
}

/// Advance ice by one step. Writes `planet.ice_thickness_m`, erodes bed rock
/// under sliding ice, deposits moraines at termini and fills `melt_m_yr` with
/// the meltwater the drainage solve should route.
pub fn run(
    planet: &mut Planet,
    ctx: &mut Ctx<'_>,
    scratch: &mut GlacialScratch,
    melt_m_yr: &mut Vec<f32>,
) {
    let n = planet.n_cells();
    let dt_yr = ctx.dt_yr;
    let sea = planet.sea_level_m;

    melt_m_yr.clear();
    melt_m_yr.resize(n, 0.0);
    scratch.surface_m.clear();
    scratch.surface_m.resize(n, 0.0);
    scratch.delta.clear();
    scratch.delta.resize(n, 0.0);
    scratch.slide_m_yr.clear();
    scratch.slide_m_yr.resize(n, 0.0);
    scratch.ice_down.clear();
    scratch.ice_down.resize(n, u32::MAX);
    scratch.load_m3.clear();
    scratch.load_m3.resize(n, 0.0);
    scratch.glaciated.clear();

    // --- mass balance ---
    for (i, melt) in melt_m_yr.iter_mut().enumerate() {
        let t = planet.temperature_c[i];
        let precip_m = (planet.precip_mm_yr[i] as f64 / 1000.0).max(0.0) as f32;
        let snow_fraction = ((SNOW_TEMP_C - t) / SNOW_RAMP_C).clamp(0.0, 1.0);
        let grounded = planet.elevation_m[i] >= sea;
        // Snow only settles on ground; open ocean is left to iw-climate.
        let accum = if grounded {
            precip_m * ACCUMULATION_EFFICIENCY * snow_fraction
        } else {
            0.0
        };
        let season = if grounded {
            MELT_SEASON_C
        } else {
            OCEAN_MELT_SEASON_C
        };
        let ablation = MELT_M_PER_C_YR * (t + season).max(0.0);
        let ice = planet.ice_thickness_m[i];
        let net = (accum - ablation) as f64 * dt_yr;
        let new_ice = ((ice as f64 + net).clamp(0.0, MAX_ICE_M as f64)) as f32;
        // Water actually released: bounded by what was there plus what fell.
        let melted = (ablation as f64).min(ice as f64 / dt_yr + accum as f64);
        *melt = melted.max(0.0) as f32;
        planet.ice_thickness_m[i] = new_ice;
    }

    // --- flow: two half-steps ---
    for _ in 0..2 {
        flow_pass(planet, ctx, scratch, dt_yr * 0.5);
    }

    // --- abrasion under sliding ice ---
    for cell in 0..n as u32 {
        let i = cell as usize;
        let slide = scratch.slide_m_yr[i];
        if slide <= 0.0 || planet.ice_thickness_m[i] < MIN_FLOW_ICE_M {
            continue;
        }
        let erod =
            1.0 + ROCK_HARDNESS_SCALE * (crate::mass::surface_erodibility(planet, cell) - 1.0);
        let cut = K_GLACIAL * erod * slide * dt_yr as f32;
        let removed = ctx.mover.erode(planet, cell, cut);
        scratch.load_m3[i] += removed as f64 * ctx.geom.area_m2[i];
    }

    // --- carry the load to the terminus and drop it there ---
    route_moraine(planet, ctx, scratch, melt_m_yr);
}

/// One explicit SIA-lite half-step.
fn flow_pass(planet: &mut Planet, ctx: &mut Ctx<'_>, scratch: &mut GlacialScratch, dt_yr: f64) {
    let n = planet.n_cells();
    let mesh = ctx.mesh;
    for i in 0..n {
        scratch.surface_m[i] = planet.elevation_m[i] + planet.ice_thickness_m[i];
        scratch.delta[i] = 0.0;
    }

    // Voronoi cells have variable valence (typically 4-9, occasionally more);
    // the scratch grows to the widest cell seen and never reallocates after.
    let mut out: Vec<f32> = Vec::with_capacity(12);
    for cell in 0..n as u32 {
        let i = cell as usize;
        let h = planet.ice_thickness_m[i];
        if h < MIN_FLOW_ICE_M {
            continue;
        }
        let s = scratch.surface_m[i];
        let base = mesh.neighbor_offsets[i] as usize;
        let nb = mesh.neighbors_of(cell);
        out.clear();
        out.resize(nb.len(), 0.0);
        let area_i = ctx.geom.area_m2[i];
        let mut total = 0.0f32;
        let mut best_slide = 0.0f32;
        let mut best_cell = u32::MAX;
        let mut best_flux = 0.0f32;
        for (k, &m) in nb.iter().enumerate() {
            let ds = s - scratch.surface_m[m as usize];
            if ds <= 0.0 {
                out[k] = 0.0;
                continue;
            }
            let dist = ctx.geom.dist_m[base + k].max(1.0);
            let slope = ds / dist;
            // q = C H^2 S (m^2/yr) through an edge of length ~0.577*dist.
            let q = FLOW_C_PER_YR * h * h * slope;
            let t = (q as f64 * (EDGE_LENGTH_FACTOR * dist) as f64 * dt_yr / area_i) as f32;
            let t = t.min(MAX_FLOW_RELIEF_FRACTION * ds);
            out[k] = t;
            total += t;
            if t > best_flux {
                best_flux = t;
                best_cell = m;
                best_slide = FLOW_C_PER_YR * h * slope;
            }
        }
        if total <= 0.0 {
            continue;
        }
        let scale = if total > CFL_MAX_FRACTION * h {
            CFL_MAX_FRACTION * h / total
        } else {
            1.0
        };
        for (k, &m) in nb.iter().enumerate() {
            let t = out[k] * scale;
            if t <= 0.0 {
                continue;
            }
            scratch.delta[i] -= t;
            scratch.delta[m as usize] += (t as f64 * area_i / ctx.geom.area_m2[m as usize]) as f32;
        }
        scratch.ice_down[i] = best_cell;
        scratch.slide_m_yr[i] = scratch.slide_m_yr[i].max(best_slide * scale);
    }

    for i in 0..n {
        planet.ice_thickness_m[i] =
            (planet.ice_thickness_m[i] + scratch.delta[i]).clamp(0.0, MAX_ICE_M);
    }
}

/// Push everything the ice quarried down the flow graph until it reaches a
/// melting or ice-free cell, then drop it as loose moraine.
fn route_moraine(
    planet: &mut Planet,
    ctx: &mut Ctx<'_>,
    scratch: &mut GlacialScratch,
    melt_m_yr: &[f32],
) {
    let n = planet.n_cells();
    for i in 0..n {
        if scratch.load_m3[i] > 0.0 || planet.ice_thickness_m[i] > 0.0 {
            scratch.glaciated.push(i as u32);
        }
    }
    // Ice-surface descending, ties by cell id: a single sweep then moves every
    // load past all the cells it can reach this step.
    let surface = &scratch.surface_m;
    scratch.glaciated.sort_unstable_by(|a, b| {
        surface[*b as usize]
            .total_cmp(&surface[*a as usize])
            .then_with(|| a.cmp(b))
    });

    for idx in 0..scratch.glaciated.len() {
        let cell = scratch.glaciated[idx];
        let i = cell as usize;
        let load = scratch.load_m3[i];
        if load <= 0.0 {
            continue;
        }
        let down = scratch.ice_down[i];
        let terminus =
            down == u32::MAX || planet.ice_thickness_m[i] <= TERMINUS_ICE_M || melt_m_yr[i] > 0.0;
        if terminus {
            let thickness = (load / ctx.geom.area_m2[i]) as f32;
            ctx.mover.deposit_loose(planet, cell, thickness);
            scratch.load_m3[i] = 0.0;
        } else {
            scratch.load_m3[down as usize] += load;
            scratch.load_m3[i] = 0.0;
        }
    }

    // Anything still in transit (a cycle in the flow graph cannot happen, but
    // a cell whose receiver sorted above it can) settles where it stands.
    for i in 0..n {
        let load = scratch.load_m3[i];
        if load > 0.0 {
            let thickness = (load / ctx.geom.area_m2[i]) as f32;
            ctx.mover.deposit_loose(planet, i as u32, thickness);
            scratch.load_m3[i] = 0.0;
        }
    }
}
