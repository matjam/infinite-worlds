//! Bedrock weathering and linear hillslope diffusion.
//!
//! # Weathering
//!
//! Bare rock converts to loose regolith at a base rate of
//! [`WEATHERING_M_PER_YR`], boosted by moisture and by freeze-thaw cycling
//! (peaking where the mean temperature sits near 0 °C) and shut down under a
//! blanket of its own product — the classic exponential soil-shielding law.
//! Rock becomes regolith in place: no elevation change beyond the density
//! difference iw-geology applies, and the ledger sees an erosion and an equal
//! deposition in the same cell.
//!
//! # Diffusion
//!
//! Loose material creeps downslope with flux proportional to the surface
//! gradient. At a 35 km cell pitch the literal soil-creep diffusivity
//! (~0.01 m^2/yr) is meaningless — the cell contains an entire dissected
//! landscape — so it is upscaled by the square of the ratio between the cell
//! pitch and the sub-grid hillslope length it stands for
//! (~450 m), giving [`HILLSLOPE_D_M2_YR`] at level 6 and scaling with pitch^2
//! at other levels. The scheme is explicit, so each edge moves at most
//! [`MAX_DIFFUSION_RELIEF_FRACTION`] of the height difference per step and a
//! cell can never export more regolith than it has.

use iw_core::Planet;
use rayon::prelude::*;

use crate::Ctx;

/// Bare-rock weathering rate at reference conditions, m/yr (50 m/Myr).
pub const WEATHERING_M_PER_YR: f32 = 5.0e-5;
/// Regolith depth over which weathering decays by 1/e, metres.
pub const SOIL_SHIELD_DEPTH_M: f32 = 5.0;
/// Precipitation that gives the reference weathering rate, mm/yr.
pub const WEATHERING_REF_PRECIP_MM: f32 = 1000.0;
/// Half-width of the freeze-thaw temperature window, °C.
pub const FREEZE_THAW_WIDTH_C: f32 = 5.0;

/// Effective hillslope diffusivity at the reference cell pitch, m^2/yr.
pub const HILLSLOPE_D_M2_YR: f32 = 60.0;
/// Cell pitch [`HILLSLOPE_D_M2_YR`] is quoted at (level 6), metres.
pub const HILLSLOPE_REF_PITCH_M: f32 = 35_000.0;
/// Largest share of a neighbour height difference one step may move.
pub const MAX_DIFFUSION_RELIEF_FRACTION: f32 = 0.25;

/// Convert bedrock to regolith wherever rock is exposed to the atmosphere.
pub fn weather(planet: &mut Planet, ctx: &mut Ctx<'_>, rate_scratch: &mut Vec<f32>) {
    let n = planet.n_cells();
    let sea = planet.sea_level_m;
    let dt_yr = ctx.dt_yr as f32;

    // Pure per-cell function of state: safe to evaluate in parallel.
    rate_scratch.clear();
    (0..n)
        .into_par_iter()
        .map(|i| {
            if planet.elevation_m[i] < sea || planet.ice_thickness_m[i] > 0.0 {
                return 0.0;
            }
            let moisture = (planet.precip_mm_yr[i] / WEATHERING_REF_PRECIP_MM).clamp(0.2, 2.0);
            let t = planet.temperature_c[i] / FREEZE_THAW_WIDTH_C;
            let freeze_thaw = 1.0 + (-(t * t)).exp();
            let shield = (-planet.sediment_m[i] / SOIL_SHIELD_DEPTH_M).exp();
            WEATHERING_M_PER_YR * moisture * freeze_thaw * shield * dt_yr
        })
        .collect_into_vec(rate_scratch);

    for cell in 0..n as u32 {
        let d = rate_scratch[cell as usize];
        if d <= 0.0 {
            continue;
        }
        let removed = ctx.mover.erode_bedrock(planet, cell, d);
        ctx.mover.deposit_loose(planet, cell, removed);
    }
}

/// Creep loose regolith downhill on the current surface.
pub fn diffuse(planet: &mut Planet, ctx: &mut Ctx<'_>, delta: &mut Vec<f32>) {
    let n = planet.n_cells();
    delta.clear();
    delta.resize(n, 0.0);

    let pitch = ctx.geom.mean_pitch_m;
    let d_eff = HILLSLOPE_D_M2_YR * (pitch / HILLSLOPE_REF_PITCH_M).powi(2);
    let coeff = d_eff as f64 * ctx.dt_yr;
    let mesh = ctx.mesh;

    // Voronoi cells have variable valence (typically 4-9, occasionally more);
    // the scratch grows to the widest cell seen and never reallocates after.
    let mut out: Vec<f32> = Vec::with_capacity(12);
    for cell in 0..n as u32 {
        let i = cell as usize;
        let avail = planet.sediment_m[i];
        if avail <= 0.0 {
            continue;
        }
        let h = planet.elevation_m[i];
        let base = mesh.neighbor_offsets[i] as usize;
        let nb = mesh.neighbors_of(cell);
        out.clear();
        out.resize(nb.len(), 0.0);
        let mut total = 0.0f32;
        for (k, &m) in nb.iter().enumerate() {
            let drop = h - planet.elevation_m[m as usize];
            if drop <= 0.0 {
                out[k] = 0.0;
                continue;
            }
            let dist = ctx.geom.dist_m[base + k].max(1.0) as f64;
            let t = (coeff * drop as f64 / (dist * dist)) as f32;
            let t = t.min(MAX_DIFFUSION_RELIEF_FRACTION * drop / nb.len() as f32);
            out[k] = t;
            total += t;
        }
        if total <= 0.0 {
            continue;
        }
        let scale = if total > avail { avail / total } else { 1.0 };
        let area_i = ctx.geom.area_m2[i];
        for (k, &m) in nb.iter().enumerate() {
            let t = out[k] * scale;
            if t <= 0.0 {
                continue;
            }
            delta[i] -= t;
            delta[m as usize] += (t as f64 * area_i / ctx.geom.area_m2[m as usize]) as f32;
        }
    }

    for (s, d) in planet.sediment_m.iter_mut().zip(delta.iter()) {
        *s = (*s + *d).max(0.0);
    }
}
