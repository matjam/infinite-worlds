//! Wind transport of loose sand and dust in arid country.
//!
//! Arid land cells (below [`ARID_PRECIP_MM_YR`], ice-free, not under water)
//! release a fraction of their regolith per step. The grain walks downwind —
//! the neighbour best aligned with `planet.wind_m_s` — through as many arid
//! cells as it can reach in one step and settles in the first cell that is wet,
//! icy or drowned: the dune field's downwind margin and the loess belt beyond
//! it. Material stays loose, so the next step can pick it up again.
//!
//! Sources read the regolith field as it stood at the start of the pass and
//! all arrivals land in a separate delta buffer, so the result does not depend
//! on the order cells are visited.

use glam::Vec3;
use iw_core::Planet;

use crate::Ctx;

/// Precipitation below which a cell is arid enough to deflate, mm/yr.
pub const ARID_PRECIP_MM_YR: f32 = 250.0;
/// Share of loose regolith entrained per year at the reference wind speed.
pub const DEFLATION_PER_YR: f32 = 2.0e-5;
/// Wind speed the deflation rate is quoted at, m/s.
pub const REFERENCE_WIND_M_S: f32 = 10.0;
/// Most a single step may lift, as a share of the cell's regolith.
pub const MAX_DEFLATION_FRACTION: f32 = 0.5;
/// Longest downwind walk within one step, in cells.
pub const MAX_HOPS: usize = 16;

/// Deflate arid cells and deposit downwind.
pub fn run(planet: &mut Planet, ctx: &mut Ctx<'_>, source: &mut Vec<f32>, delta: &mut Vec<f32>) {
    let n = planet.n_cells();
    let sea = planet.sea_level_m;
    source.clear();
    source.extend_from_slice(&planet.sediment_m);
    delta.clear();
    delta.resize(n, 0.0);

    let mesh = ctx.mesh;
    let dt_yr = ctx.dt_yr as f32;

    let arid = |i: usize, p: &Planet| -> bool {
        p.precip_mm_yr[i] < ARID_PRECIP_MM_YR
            && p.ice_thickness_m[i] <= 0.0
            && p.lake_depth_m[i] <= 0.0
            && p.elevation_m[i] >= sea
    };

    for cell in 0..n as u32 {
        let i = cell as usize;
        if !arid(i, planet) || source[i] <= 0.0 {
            continue;
        }
        let wind = planet.wind_m_s[i];
        let speed = wind.length();
        if speed <= 0.0 {
            continue;
        }
        let frac =
            (DEFLATION_PER_YR * (speed / REFERENCE_WIND_M_S) * dt_yr).min(MAX_DEFLATION_FRACTION);
        let lifted = source[i] * frac;
        if lifted <= 0.0 {
            continue;
        }
        delta[i] -= lifted;

        // Walk downwind until the air runs out of desert.
        let mut here = cell;
        let mut carried_m3 = lifted as f64 * ctx.geom.area_m2[i];
        for _ in 0..MAX_HOPS {
            let Some(next) = downwind(mesh, planet, here) else {
                break;
            };
            here = next;
            if !arid(here as usize, planet) {
                break;
            }
        }
        let a = ctx.geom.area_m2[here as usize];
        if here == cell {
            // Nowhere to go: put it back.
            delta[i] += lifted;
            continue;
        }
        carried_m3 /= a;
        delta[here as usize] += carried_m3 as f32;
    }

    for (s, d) in planet.sediment_m.iter_mut().zip(delta.iter()) {
        *s = (*s + *d).max(0.0);
    }
}

/// Neighbour best aligned with the local wind, if the wind points at one.
fn downwind(mesh: &iw_mesh::Mesh, planet: &Planet, cell: u32) -> Option<u32> {
    let w = planet.wind_m_s[cell as usize];
    if w.length_squared() <= 0.0 {
        return None;
    }
    let w = w.normalize();
    let c = mesh.centers[cell as usize];
    let mut best = None;
    let mut best_dot = 0.0f32;
    for &m in mesh.neighbors_of(cell) {
        let dir = to_tangent(mesh.centers[m as usize], c);
        let d = dir.dot(w);
        if d > best_dot {
            best_dot = d;
            best = Some(m);
        }
    }
    best
}

/// Unit direction from `from` toward `to`, projected into `from`'s tangent plane.
#[inline]
fn to_tangent(to: Vec3, from: Vec3) -> Vec3 {
    let d = to - from * to.dot(from);
    if d.length_squared() > 0.0 {
        d.normalize()
    } else {
        Vec3::ZERO
    }
}
