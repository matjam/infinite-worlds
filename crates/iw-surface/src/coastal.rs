//! Wave erosion on exposed shorelines.
//!
//! A land cell with at least [`MIN_FETCH_NEIGHBOURS`] ocean neighbours faces
//! open water on most sides — a headland or an exposed strand rather than the
//! back of a bay — and retreats at a rate set by the rock it is cut in. The
//! debris goes into the shallowest adjacent sea cell, where the fluvial and
//! lithification passes treat it like any other clastic supply: barrier sands
//! in the shallows, mud further out.

use iw_core::Planet;

use crate::Ctx;

/// Ocean neighbours required for a shoreline cell to count as exposed.
pub const MIN_FETCH_NEIGHBOURS: usize = 3;
/// Surface lowering of an exposed shore in average rock, m/yr.
pub const WAVE_EROSION_M_PER_YR: f32 = 2.0e-5;

/// Cut exposed shorelines and dump the debris just offshore.
pub fn run(planet: &mut Planet, ctx: &mut Ctx<'_>) {
    let n = planet.n_cells();
    let sea = planet.sea_level_m;
    let dt_yr = ctx.dt_yr as f32;
    let mesh = ctx.mesh;

    for cell in 0..n as u32 {
        let i = cell as usize;
        if planet.elevation_m[i] < sea || planet.ice_thickness_m[i] > 0.0 {
            continue;
        }
        let mut fetch = 0usize;
        let mut sink = u32::MAX;
        let mut sink_elev = f32::NEG_INFINITY;
        for &m in mesh.neighbors_of(cell) {
            let e = planet.elevation_m[m as usize];
            if e < sea {
                fetch += 1;
                // Shallowest neighbour: the surf zone, where the bar builds.
                if e > sink_elev || (e == sink_elev && m < sink) {
                    sink_elev = e;
                    sink = m;
                }
            }
        }
        if fetch < MIN_FETCH_NEIGHBOURS || sink == u32::MAX {
            continue;
        }
        let erod = crate::mass::surface_erodibility(planet, cell);
        let cut = WAVE_EROSION_M_PER_YR * erod * dt_yr;
        let removed = ctx.mover.erode(planet, cell, cut);
        if removed <= 0.0 {
            continue;
        }
        let volume = removed as f64 * ctx.geom.area_m2[i];
        let thickness = (volume / ctx.geom.area_m2[sink as usize]) as f32;
        ctx.mover.deposit_loose(planet, sink, thickness);
    }
}
