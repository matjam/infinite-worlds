//! Prescribed wind belts and the mesh-derived advection operator.
//!
//! Everything here is a pure function of the mesh: the belts are latitude-only
//! and the advection weights are built from the belts, so the whole structure is
//! cached per mesh and never depends on planet state (see [`MeshCache`]).

use glam::Vec3;
use iw_mesh::Mesh;
use rayon::prelude::*;

/// Peak trade-wind speed, m/s (belt centre, ~15 deg).
pub const TRADE_SPEED_M_S: f32 = 7.0;
/// Peak westerly speed, m/s (belt centre, ~45 deg).
pub const WESTERLY_SPEED_M_S: f32 = 9.0;
/// Peak polar-easterly speed, m/s (belt centre, ~75 deg).
pub const POLAR_SPEED_M_S: f32 = 5.0;

/// Half-width of the smooth blend across a belt boundary, degrees of latitude.
const BLEND_DEG: f32 = 7.5;

/// Unit (eastward, poleward) direction of each belt. Coriolis deflection is
/// baked in: trades blow west and toward the equator, westerlies east and toward
/// the pole, polar easterlies west and toward the equator.
const TRADE_DIR: (f32, f32) = (-0.800, -0.600);
const WESTERLY_DIR: (f32, f32) = (0.920, 0.392);
const POLAR_DIR: (f32, f32) = (-0.707, -0.707);

#[inline]
fn smoothstep(e0: f32, e1: f32, x: f32) -> f32 {
    let t = ((x - e0) / (e1 - e0)).clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}

/// Zonal wind components at a latitude: `(eastward, poleward)` in m/s.
///
/// Three belts per hemisphere blended with a smoothstep over +-7.5 deg around
/// the 30 deg and 60 deg boundaries. Both components are hemisphere-symmetric;
/// "poleward" is turned into a world-space direction by [`wind_vector`].
///
/// The blend deliberately leaves the horse latitudes (~30 deg) and the polar
/// front (~60 deg) with weak, nearly cancelling flow, which is what makes the
/// subtropics dry in the precipitation model.
pub fn wind_components_m_s(lat_rad: f32) -> (f32, f32) {
    let a = lat_rad.to_degrees().abs();
    let s1 = smoothstep(30.0 - BLEND_DEG, 30.0 + BLEND_DEG, a);
    let s2 = smoothstep(60.0 - BLEND_DEG, 60.0 + BLEND_DEG, a);
    let (w_trade, w_west, w_polar) = (1.0 - s1, s1 * (1.0 - s2), s2);

    let east = w_trade * TRADE_DIR.0 * TRADE_SPEED_M_S
        + w_west * WESTERLY_DIR.0 * WESTERLY_SPEED_M_S
        + w_polar * POLAR_DIR.0 * POLAR_SPEED_M_S;
    let poleward = w_trade * TRADE_DIR.1 * TRADE_SPEED_M_S
        + w_west * WESTERLY_DIR.1 * WESTERLY_SPEED_M_S
        + w_polar * POLAR_DIR.1 * POLAR_SPEED_M_S;
    (east, poleward)
}

/// World-space wind vector (m/s) in the tangent plane at a cell, given the
/// cell's latitude and its local `(east, north)` basis.
///
/// The meridional component flips sign across the equator (poleward is +north
/// in the north, -north in the south); the flip is smoothed over ~4 deg so the
/// two trade belts converge at the ITCZ instead of meeting a discontinuity.
pub fn wind_vector(lat_rad: f32, east: Vec3, north: Vec3) -> Vec3 {
    let (u, poleward) = wind_components_m_s(lat_rad);
    let hemi = (lat_rad.to_degrees() / 4.0).tanh();
    east * u + north * (poleward * hemi)
}

/// Wind field for a whole mesh, m/s, tangent to the sphere at every cell.
pub fn wind_field(mesh: &Mesh) -> Vec<Vec3> {
    (0..mesh.n_cells() as u32)
        .into_par_iter()
        .map(|c| {
            let (east, north) = mesh.east_north(c);
            wind_vector(mesh.latlon[c as usize][0], east, north)
        })
        .collect()
}

/// Wind field plus the CSR advection operator derived from it.
///
/// The operator is a normalized outflow split: cell `i` sends its moisture to
/// the neighbours whose direction has a positive projection on the local wind,
/// in proportion to that projection. `rev` lets the sweep run as a gather (each
/// output cell reads its inflows), which keeps it parallel and deterministic.
pub(crate) struct MeshCache {
    /// Per-cell wind vector, m/s.
    pub wind: Vec<Vec3>,
    /// Outflow weight per CSR neighbour slot; the slots of one cell sum to 1
    /// (or to 0 for a cell with no downwind neighbour).
    pub out_w: Vec<f32>,
    /// `rev[k]` is the CSR slot carrying flow *into* cell `i` from the
    /// neighbour stored at slot `k` of cell `i`; `u32::MAX` if the adjacency is
    /// not symmetric (never happens on a well-formed mesh).
    pub rev: Vec<u32>,
    /// 1.0 for cells with no downwind neighbour (moisture stays put), else 0.0.
    pub stay: Vec<f32>,
}

impl MeshCache {
    /// Build the wind field and advection operator for `mesh`.
    pub fn build(mesh: &Mesh) -> MeshCache {
        let n = mesh.n_cells();
        let wind = wind_field(mesh);
        let mut out_w = vec![0.0f32; mesh.neighbors.len()];
        let mut stay = vec![0.0f32; n];

        for i in 0..n {
            let a = mesh.neighbor_offsets[i] as usize;
            let b = mesh.neighbor_offsets[i + 1] as usize;
            let ci = mesh.centers[i];
            let w_hat = wind[i].normalize_or_zero();
            let mut sum = 0.0f32;
            for (k, &j) in mesh.neighbors[a..b].iter().enumerate() {
                let d = mesh.centers[j as usize] - ci;
                let t = (d - ci * d.dot(ci)).normalize_or_zero();
                let w = w_hat.dot(t).max(0.0);
                out_w[a + k] = w;
                sum += w;
            }
            if sum > 0.0 {
                let inv = 1.0 / sum;
                for w in &mut out_w[a..b] {
                    *w *= inv;
                }
            } else {
                stay[i] = 1.0;
            }
        }

        let mut rev = vec![u32::MAX; mesh.neighbors.len()];
        for i in 0..n {
            let a = mesh.neighbor_offsets[i] as usize;
            let b = mesh.neighbor_offsets[i + 1] as usize;
            for (k, &j) in mesh.neighbors[a..b].iter().enumerate() {
                let ja = mesh.neighbor_offsets[j as usize] as usize;
                if let Some(m) = mesh.neighbors_of(j).iter().position(|&x| x == i as u32) {
                    rev[a + k] = (ja + m) as u32;
                }
            }
        }
        debug_assert!(rev.iter().all(|r| *r != u32::MAX), "asymmetric adjacency");

        MeshCache {
            wind,
            out_w,
            rev,
            stay,
        }
    }
}
