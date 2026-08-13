//! Plate boundary extraction and kinematic classification.
//!
//! A boundary edge is a mesh edge whose two cells belong to different plates.
//! Its kind follows from the relative velocity of the two plates at the edge
//! midpoint, decomposed into the edge normal (convergence/divergence) and the
//! along-strike component (transform).

use glam::Vec3;
use iw_core::Planet;
use iw_mesh::{Mesh, EARTH_RADIUS_M};

/// Relative motion below this is treated as no motion, m/yr.
const QUIESCENT_M_YR: f32 = 0.002;

/// How two plates are moving with respect to each other across one edge.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Kind {
    /// Plates separating: ridge / rift.
    Divergent,
    /// Plates closing: subduction or collision.
    Convergent,
    /// Lateral slip dominates.
    Transform,
}

/// One plate-boundary edge, with everything the effect passes need.
pub(crate) struct Edge {
    /// Cell on the low-index side.
    pub(crate) a: u32,
    /// Cell on the high-index side.
    pub(crate) b: u32,
    /// Plate of [`Edge::a`], sampled before any reassignment this step.
    pub(crate) pa: u16,
    /// Plate of [`Edge::b`].
    pub(crate) pb: u16,
    /// Unit vector at the edge midpoint.
    pub(crate) mid: Vec3,
    /// Unit tangent at `mid` pointing from `a` to `b`.
    pub(crate) n: Vec3,
    /// Length of the shared cell wall, metres.
    pub(crate) len_m: f32,
    /// Relative velocity `v(b) - v(a)` at `mid`, m/yr.
    pub(crate) rel: Vec3,
    /// Closing rate along `n`; negative means opening. m/yr.
    pub(crate) conv_m_yr: f32,
    pub(crate) kind: Kind,
}

/// Collect every plate-boundary edge, in ascending `(a, b)` order.
pub(crate) fn build_edges(planet: &Planet, mesh: &Mesh) -> Vec<Edge> {
    let n_cells = planet.n_cells();
    let np = planet.plates.len();
    let mut out = Vec::new();
    for a in 0..n_cells as u32 {
        let pa = planet.plate_id[a as usize];
        if pa as usize >= np {
            continue;
        }
        let nb = mesh.neighbors_of(a);
        let corners = mesh.corners_of(a);
        for (k, &b) in nb.iter().enumerate() {
            if b <= a {
                continue;
            }
            let pb = planet.plate_id[b as usize];
            if pb == pa || pb as usize >= np {
                continue;
            }
            let ca = mesh.centers[a as usize];
            let cb = mesh.centers[b as usize];
            let mid = (ca + cb).normalize();
            let d = cb - ca;
            let t = d - mid * mid.dot(d);
            if t.length_squared() < 1e-16 {
                continue;
            }
            let n = t.normalize();
            // The wall between a and its k-th neighbour spans corners k, k+1.
            let v0 = mesh.vertices[corners[k] as usize];
            let v1 = mesh.vertices[corners[(k + 1) % corners.len()] as usize];
            let len_m = ((v1 - v0).length() as f64 * EARTH_RADIUS_M) as f32;

            let va = planet.plates[pa as usize].velocity_m_yr(mid);
            let vb = planet.plates[pb as usize].velocity_m_yr(mid);
            let rel = vb - va;
            let conv = -rel.dot(n);
            let shear = (rel - n * rel.dot(n)).length();
            let kind = if conv > shear && conv > QUIESCENT_M_YR {
                Kind::Convergent
            } else if -conv > shear && -conv > QUIESCENT_M_YR {
                Kind::Divergent
            } else {
                Kind::Transform
            };
            out.push(Edge {
                a,
                b,
                pa,
                pb,
                mid,
                n,
                len_m,
                rel,
                conv_m_yr: conv,
                kind,
            });
        }
    }
    out
}
