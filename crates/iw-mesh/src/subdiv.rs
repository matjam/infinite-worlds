//! Icosahedron subdivision at frequency `n = 2^level`.
//!
//! Produces the primal geodesic grid: `10n^2 + 2` vertices and `20n^2` triangles,
//! both with globally fixed, hash-free indexing so the output is bit-identical
//! across runs, platforms and thread counts.
//!
//! Vertex id layout:
//! * `0..12`          — the original icosahedron vertices (these become the 12 pentagons)
//! * `12..12+30(n-1)` — edge-interior vertices, `n-1` per canonical edge
//! * rest             — face-interior vertices, `(n-1)(n-2)/2` per face
//!
//! Triangle id layout: `n^2` per face, in face order; within a face the
//! `n(n+1)/2` upward triangles first, then the `n(n-1)/2` downward ones.

use glam::{DVec3, Vec3};
use rayon::prelude::*;

use crate::icosa;

/// Lattice/indexing scheme for one subdivision frequency.
pub struct Subdiv {
    /// Subdivision frequency, `2^level`.
    pub n: u32,
    /// Per face, per edge slot (AB, BC, AC): canonical edge index and whether the
    /// face's traversal direction matches the canonical `[lo, hi]` direction.
    face_edges: [[(u32, bool); 3]; icosa::N_FACES],
    edge_base: u32,
    face_base: u32,
    face_interior: u32,
}

impl Subdiv {
    /// Build the indexing scheme for frequency `n` (a power of two, `>= 1`).
    pub fn new(n: u32) -> Subdiv {
        let (_, lookup) = icosa::edges();
        let mut face_edges = [[(0u32, true); 3]; icosa::N_FACES];
        for (fe, f) in face_edges.iter_mut().zip(icosa::FACES.iter()) {
            let slots = [(f[0], f[1]), (f[1], f[2]), (f[0], f[2])];
            for (slot, (a, b)) in fe.iter_mut().zip(slots.iter()) {
                let e = lookup[*a as usize][*b as usize];
                debug_assert_ne!(e, u8::MAX);
                *slot = (e as u32, a < b);
            }
        }
        Subdiv {
            n,
            face_edges,
            edge_base: icosa::N_VERTS as u32,
            face_base: icosa::N_VERTS as u32 + icosa::N_EDGES as u32 * (n - 1),
            face_interior: (n - 1) * (n.saturating_sub(2)) / 2,
        }
    }

    /// Number of primal vertices, i.e. the number of dual cells.
    #[inline]
    pub fn n_verts(&self) -> usize {
        10 * (self.n as usize) * (self.n as usize) + 2
    }

    /// Number of primal triangles, i.e. the number of dual corner vertices.
    #[inline]
    pub fn n_tris(&self) -> usize {
        20 * (self.n as usize) * (self.n as usize)
    }

    /// Global vertex id of lattice point `(i, j)` on face `f`, `0 <= j <= i <= n`.
    ///
    /// Barycentric weights are `A: n-i`, `B: i-j`, `C: j`, so `(0,0) = A`,
    /// `(n,0) = B`, `(n,n) = C`.
    #[inline]
    pub fn vid(&self, f: usize, i: u32, j: u32) -> u32 {
        let n = self.n;
        let fv = &icosa::FACES[f];
        debug_assert!(j <= i && i <= n);
        if i == 0 {
            return fv[0];
        }
        if i == n && j == 0 {
            return fv[1];
        }
        if i == n && j == n {
            return fv[2];
        }
        let edge_vert = |slot: usize, p: u32| {
            let (e, fwd) = self.face_edges[f][slot];
            let p = if fwd { p } else { n - p };
            self.edge_base + e * (n - 1) + (p - 1)
        };
        if j == 0 {
            return edge_vert(0, i);
        }
        if i == n {
            return edge_vert(1, j);
        }
        if i == j {
            return edge_vert(2, i);
        }
        self.face_base + f as u32 * self.face_interior + (i - 1) * (i - 2) / 2 + (j - 1)
    }

    /// Unit-vector position of every primal vertex, indexed by [`Subdiv::vid`].
    pub fn vertex_positions(&self) -> Vec<Vec3> {
        let n = self.n;
        let iv = icosa::vertices();
        let (edge_list, _) = icosa::edges();
        let mut pos = vec![Vec3::ZERO; self.n_verts()];

        let (base, rest) = pos.split_at_mut(icosa::N_VERTS);
        for (p, v) in base.iter_mut().zip(iv.iter()) {
            *p = v.as_vec3();
        }
        let (edge_region, face_region) = rest.split_at_mut(icosa::N_EDGES * (n as usize - 1));

        if n > 1 {
            edge_region
                .par_chunks_mut(n as usize - 1)
                .enumerate()
                .for_each(|(e, chunk)| {
                    let a = iv[edge_list[e][0] as usize];
                    let b = iv[edge_list[e][1] as usize];
                    for (k, slot) in chunk.iter_mut().enumerate() {
                        let k = k as u32 + 1;
                        *slot = slerp(a, b, k as f64 / n as f64).as_vec3();
                    }
                });
        }
        if self.face_interior > 0 {
            face_region
                .par_chunks_mut(self.face_interior as usize)
                .enumerate()
                .for_each(|(f, chunk)| {
                    let fv = &icosa::FACES[f];
                    let (a, b, c) = (iv[fv[0] as usize], iv[fv[1] as usize], iv[fv[2] as usize]);
                    let inv = 1.0 / n as f64;
                    for i in 2..n {
                        for j in 1..i {
                            let off = ((i - 1) * (i - 2) / 2 + (j - 1)) as usize;
                            let (wa, wb, wc) =
                                ((n - i) as f64 * inv, (i - j) as f64 * inv, j as f64 * inv);
                            chunk[off] = spherical_barycentric(a, b, c, wa, wb, wc).as_vec3();
                        }
                    }
                });
        }
        pos
    }

    /// Primal triangles as vertex-id triples, wound counter-clockwise seen from
    /// outside. Indexed by triangle id (= dual corner vertex id).
    pub fn triangles(&self) -> Vec<[u32; 3]> {
        let n = self.n;
        let per_face = (n * n) as usize;
        let mut tris = vec![[0u32; 3]; self.n_tris()];
        tris.par_chunks_mut(per_face)
            .enumerate()
            .for_each(|(f, chunk)| {
                let n_up = (n * (n + 1) / 2) as usize;
                for i in 1..=n {
                    for j in 0..i {
                        let off = (i * (i - 1) / 2 + j) as usize;
                        chunk[off] = [
                            self.vid(f, i - 1, j),
                            self.vid(f, i, j),
                            self.vid(f, i, j + 1),
                        ];
                    }
                }
                for i in 2..=n {
                    for j in 0..i - 1 {
                        let off = n_up + ((i - 1) * (i - 2) / 2 + j) as usize;
                        chunk[off] = [
                            self.vid(f, i - 1, j),
                            self.vid(f, i, j + 1),
                            self.vid(f, i - 1, j + 1),
                        ];
                    }
                }
            });
        tris
    }
}

/// Great-circle interpolation between two unit vectors.
///
/// Arc-length uniform in `t`, which is what keeps the lattice spacing even along
/// icosahedron edges; the naive "lerp the chord, then normalize" bunches points
/// at the ends by a factor of ~1.38 and costs a 2.7x cell-area spread.
#[inline]
fn slerp(a: DVec3, b: DVec3, t: f64) -> DVec3 {
    let cos = a.dot(b).clamp(-1.0, 1.0);
    let gamma = cos.acos();
    let s = gamma.sin();
    if s < 1e-12 {
        return (a * (1.0 - t) + b * t).normalize();
    }
    ((a * ((1.0 - t) * gamma).sin() + b * (t * gamma).sin()) / s).normalize()
}

/// Symmetric spherical barycentric point for weights `(wa, wb, wc)` summing to 1.
///
/// Averages the three "slerp from an apex toward the opposite edge"
/// constructions so the result is independent of corner labelling. Reduces to
/// the arc-uniform edge subdivision when a weight is zero, and to the spherical
/// centroid at equal weights.
#[inline]
fn spherical_barycentric(a: DVec3, b: DVec3, c: DVec3, wa: f64, wb: f64, wc: f64) -> DVec3 {
    let pa = slerp(a, slerp(b, c, wc / (wb + wc)), wb + wc);
    let pb = slerp(b, slerp(c, a, wa / (wc + wa)), wc + wa);
    let pc = slerp(c, slerp(a, b, wb / (wa + wb)), wa + wb);
    (pa + pb + pc).normalize()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn vertex_ids_are_a_bijection() {
        for level in 0..=4u8 {
            let n = 1u32 << level;
            let s = Subdiv::new(n);
            let mut seen = vec![0u32; s.n_verts()];
            for f in 0..icosa::N_FACES {
                for i in 0..=n {
                    for j in 0..=i {
                        let v = s.vid(f, i, j);
                        assert!((v as usize) < s.n_verts(), "level {level} vid out of range");
                        seen[v as usize] += 1;
                    }
                }
            }
            assert!(
                seen.iter().all(|c| *c > 0),
                "level {level}: unused vertex id"
            );
        }
    }

    #[test]
    fn triangles_are_outward_ccw() {
        let s = Subdiv::new(4);
        let pos = s.vertex_positions();
        for t in s.triangles() {
            let (a, b, c) = (pos[t[0] as usize], pos[t[1] as usize], pos[t[2] as usize]);
            assert!((b - a).cross(c - a).dot(a + b + c) > 0.0);
        }
    }
}
