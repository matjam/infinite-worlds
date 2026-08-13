//! Randomized incremental convex hull of points on the unit sphere.
//!
//! For points on a sphere every input point is a hull vertex, and the hull's
//! triangulation IS the spherical Delaunay triangulation — whose dual is the
//! Voronoi tessellation the v2 mesh is built from (docs/voronoi-v2.md).
//!
//! Classic conflict-list incremental construction, expected O(n log n) with a
//! deterministic seeded insertion order. Coordinates are f64; generators come
//! from seeded sampling with a deterministic micro-jitter, so exact
//! degeneracies (four exactly-cocircular points) do not occur in practice, and
//! near-degeneracies only flip which of two valid triangulations of a flat
//! quad is produced — harmless for a Voronoi dual.

use glam::DVec3;

/// One triangular face of the hull, counter-clockwise seen from outside.
#[derive(Debug, Clone, Copy)]
pub struct Face {
    /// First vertex (input point index).
    pub a: u32,
    /// Second vertex, CCW from `a` seen from outside.
    pub b: u32,
    /// Third vertex.
    pub c: u32,
}

/// Signed volume of the tetrahedron (a, b, c, p): positive when `p` is on the
/// outward side of the CCW face (a, b, c).
#[inline]
fn orient(a: DVec3, b: DVec3, c: DVec3, p: DVec3) -> f64 {
    (b - a).cross(c - a).dot(p - a)
}

/// Convex hull of `points` (unit vectors, at least 4, in general position).
/// Returns the CCW triangle list. Deterministic for a given input order.
pub fn convex_hull(points: &[DVec3]) -> Vec<Face> {
    assert!(points.len() >= 4, "hull needs at least 4 points");
    let n = points.len();

    // --- data structures -----------------------------------------------------
    // Faces are stored in a slab with a live flag; half-edge adjacency is kept
    // as `neighbor[face][edge]` = face across edge `edge` (edges 0,1,2 are
    // ab, bc, ca).
    #[derive(Clone, Copy)]
    struct F {
        v: [u32; 3],
        nb: [u32; 3],
        alive: bool,
    }
    const NONE: u32 = u32::MAX;

    let mut faces: Vec<F> = Vec::with_capacity(2 * n);
    // Conflict lists: per face, the uninserted points that can see it; per
    // point, one face it sees.
    let mut face_pts: Vec<Vec<u32>> = Vec::with_capacity(2 * n);
    let mut pt_face: Vec<u32> = vec![NONE; n];

    // --- initial tetrahedron -------------------------------------------------
    // Points 0, 1, the point farthest from segment 01, and the point farthest
    // from that plane. Inputs are sphere points, so any spread choice works;
    // this one is deterministic.
    let p0 = 0u32;
    let mut p1 = 1u32;
    let mut best = -1.0;
    for (i, p) in points.iter().enumerate().skip(1) {
        let d = (*p - points[0]).length_squared();
        if d > best {
            best = d;
            p1 = i as u32;
        }
    }
    let (a, b) = (points[p0 as usize], points[p1 as usize]);
    let mut p2 = NONE;
    let mut best = -1.0;
    for (i, p) in points.iter().enumerate() {
        let d = (*p - a).cross(*p - b).length_squared();
        if i as u32 != p0 && i as u32 != p1 && d > best {
            best = d;
            p2 = i as u32;
        }
    }
    let c = points[p2 as usize];
    let mut p3 = NONE;
    let mut best = 0.0;
    for (i, p) in points.iter().enumerate() {
        let i = i as u32;
        if i == p0 || i == p1 || i == p2 {
            continue;
        }
        let d = orient(a, b, c, *p);
        if d.abs() > best {
            best = d.abs();
            p3 = i;
        }
    }
    assert!(p3 != NONE, "all points coplanar");

    // Orient the base so p3 is behind it, then build the 4 CCW faces.
    let (t0, t1, t2) = if orient(a, b, c, points[p3 as usize]) < 0.0 {
        (p0, p1, p2)
    } else {
        (p0, p2, p1)
    };
    let t3 = p3;
    // Faces of tetrahedron (t0,t1,t2,t3), all CCW from outside:
    let init = [
        [t0, t1, t2], // base, t3 behind
        [t1, t0, t3],
        [t2, t1, t3],
        [t0, t2, t3],
    ];
    // neighbor[face][edge]: edge k is (v[k], v[k+1]).
    let init_nb = [
        [1u32, 2, 3], // base: ab->f1(t1,t0), bc->f2(t2,t1), ca->f3(t0,t2)
        [0, 3, 2],
        [0, 1, 3],
        [0, 2, 1],
    ];
    for k in 0..4 {
        faces.push(F {
            v: init[k],
            nb: init_nb[k],
            alive: true,
        });
        face_pts.push(Vec::new());
    }

    // Seed conflict lists.
    for (i, p) in points.iter().enumerate() {
        let i = i as u32;
        if i == t0 || i == t1 || i == t2 || i == t3 {
            continue;
        }
        for f in 0..4u32 {
            let fv = faces[f as usize].v;
            if orient(
                points[fv[0] as usize],
                points[fv[1] as usize],
                points[fv[2] as usize],
                *p,
            ) > 0.0
            {
                face_pts[f as usize].push(i);
                pt_face[i as usize] = f;
                break;
            }
        }
        // A point inside the tetrahedron cannot happen for sphere points, but
        // a point exactly on a face plane could leave it NONE; it is then
        // already on the hull surface of the tet and can be skipped safely
        // only if degenerate input — guarded by the caller's jitter.
    }

    // --- incremental insertion ----------------------------------------------
    let mut visible: Vec<u32> = Vec::new();
    let mut horizon: Vec<(u32, u8)> = Vec::new(); // (face, edge) pairs on the rim
    let mut stack: Vec<u32> = Vec::new();
    let mut orphan_buf: Vec<u32> = Vec::new();
    // Generation-stamped visited/visibility marks: O(1) reset per insertion.
    // mark[f] == gen -> visited & visible; mark[f] == gen-off -> visited,
    // hidden; anything else -> untouched this round.
    let mut mark: Vec<u64> = vec![0; faces.len()];
    let mut gen: u64 = 0;
    const HIDDEN_OFF: u64 = 1 << 63;

    for p in 0..n as u32 {
        let start = pt_face[p as usize];
        if start == NONE {
            continue; // tetrahedron vertex, or numerically inside the hull
        }
        debug_assert!(
            faces[start as usize].alive,
            "conflict face died without re-homing its points"
        );
        let pp = points[p as usize];

        // BFS the visible region from the conflict face.
        visible.clear();
        horizon.clear();
        stack.clear();
        stack.push(start);
        gen += 1;
        mark.resize(faces.len().max(mark.len()), 0);
        mark[start as usize] = gen; // start is visible by definition of pt_face
        visible.push(start);
        while let Some(f) = stack.pop() {
            for e in 0..3u8 {
                let g = faces[f as usize].nb[e as usize];
                let m = mark[g as usize];
                if m == gen {
                    continue; // visited, visible
                }
                if m == gen | HIDDEN_OFF {
                    horizon.push((f, e)); // visited, hidden: rim edge
                    continue;
                }
                let gv = faces[g as usize].v;
                let gvis = orient(
                    points[gv[0] as usize],
                    points[gv[1] as usize],
                    points[gv[2] as usize],
                    pp,
                ) > 0.0;
                if gvis {
                    mark[g as usize] = gen;
                    visible.push(g);
                    stack.push(g);
                } else {
                    mark[g as usize] = gen | HIDDEN_OFF;
                    horizon.push((f, e));
                }
            }
        }
        if visible.is_empty() {
            continue; // numerically inside; drop
        }

        // Order the horizon into a closed loop. Each horizon entry is an edge
        // (u, v) of a visible face whose neighbor is hidden. Walk them into a
        // cycle by matching endpoints.
        // Collect as directed edges (u -> v) with the hidden face across.
        let mut rim: Vec<(u32, u32, u32)> = Vec::new(); // (u, v, hidden_face)
        for &(f, e) in &horizon {
            let fv = faces[f as usize].v;
            let u = fv[e as usize];
            let v = fv[((e + 1) % 3) as usize];
            let hidden = faces[f as usize].nb[e as usize];
            if !rim.iter().any(|r| r.0 == u && r.1 == v) {
                rim.push((u, v, hidden));
            }
        }
        // Chain into a loop.
        let mut loop_edges: Vec<(u32, u32, u32)> = Vec::with_capacity(rim.len());
        loop_edges.push(rim[0]);
        while loop_edges.len() < rim.len() {
            let tail = loop_edges.last().expect("non-empty").1;
            let next = rim
                .iter()
                .find(|r| r.0 == tail && !loop_edges.contains(*r))
                .copied();
            match next {
                Some(e) => loop_edges.push(e),
                None => break, // open chain = numerical trouble; bail on point
            }
        }
        if loop_edges.len() != rim.len()
            || loop_edges.last().expect("non-empty").1 != loop_edges[0].0
        {
            continue; // degenerate horizon; skip the point (jitter prevents this)
        }

        // Gather orphaned conflict points, kill visible faces.
        orphan_buf.clear();
        for &f in &visible {
            orphan_buf.extend_from_slice(&face_pts[f as usize]);
            face_pts[f as usize].clear();
            faces[f as usize].alive = false;
        }

        // Build the new fan: one face per rim edge (u, v, p).
        let first_new = faces.len() as u32;
        let k = loop_edges.len() as u32;
        for (i, &(u, v, hidden)) in loop_edges.iter().enumerate() {
            let fid = first_new + i as u32;
            let prev = first_new + ((i as u32 + k - 1) % k);
            let next = first_new + ((i as u32 + 1) % k);
            // Fix the hidden face's back-pointer.
            let h = &mut faces[hidden as usize];
            for e in 0..3 {
                if h.v[e] == v && h.v[(e + 1) % 3] == u {
                    h.nb[e] = fid;
                }
            }
            faces.push(F {
                v: [u, v, p],
                nb: [hidden, next, prev],
                alive: true,
            });
            face_pts.push(Vec::new());
        }

        // Re-home orphans among the new faces.
        for &q in &orphan_buf {
            if q == p {
                continue;
            }
            let qq = points[q as usize];
            pt_face[q as usize] = NONE;
            for fid in first_new..faces.len() as u32 {
                let fv = faces[fid as usize].v;
                if orient(
                    points[fv[0] as usize],
                    points[fv[1] as usize],
                    points[fv[2] as usize],
                    qq,
                ) > 0.0
                {
                    face_pts[fid as usize].push(q);
                    pt_face[q as usize] = fid;
                    break;
                }
            }
        }
        pt_face[p as usize] = NONE;
    }

    faces
        .iter()
        .filter(|f| f.alive)
        .map(|f| Face {
            a: f.v[0],
            b: f.v[1],
            c: f.v[2],
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sphere_points(n: usize, seed: u64) -> Vec<DVec3> {
        // Deterministic pseudo-random unit vectors.
        let mut pts = Vec::with_capacity(n);
        let mut state = seed.wrapping_mul(0x9e3779b97f4a7c15).wrapping_add(1);
        let mut next = || {
            state ^= state >> 12;
            state ^= state << 25;
            state ^= state >> 27;
            state.wrapping_mul(0x2545F4914F6CDD1D)
        };
        while pts.len() < n {
            let z = (next() as f64 / u64::MAX as f64) * 2.0 - 1.0;
            let phi = (next() as f64 / u64::MAX as f64) * std::f64::consts::TAU;
            let r = (1.0 - z * z).max(0.0).sqrt();
            pts.push(DVec3::new(r * phi.cos(), r * phi.sin(), z));
        }
        pts
    }

    #[test]
    fn euler_formula_holds() {
        for n in [10usize, 100, 1_000, 10_000] {
            let pts = sphere_points(n, 42);
            let faces = convex_hull(&pts);
            // Sphere points: V = n, F = 2n - 4, E = 3n - 6.
            assert_eq!(faces.len(), 2 * n - 4, "n = {n}");
            let mut on_hull = vec![false; n];
            for f in &faces {
                on_hull[f.a as usize] = true;
                on_hull[f.b as usize] = true;
                on_hull[f.c as usize] = true;
            }
            assert!(on_hull.iter().all(|x| *x), "n = {n}: point off hull");
        }
    }

    #[test]
    fn faces_are_outward_ccw() {
        let pts = sphere_points(500, 7);
        for f in convex_hull(&pts) {
            let (a, b, c) = (pts[f.a as usize], pts[f.b as usize], pts[f.c as usize]);
            let nrm = (b - a).cross(c - a);
            assert!(
                nrm.dot(a + b + c) > 0.0,
                "face ({},{},{}) winds inward",
                f.a,
                f.b,
                f.c
            );
            // No other point in front of the face plane.
            for (i, p) in pts.iter().enumerate() {
                let i = i as u32;
                if i == f.a || i == f.b || i == f.c {
                    continue;
                }
                assert!(
                    orient(a, b, c, *p) <= 1e-12,
                    "point {i} outside face ({},{},{})",
                    f.a,
                    f.b,
                    f.c
                );
            }
        }
    }

    /// Budget guard for the v2 tessellation: a couple of million generators
    /// must hull in seconds, not minutes. Run with --ignored --release.
    #[test]
    #[ignore = "performance measurement; run explicitly in release"]
    fn hull_scales_to_millions() {
        let n = 2_000_000usize;
        let pts = sphere_points(n, 42);
        let t0 = std::time::Instant::now();
        let faces = convex_hull(&pts);
        let dt = t0.elapsed();
        println!("hull of {n} points: {} faces in {dt:.2?}", faces.len());
        assert_eq!(faces.len(), 2 * n - 4);
        assert!(dt.as_secs() < 60, "hull too slow: {dt:?}");
    }

    #[test]
    fn deterministic() {
        let pts = sphere_points(2_000, 1337);
        let f1 = convex_hull(&pts);
        let f2 = convex_hull(&pts);
        assert_eq!(f1.len(), f2.len());
        for (x, y) in f1.iter().zip(&f2) {
            assert_eq!((x.a, x.b, x.c), (y.a, y.b, y.c));
        }
    }
}
