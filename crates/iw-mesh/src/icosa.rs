//! Base icosahedron tables.
//!
//! The 12 vertices and 20 faces are hard-coded; faces are wound counter-clockwise
//! seen from outside the sphere (verified by a unit test). The 30-edge list is
//! derived deterministically from the face table so that subdivided-vertex ids are
//! a pure function of these tables.

use glam::DVec3;

/// Icosahedron vertex count.
pub const N_VERTS: usize = 12;
/// Icosahedron face count.
pub const N_FACES: usize = 20;
/// Icosahedron edge count.
pub const N_EDGES: usize = 30;

/// Icosahedron faces as vertex triples, counter-clockwise seen from outside.
pub const FACES: [[u32; 3]; N_FACES] = [
    [0, 11, 5],
    [0, 5, 1],
    [0, 1, 7],
    [0, 7, 10],
    [0, 10, 11],
    [1, 5, 9],
    [5, 11, 4],
    [11, 10, 2],
    [10, 7, 6],
    [7, 1, 8],
    [3, 9, 4],
    [3, 4, 2],
    [3, 2, 6],
    [3, 6, 8],
    [3, 8, 9],
    [4, 9, 5],
    [2, 4, 11],
    [6, 2, 10],
    [8, 6, 7],
    [9, 8, 1],
];

/// The 12 icosahedron vertices as unit vectors.
pub fn vertices() -> [DVec3; N_VERTS] {
    let t = (1.0 + 5.0f64.sqrt()) / 2.0;
    let raw = [
        [-1.0, t, 0.0],
        [1.0, t, 0.0],
        [-1.0, -t, 0.0],
        [1.0, -t, 0.0],
        [0.0, -1.0, t],
        [0.0, 1.0, t],
        [0.0, -1.0, -t],
        [0.0, 1.0, -t],
        [t, 0.0, -1.0],
        [t, 0.0, 1.0],
        [-t, 0.0, -1.0],
        [-t, 0.0, 1.0],
    ];
    let mut out = [DVec3::ZERO; N_VERTS];
    for (o, r) in out.iter_mut().zip(raw.iter()) {
        *o = DVec3::new(r[0], r[1], r[2]).normalize();
    }
    out
}

/// Canonical edge list: `[lo, hi]` pairs, in first-seen order scanning `FACES`.
///
/// Also returns a `12x12` lookup table mapping a vertex pair to its edge index
/// (`u8::MAX` where the pair is not an edge).
pub fn edges() -> ([[u32; 2]; N_EDGES], [[u8; N_VERTS]; N_VERTS]) {
    let mut lookup = [[u8::MAX; N_VERTS]; N_VERTS];
    let mut list = [[0u32; 2]; N_EDGES];
    let mut count = 0usize;
    for f in FACES.iter() {
        for k in 0..3 {
            let (a, b) = (f[k], f[(k + 1) % 3]);
            let (lo, hi) = if a < b { (a, b) } else { (b, a) };
            if lookup[lo as usize][hi as usize] == u8::MAX {
                lookup[lo as usize][hi as usize] = count as u8;
                lookup[hi as usize][lo as usize] = count as u8;
                list[count] = [lo, hi];
                count += 1;
            }
        }
    }
    debug_assert_eq!(count, N_EDGES);
    (list, lookup)
}

/// Unit vectors at the centroid of each icosahedron face.
pub fn face_centers() -> [DVec3; N_FACES] {
    let v = vertices();
    let mut out = [DVec3::ZERO; N_FACES];
    for (o, f) in out.iter_mut().zip(FACES.iter()) {
        *o = (v[f[0] as usize] + v[f[1] as usize] + v[f[2] as usize]).normalize();
    }
    out
}

/// Unit vectors at the centroids of the four sub-triangles of each icosahedron
/// face (the mid-edge subdivision), in a fixed order: corner 0, corner 1,
/// corner 2, middle.
pub fn face_quarter_centers() -> [[DVec3; 4]; N_FACES] {
    let v = vertices();
    let mut out = [[DVec3::ZERO; 4]; N_FACES];
    for (o, f) in out.iter_mut().zip(FACES.iter()) {
        let (a, b, c) = (v[f[0] as usize], v[f[1] as usize], v[f[2] as usize]);
        let ab = (a + b).normalize();
        let bc = (b + c).normalize();
        let ca = (c + a).normalize();
        *o = [
            (a + ab + ca).normalize(),
            (ab + b + bc).normalize(),
            (ca + bc + c).normalize(),
            (ab + bc + ca).normalize(),
        ];
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn faces_are_ccw_from_outside() {
        let v = vertices();
        for f in FACES.iter() {
            let (a, b, c) = (v[f[0] as usize], v[f[1] as usize], v[f[2] as usize]);
            let n = (b - a).cross(c - a);
            assert!(n.dot(a + b + c) > 0.0, "face {f:?} wound inward");
        }
    }

    #[test]
    fn edge_table_is_complete() {
        let (list, lookup) = edges();
        for e in list.iter() {
            assert_ne!(lookup[e[0] as usize][e[1] as usize], u8::MAX);
        }
        // Every vertex has degree 5.
        for row in lookup.iter() {
            let deg = row.iter().filter(|e| **e != u8::MAX).count();
            assert_eq!(deg, 5);
        }
    }
}
