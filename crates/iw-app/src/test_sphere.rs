//! A self-contained icosphere used for GPU pipeline bring-up (`--test-sphere`).
//!
//! It builds a `Mesh` whose "cells" are the triangles of a subdivided
//! icosahedron rather than Goldberg polygons. That is enough to exercise every
//! renderer path (corner fans, per-cell ids, chunk cones) without depending on
//! `Mesh::build`, and is deliberately not used by default.

use std::collections::HashMap;

use glam::Vec3;
use iw_mesh::{latlon_of, Chunk, Mesh, EARTH_RADIUS_KM};

const X: f32 = 0.525_731_1;
const Z: f32 = 0.850_650_8;

const ICO_VERTS: [[f32; 3]; 12] = [
    [-X, 0.0, Z],
    [X, 0.0, Z],
    [-X, 0.0, -Z],
    [X, 0.0, -Z],
    [0.0, Z, X],
    [0.0, Z, -X],
    [0.0, -Z, X],
    [0.0, -Z, -X],
    [Z, X, 0.0],
    [-Z, X, 0.0],
    [Z, -X, 0.0],
    [-Z, -X, 0.0],
];

const ICO_FACES: [[u32; 3]; 20] = [
    [0, 4, 1],
    [0, 9, 4],
    [9, 5, 4],
    [4, 5, 8],
    [4, 8, 1],
    [8, 10, 1],
    [8, 3, 10],
    [5, 3, 8],
    [5, 2, 3],
    [2, 7, 3],
    [7, 10, 3],
    [7, 6, 10],
    [7, 11, 6],
    [11, 0, 6],
    [0, 1, 6],
    [6, 1, 10],
    [9, 0, 11],
    [9, 11, 2],
    [9, 2, 5],
    [7, 2, 11],
];

/// Build an icosphere `Mesh` with `20 * 4^level` triangular cells.
pub fn build(level: u8) -> Mesh {
    let level = level.min(8);
    let mut vertices: Vec<Vec3> = ICO_VERTS
        .iter()
        .map(|v| Vec3::from(*v).normalize())
        .collect();
    let mut midpoints: HashMap<(u32, u32), u32> = HashMap::new();
    let mut midpoint = |vertices: &mut Vec<Vec3>, a: u32, b: u32| -> u32 {
        let key = if a < b { (a, b) } else { (b, a) };
        *midpoints.entry(key).or_insert_with(|| {
            let m = ((vertices[a as usize] + vertices[b as usize]) * 0.5).normalize();
            vertices.push(m);
            vertices.len() as u32 - 1
        })
    };

    // Keep the parent face index alongside each triangle so chunks can group
    // by face, exactly as the real mesh does.
    let mut tris: Vec<([u32; 3], u16)> = ICO_FACES
        .iter()
        .enumerate()
        .map(|(i, f)| (*f, i as u16))
        .collect();
    for _ in 0..level {
        let mut next = Vec::with_capacity(tris.len() * 4);
        for (t, face) in &tris {
            let (a, b, c) = (t[0], t[1], t[2]);
            let ab = midpoint(&mut vertices, a, b);
            let bc = midpoint(&mut vertices, b, c);
            let ca = midpoint(&mut vertices, c, a);
            next.push(([a, ab, ca], *face));
            next.push(([ab, b, bc], *face));
            next.push(([ca, bc, c], *face));
            next.push(([ab, bc, ca], *face));
        }
        tris = next;
    }

    let n = tris.len();
    let mut centers = Vec::with_capacity(n);
    let mut corners = Vec::with_capacity(n * 3);
    let mut corner_offsets = Vec::with_capacity(n + 1);
    let mut cell_chunk = Vec::with_capacity(n);
    corner_offsets.push(0);
    for (t, face) in &tris {
        let c = ((vertices[t[0] as usize] + vertices[t[1] as usize] + vertices[t[2] as usize])
            / 3.0)
            .normalize();
        centers.push(c);
        corners.extend_from_slice(t);
        corner_offsets.push(corners.len() as u32);
        cell_chunk.push(*face);
    }

    let sphere_area = 4.0 * std::f32::consts::PI * EARTH_RADIUS_KM * EARTH_RADIUS_KM;
    let areas_km2 = vec![sphere_area / n as f32; n];
    let latlon = centers.iter().map(|c| latlon_of(*c)).collect();

    let mut chunks: Vec<Chunk> = (0..ICO_FACES.len())
        .map(|_| Chunk {
            cells: Vec::new(),
            center: Vec3::Z,
            cos_radius: 1.0,
        })
        .collect();
    for (cell, face) in cell_chunk.iter().enumerate() {
        chunks[*face as usize].cells.push(cell as u32);
    }
    for chunk in &mut chunks {
        let mut sum = Vec3::ZERO;
        for &c in &chunk.cells {
            sum += centers[c as usize];
        }
        chunk.center = sum.normalize_or(Vec3::Z);
        chunk.cos_radius = chunk
            .cells
            .iter()
            .map(|&c| chunk.center.dot(centers[c as usize]))
            .fold(1.0f32, f32::min);
    }

    Mesh {
        generators: Vec::new(),
        level,
        centers,
        areas_km2,
        latlon,
        // Adjacency is unused by the renderer; a valid empty CSR keeps the
        // accessors from panicking.
        neighbor_offsets: vec![0; n + 1],
        neighbors: Vec::new(),
        vertices,
        corner_offsets,
        corners,
        chunks,
        cell_chunk,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cell_count_and_corner_layout() {
        let m = build(2);
        assert_eq!(m.n_cells(), 20 * 4usize.pow(2));
        assert_eq!(m.chunks.len(), 20);
        for c in 0..m.n_cells() as u32 {
            assert_eq!(m.corners_of(c).len(), 3);
        }
    }

    #[test]
    fn centers_and_vertices_are_unit() {
        let m = build(2);
        assert!(m.centers.iter().all(|c| (c.length() - 1.0).abs() < 1e-4));
        assert!(m.vertices.iter().all(|v| (v.length() - 1.0).abs() < 1e-4));
    }

    #[test]
    fn chunk_cones_cover_their_cells() {
        let m = build(3);
        let mut seen = 0;
        for chunk in &m.chunks {
            for &c in &chunk.cells {
                assert!(chunk.center.dot(m.centers[c as usize]) >= chunk.cos_radius - 1e-5);
                seen += 1;
            }
        }
        assert_eq!(seen, m.n_cells());
    }
}
