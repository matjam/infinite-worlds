//! Voronoi tessellation assembly: from generators + their spherical Delaunay
//! (the convex hull) to a full [`crate::Mesh`] with the same contract the
//! Goldberg builder produced — CSR adjacency, CCW corner fans, chunks —
//! but over irregular, terrain-sized polygons (docs/voronoi-v2.md).

use glam::{DVec3, Vec3};
use rayon::prelude::*;

use crate::hull::convex_hull;
use crate::{latlon_of, Chunk, Mesh, EARTH_RADIUS_KM};

/// Build the Voronoi mesh for `generators` (unit vectors, ≥ 4, general
/// position). Cell i is generator i's Voronoi region; corners are the
/// circumcenters of its incident Delaunay triangles, CCW from outside;
/// neighbor k sits across the edge between corner k and corner k+1 —
/// exactly the contract the rest of the workspace already consumes.
pub fn build_from_generators(generators: &[DVec3]) -> Mesh {
    let n = generators.len();
    let faces = convex_hull(generators);
    let nf = faces.len();

    // Dual vertices: one per Delaunay triangle (its circumcenter direction).
    let vertices: Vec<Vec3> = faces
        .par_iter()
        .map(|f| {
            let (a, b, c) = (
                generators[f.a as usize],
                generators[f.b as usize],
                generators[f.c as usize],
            );
            let cc = (b - a).cross(c - a);
            let cc = if cc.length_squared() > 1e-30 {
                cc.normalize()
            } else {
                (a + b + c).normalize()
            };
            cc.as_vec3()
        })
        .collect();

    // Incident triangles per generator (counting pass then scatter).
    let mut valence = vec![0u32; n];
    for f in &faces {
        valence[f.a as usize] += 1;
        valence[f.b as usize] += 1;
        valence[f.c as usize] += 1;
    }
    let mut corner_offsets = vec![0u32; n + 1];
    for i in 0..n {
        corner_offsets[i + 1] = corner_offsets[i] + valence[i];
    }
    let total = corner_offsets[n] as usize;
    debug_assert_eq!(total, 3 * nf);
    let mut incident = vec![0u32; total];
    let mut fill = vec![0u32; n];
    for (t, f) in faces.iter().enumerate() {
        for v in [f.a, f.b, f.c] {
            let slot = corner_offsets[v as usize] + fill[v as usize];
            incident[slot as usize] = t as u32;
            fill[v as usize] += 1;
        }
    }

    // Build each cell's corner ring COMBINATORIALLY: rotate every incident
    // triangle so the generator comes first, (g, u, v) — CCW around g means
    // the successor is the triangle whose `u` equals this one's `v`. This is
    // exact for any geometry (an angle sort scrambles near-cocircular
    // circumcenters and corrupts the ring — the confetti bug). The neighbor
    // across the edge between corner k and k+1 is exactly `v` of corner k's
    // triangle, so the CSR contract holds by construction.
    let mut corners = vec![0u32; total];
    let mut neighbors = vec![0u32; total];
    let centers_f32: Vec<Vec3> = generators.iter().map(|g| g.as_vec3()).collect();

    // Split output CSR ranges per cell for safe parallel writes.
    let mut cell_ranges: Vec<(usize, usize)> = Vec::with_capacity(n);
    for i in 0..n {
        cell_ranges.push((corner_offsets[i] as usize, corner_offsets[i + 1] as usize));
    }
    let corners_ptr = SendPtr(corners.as_mut_ptr());
    let neighbors_ptr = SendPtr(neighbors.as_mut_ptr());
    (0..n).into_par_iter().for_each(|g| {
        let (lo, hi) = cell_ranges[g];
        let k = hi - lo;
        // (u, v, tri) per incident triangle, rotated so g comes first.
        let mut ring: smallvec::SmallVec<[(u32, u32, u32); 10]> = smallvec::SmallVec::new();
        for &tri in &incident[lo..hi] {
            let f = faces[tri as usize];
            let (u, v) = if f.a == g as u32 {
                (f.b, f.c)
            } else if f.b == g as u32 {
                (f.c, f.a)
            } else {
                (f.a, f.b)
            };
            ring.push((u, v, tri));
        }
        // Chain: successor of (_, v, _) is the entry whose u == v.
        let mut idx = 0usize;
        for i in 0..k {
            let (_, v, tri) = ring[idx];
            unsafe {
                *corners_ptr.get().add(lo + i) = tri;
                *neighbors_ptr.get().add(lo + i) = v;
            }
            if i + 1 < k {
                idx = ring
                    .iter()
                    .position(|(u, _, _)| *u == v)
                    .expect("open dual ring: hull adjacency is inconsistent");
            }
        }
    });

    // Areas: spherical polygon excess, fanned from the cell center (f64).
    let r2 = (EARTH_RADIUS_KM as f64) * (EARTH_RADIUS_KM as f64);
    let areas_km2: Vec<f32> = (0..n)
        .into_par_iter()
        .map(|g| {
            let (lo, hi) = (corner_offsets[g] as usize, corner_offsets[g + 1] as usize);
            let c = generators[g];
            let mut excess = 0.0f64;
            let k = hi - lo;
            for i in 0..k {
                let p = vertices[corners[lo + i] as usize].as_dvec3();
                let q = vertices[corners[lo + (i + 1) % k] as usize].as_dvec3();
                excess += spherical_triangle_area(c, p, q);
            }
            (excess * r2) as f32
        })
        .collect();

    let latlon: Vec<[f32; 2]> = centers_f32.iter().map(|c| latlon_of(*c)).collect();

    // Chunks: nearest of K spiral seed directions, cos_radius covering all
    // member corners.
    let k_chunks = (n / 8_192).clamp(20, 160);
    let seeds: Vec<Vec3> = fibonacci_dirs(k_chunks);
    let cell_chunk: Vec<u16> = centers_f32
        .par_iter()
        .map(|c| {
            let mut best = 0u16;
            let mut bd = f32::NEG_INFINITY;
            for (i, s) in seeds.iter().enumerate() {
                let d = c.dot(*s);
                if d > bd {
                    bd = d;
                    best = i as u16;
                }
            }
            best
        })
        .collect();
    let mut chunk_cells: Vec<Vec<u32>> = vec![Vec::new(); k_chunks];
    for (c, ch) in cell_chunk.iter().enumerate() {
        chunk_cells[*ch as usize].push(c as u32);
    }
    let mut chunks: Vec<Chunk> = Vec::with_capacity(k_chunks);
    let mut remap = vec![u16::MAX; k_chunks];
    for (i, cells) in chunk_cells.into_iter().enumerate() {
        if cells.is_empty() {
            continue;
        }
        let mut center = Vec3::ZERO;
        for &c in &cells {
            center += centers_f32[c as usize];
        }
        let center = center.normalize_or(Vec3::Z);
        let mut cosr = 1.0f32;
        for &c in &cells {
            cosr = cosr.min(center.dot(centers_f32[c as usize]));
            let (lo, hi) = (
                corner_offsets[c as usize] as usize,
                corner_offsets[c as usize + 1] as usize,
            );
            for k in lo..hi {
                cosr = cosr.min(center.dot(vertices[corners[k] as usize]));
            }
        }
        // Seed cell: closest member to the cone axis, first for cell_at.
        let mut cells = cells;
        let seed_idx = cells
            .iter()
            .enumerate()
            .max_by(|a, b| {
                center
                    .dot(centers_f32[*a.1 as usize])
                    .total_cmp(&center.dot(centers_f32[*b.1 as usize]))
            })
            .map(|(i, _)| i)
            .unwrap_or(0);
        cells.swap(0, seed_idx);
        remap[i] = chunks.len() as u16;
        chunks.push(Chunk {
            cells,
            center,
            cos_radius: cosr - 1e-6,
        });
    }
    let cell_chunk: Vec<u16> = cell_chunk.iter().map(|c| remap[*c as usize]).collect();

    Mesh {
        generators: generators.to_vec(),
        level: 0, // not a subdivision mesh; level is meaningless here
        centers: centers_f32,
        areas_km2,
        latlon,
        neighbor_offsets: corner_offsets.clone(),
        neighbors,
        vertices,
        corner_offsets,
        corners,
        chunks,
        cell_chunk,
    }
}

/// L'Huilier-free signed spherical triangle area (van Oosterom–Strackee).
fn spherical_triangle_area(a: DVec3, b: DVec3, c: DVec3) -> f64 {
    let num = a.dot(b.cross(c));
    let den = 1.0 + a.dot(b) + b.dot(c) + c.dot(a);
    2.0 * num.atan2(den)
}

/// `k` roughly-even directions on the sphere (Fibonacci spiral).
fn fibonacci_dirs(k: usize) -> Vec<Vec3> {
    const GOLDEN_ANGLE: f32 = 2.399_963_2;
    (0..k)
        .map(|i| {
            let z = 1.0 - 2.0 * (i as f32 + 0.5) / k as f32;
            let r = (1.0 - z * z).max(0.0).sqrt();
            let phi = i as f32 * GOLDEN_ANGLE;
            Vec3::new(r * phi.cos(), r * phi.sin(), z)
        })
        .collect()
}

/// Raw-pointer wrapper for disjoint-range parallel writes. Safety: every cell
/// writes only inside its own CSR range `[lo, hi)`; ranges partition the
/// arrays. The accessor exists so closures capture the wrapper (Sync), not
/// the raw pointer field (2021 disjoint capture would otherwise grab `*mut T`
/// directly).
struct SendPtr<T>(*mut T);
unsafe impl<T> Send for SendPtr<T> {}
unsafe impl<T> Sync for SendPtr<T> {}
impl<T> SendPtr<T> {
    fn get(&self) -> *mut T {
        self.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sample::sample_generators;

    fn uniform_mesh(n: usize, seed: u64) -> Mesh {
        let gens = sample_generators(n, seed, &|_| 1.0, 2);
        build_from_generators(&gens)
    }

    #[test]
    fn dual_satisfies_the_mesh_contract() {
        let mesh = uniform_mesh(5_000, 42);
        let n = mesh.n_cells();
        assert_eq!(n, 5_000);
        // Euler for the dual polyhedron: F = n cells, V = 2n - 4 corners,
        // E = 3n - 6 edges (sum of valences = 2E).
        let valence_sum: usize = (0..n as u32).map(|c| mesh.neighbors_of(c).len()).sum();
        assert_eq!(mesh.vertices.len(), 2 * n - 4);
        assert_eq!(valence_sum, 6 * n - 12);
        // Adjacency symmetric; neighbor k shares corners k, k+1.
        for c in 0..n as u32 {
            let nb = mesh.neighbors_of(c);
            let co = mesh.corners_of(c);
            assert_eq!(nb.len(), co.len());
            assert!(nb.len() >= 3, "cell {c} has valence {}", nb.len());
            for (k, &m) in nb.iter().enumerate() {
                assert!(
                    mesh.neighbors_of(m).contains(&c),
                    "adjacency not symmetric: {c} -> {m}"
                );
                let shared = [co[k], co[(k + 1) % co.len()]];
                let mco = mesh.corners_of(m);
                assert!(
                    shared.iter().all(|s| mco.contains(s)),
                    "cell {c} neighbor {k} does not share corners {shared:?}"
                );
            }
        }
    }

    #[test]
    fn areas_sum_to_the_sphere() {
        let mesh = uniform_mesh(3_000, 7);
        let total: f64 = mesh.areas_km2.iter().map(|a| *a as f64).sum();
        let sphere =
            4.0 * std::f64::consts::PI * (EARTH_RADIUS_KM as f64) * (EARTH_RADIUS_KM as f64);
        let err = (total - sphere).abs() / sphere;
        assert!(err < 5e-3, "area sum off by {err}");
    }

    #[test]
    fn cell_at_finds_the_nearest_generator() {
        let mesh = uniform_mesh(2_000, 1337);
        let mut state = 99u64;
        let mut next = || {
            state ^= state >> 12;
            state ^= state << 25;
            state ^= state >> 27;
            state.wrapping_mul(0x2545F4914F6CDD1D)
        };
        for _ in 0..500 {
            let z = (next() as f64 / u64::MAX as f64) * 2.0 - 1.0;
            let phi = (next() as f64 / u64::MAX as f64) * std::f64::consts::TAU;
            let r = (1.0 - z * z).max(0.0).sqrt();
            let dir = Vec3::new((r * phi.cos()) as f32, (r * phi.sin()) as f32, z as f32);
            let found = mesh.cell_at(dir);
            let brute = (0..mesh.n_cells() as u32)
                .max_by(|a, b| {
                    mesh.centers[*a as usize]
                        .dot(dir)
                        .total_cmp(&mesh.centers[*b as usize].dot(dir))
                })
                .expect("cells");
            assert_eq!(found, brute, "cell_at mismatch at {dir:?}");
        }
    }

    #[test]
    fn density_drives_cell_size() {
        let density = |v: Vec3| if v.z > 0.5 { 1.0 } else { 0.06 };
        let gens = sample_generators(4_000, 42, &density, 2);
        let mesh = build_from_generators(&gens);
        let mut dense = Vec::new();
        let mut sparse = Vec::new();
        for c in 0..mesh.n_cells() {
            if mesh.centers[c].z > 0.6 {
                dense.push(mesh.areas_km2[c] as f64);
            } else if mesh.centers[c].z < 0.3 {
                sparse.push(mesh.areas_km2[c] as f64);
            }
        }
        let mean = |v: &[f64]| v.iter().sum::<f64>() / v.len().max(1) as f64;
        let ratio = mean(&sparse) / mean(&dense);
        println!("sparse/dense mean-area ratio: {ratio:.1}");
        assert!(
            ratio > 6.0,
            "terrain-driven size contrast too weak: ratio {ratio:.1}"
        );
    }
}
