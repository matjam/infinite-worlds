//! Goldberg-polyhedron planet mesh: the dual of a subdivided icosahedron.
//!
//! Cell count for subdivision `level` is `10 * 4^level + 2`: exactly 12 pentagons
//! (at the icosahedron vertices), hexagons everywhere else, all cells within a
//! factor ~2 of equal area. All per-cell arrays across the project are indexed by
//! cell id (`u32`), in the order of `centers`.
//!
//! API contract is fixed (see IMPLEMENTATION_PLAN.md §2): signatures and field
//! meanings must not change.
//!
//! # Construction
//!
//! The primal grid subdivides each icosahedron face into `n^2 = 4^level` triangles
//! and projects the lattice onto the sphere. Cell `c` is the dual of primal vertex
//! `c`; its corners are the normalized centroids of the triangles incident to that
//! vertex, so corner vertices are shared by exactly three cells and are
//! deduplicated by construction (corner id == primal triangle id). Vertex ids
//! 0..12 are the original icosahedron vertices, which is why the 12 pentagons are
//! always cells 0..12.
//!
//! Everything is indexed arithmetically — no hashing, no sorting by float except
//! within a single cell's 5–6 corners — so `build` is bit-identical across runs,
//! platforms and thread counts.

#![warn(missing_docs)]

use glam::{DVec3, Vec3};
use rayon::prelude::*;

mod icosa;
mod subdiv;

use subdiv::Subdiv;

/// Planet radius used for all area/distance outputs, in kilometres.
pub const EARTH_RADIUS_KM: f32 = 6371.0;
/// Planet radius used for all area/distance outputs, in metres.
pub const EARTH_RADIUS_M: f64 = 6_371_000.0;

/// Highest subdivision level [`Mesh::build`] accepts.
pub const MAX_LEVEL: u8 = 10;

/// Angular slack subtracted from every chunk cone so the containment test is a
/// non-strict inequality even after float rounding.
const CONE_MARGIN: f32 = 1e-6;

/// A render/cull patch: a contiguous-ish group of cells under one bounding cone.
#[derive(Debug, Clone)]
pub struct Chunk {
    /// Member cell ids. `cells[0]` is the member closest to `center` and is used
    /// as the seed for [`Mesh::cell_at`]'s walk; the rest are in ascending order.
    pub cells: Vec<u32>,
    /// Unit vector at the chunk's angular center.
    pub center: Vec3,
    /// Cosine of the cone half-angle that bounds every cell center in the chunk.
    /// The cone is conservative: it also bounds every corner of every member cell.
    pub cos_radius: f32,
}

/// Immutable spherical cell mesh. Built once per subdivision level.
pub struct Mesh {
    /// Subdivision level this mesh was built at.
    pub level: u8,
    /// Unit-sphere direction of each cell center. Length = n_cells.
    pub centers: Vec<Vec3>,
    /// Cell area at Earth radius. Sums to 4*pi*R^2 (within float error).
    pub areas_km2: Vec<f32>,
    /// [latitude_rad, longitude_rad] per cell.
    pub latlon: Vec<[f32; 2]>,
    /// CSR adjacency: neighbors of cell c are
    /// `neighbors[neighbor_offsets[c]..neighbor_offsets[c+1]]`.
    pub neighbor_offsets: Vec<u32>,
    /// CSR adjacency payload; see [`Mesh::neighbor_offsets`]. Neighbor `k` is the
    /// cell across the edge between corner `k` and corner `k+1`.
    pub neighbors: Vec<u32>,
    /// Corner vertex pool (unit vectors) shared between adjacent cells.
    pub vertices: Vec<Vec3>,
    /// CSR corner lists per cell, counter-clockwise seen from outside the sphere.
    pub corner_offsets: Vec<u32>,
    /// CSR corner payload; see [`Mesh::corner_offsets`]. Indexes [`Mesh::vertices`].
    pub corners: Vec<u32>,
    /// Render/cull patches partitioning all cells.
    pub chunks: Vec<Chunk>,
    /// Chunk index of each cell.
    pub cell_chunk: Vec<u16>,
}

impl Mesh {
    /// Number of cells at the given subdivision level: `10 * 4^level + 2`.
    pub fn expected_cells(level: u8) -> usize {
        10 * 4usize.pow(level as u32) + 2
    }

    /// Build the Goldberg mesh for `level` (0..=10). Deterministic.
    ///
    /// # Panics
    /// If `level > 10`.
    pub fn build(level: u8) -> Mesh {
        assert!(
            level <= MAX_LEVEL,
            "subdivision level {level} exceeds MAX_LEVEL {MAX_LEVEL}"
        );
        let n = 1u32 << level;
        let sd = Subdiv::new(n);
        let n_cells = sd.n_verts();

        // Primal grid. Cell centers ARE the primal vertices.
        let centers = sd.vertex_positions();
        let tris = sd.triangles();

        // Dual corner pool: one per primal triangle, so sharing is exact.
        let vertices: Vec<Vec3> = tris
            .par_iter()
            .map(|t| {
                let a = centers[t[0] as usize].as_dvec3();
                let b = centers[t[1] as usize].as_dvec3();
                let c = centers[t[2] as usize].as_dvec3();
                (a + b + c).normalize().as_vec3()
            })
            .collect();

        // Valence is 5 for the 12 icosahedron vertices and 6 everywhere else, so
        // the CSR offsets are closed-form and identical for corners and neighbors.
        let corner_offsets: Vec<u32> = (0..=n_cells)
            .map(|v| {
                if v <= icosa::N_VERTS {
                    5 * v as u32
                } else {
                    60 + 6 * (v as u32 - icosa::N_VERTS as u32)
                }
            })
            .collect();
        let total = *corner_offsets.last().expect("offsets non-empty") as usize;
        debug_assert_eq!(total, 3 * tris.len());

        // Scatter incident triangles into the CSR slots (order still arbitrary).
        let mut corners = vec![0u32; total];
        let mut fill = vec![0u32; n_cells];
        for (t, tv) in tris.iter().enumerate() {
            for v in tv {
                let v = *v as usize;
                let slot = corner_offsets[v] + fill[v];
                corners[slot as usize] = t as u32;
                fill[v] += 1;
            }
        }
        debug_assert!(fill
            .iter()
            .enumerate()
            .all(|(v, f)| *f == corner_offsets[v + 1] - corner_offsets[v]));
        drop(fill);

        // Order each cell's corners CCW seen from outside, then read the neighbor
        // ring off the ordered corners: consecutive incident triangles share an
        // edge, whose far endpoint is the neighboring cell.
        let mut neighbors = vec![0u32; total];
        {
            let (pent_c, hex_c) = corners.split_at_mut(60);
            let (pent_n, hex_n) = neighbors.split_at_mut(60);
            pent_c
                .par_chunks_mut(5)
                .zip(pent_n.par_chunks_mut(5))
                .enumerate()
                .for_each(|(v, (cs, ns))| {
                    finish_cell(v as u32, cs, ns, &centers, &vertices, &tris)
                });
            hex_c
                .par_chunks_mut(6)
                .zip(hex_n.par_chunks_mut(6))
                .enumerate()
                .for_each(|(k, (cs, ns))| {
                    let v = k as u32 + icosa::N_VERTS as u32;
                    finish_cell(v, cs, ns, &centers, &vertices, &tris)
                });
        }
        drop(tris);

        let r2 = (EARTH_RADIUS_KM as f64) * (EARTH_RADIUS_KM as f64);
        let areas_km2: Vec<f32> = (0..n_cells)
            .into_par_iter()
            .map(|v| {
                let a = corner_offsets[v] as usize;
                let b = corner_offsets[v + 1] as usize;
                let cs = &corners[a..b];
                let c = centers[v].as_dvec3();
                let mut excess = 0.0f64;
                for k in 0..cs.len() {
                    let p = vertices[cs[k] as usize].as_dvec3();
                    let q = vertices[cs[(k + 1) % cs.len()] as usize].as_dvec3();
                    excess += spherical_excess(c, p, q);
                }
                (excess * r2) as f32
            })
            .collect();

        let latlon: Vec<[f32; 2]> = centers.par_iter().map(|c| latlon_of(*c)).collect();

        let (chunks, cell_chunk) =
            build_chunks(level, &centers, &corner_offsets, &corners, &vertices);

        Mesh {
            level,
            centers,
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

    /// Number of cells in this mesh.
    #[inline]
    pub fn n_cells(&self) -> usize {
        self.centers.len()
    }

    /// Neighbor cell ids of `cell`, in the same rotational order as its corners.
    #[inline]
    pub fn neighbors_of(&self, cell: u32) -> &[u32] {
        let (a, b) = (
            self.neighbor_offsets[cell as usize] as usize,
            self.neighbor_offsets[cell as usize + 1] as usize,
        );
        &self.neighbors[a..b]
    }

    /// Corner vertex ids of `cell`, counter-clockwise seen from outside.
    #[inline]
    pub fn corners_of(&self, cell: u32) -> &[u32] {
        let (a, b) = (
            self.corner_offsets[cell as usize] as usize,
            self.corner_offsets[cell as usize + 1] as usize,
        );
        &self.corners[a..b]
    }

    /// True for the 12 pentagonal cells (the original icosahedron vertices).
    #[inline]
    pub fn is_pentagon(&self, cell: u32) -> bool {
        self.corners_of(cell).len() == 5
    }

    /// Cell containing the given direction (need not be normalized).
    /// Coarse chunk-cone test, then greedy walk on neighbor angular distance.
    ///
    /// Returns the cell whose center maximizes `dot(dir_hat, center)`. A
    /// zero-length `dir` is treated as `+Z`.
    ///
    /// The walk compares in `f64`: at level 8 and above, adjacent cell centers
    /// differ by only a few ULP of an `f32` dot product, and ties there would
    /// make the result depend on the walk's arrival path rather than on geometry.
    pub fn cell_at(&self, dir: Vec3) -> u32 {
        let d = dir.as_dvec3();
        let d = if d.length_squared() > 0.0 {
            d.normalize()
        } else {
            DVec3::Z
        };

        // Coarse: the chunk whose cone axis is closest to `d`. Linear over <= 80.
        let mut cur = 0u32;
        let mut best_axis = f64::NEG_INFINITY;
        for ch in self.chunks.iter() {
            let s = d.dot(ch.center.as_dvec3());
            if s > best_axis {
                best_axis = s;
                cur = ch.cells[0];
            }
        }

        // Fine: greedy ascent on dot(d, center). The tiling is convex, so the
        // 1-ring step converges; the 2-ring fallback covers the rare non-Delaunay
        // primal edge near an icosahedron seam. `cur_dot` strictly increases, so
        // the loop terminates.
        let mut cur_dot = d.dot(self.centers[cur as usize].as_dvec3());
        loop {
            let mut next = cur;
            let mut next_dot = cur_dot;
            for &m in self.neighbors_of(cur) {
                let s = d.dot(self.centers[m as usize].as_dvec3());
                if s > next_dot {
                    next_dot = s;
                    next = m;
                }
            }
            if next == cur {
                for &m in self.neighbors_of(cur) {
                    for &m2 in self.neighbors_of(m) {
                        let s = d.dot(self.centers[m2 as usize].as_dvec3());
                        if s > next_dot {
                            next_dot = s;
                            next = m2;
                        }
                    }
                }
                if next == cur {
                    return cur;
                }
            }
            cur = next;
            cur_dot = next_dot;
        }
    }

    /// Local tangent basis at a cell center: (east, north) unit vectors.
    /// At the poles falls back to an arbitrary but deterministic basis.
    pub fn east_north(&self, cell: u32) -> (Vec3, Vec3) {
        let r = self.centers[cell as usize];
        let east = Vec3::Z.cross(r);
        let east = if east.length_squared() < 1e-10 {
            Vec3::X
        } else {
            east.normalize()
        };
        // glam is y-up nowhere here: we use z as the north pole axis project-wide.
        let north = r.cross(east).normalize();
        let east = north.cross(r).normalize();
        (east, north)
    }
}

/// Order one cell's incident triangles CCW and derive its neighbor ring.
fn finish_cell(
    v: u32,
    cs: &mut [u32],
    ns: &mut [u32],
    centers: &[Vec3],
    vertices: &[Vec3],
    tris: &[[u32; 3]],
) {
    let c = centers[v as usize];
    // Right-handed tangent frame (t, b, c): increasing atan2(p.b, p.t) is CCW
    // seen from outside the sphere.
    let up = if c.z.abs() < 0.9 { Vec3::Z } else { Vec3::X };
    let t = up.cross(c).normalize();
    let b = c.cross(t);

    let mut keyed: [(f32, u32); 6] = [(0.0, 0); 6];
    for (k, tri) in cs.iter().enumerate() {
        let p = vertices[*tri as usize];
        keyed[k] = ((p.dot(b)).atan2(p.dot(t)), *tri);
    }
    let keyed = &mut keyed[..cs.len()];
    keyed.sort_unstable_by(|a, b| a.0.total_cmp(&b.0));
    for (slot, (_, tri)) in cs.iter_mut().zip(keyed.iter()) {
        *slot = *tri;
    }

    let m = cs.len();
    for k in 0..m {
        let a = &tris[cs[k] as usize];
        let b = &tris[cs[(k + 1) % m] as usize];
        // The two consecutive incident triangles share edge (v, w).
        let mut found = u32::MAX;
        for x in a {
            if *x != v && b.contains(x) {
                found = *x;
                break;
            }
        }
        debug_assert_ne!(found, u32::MAX, "cell {v} corner ring is not a fan");
        ns[k] = found;
    }
}

/// Signed area of the spherical triangle `(a, b, c)` on the unit sphere, in
/// steradians (van Oosterom & Strackee). Positive for CCW winding seen from
/// outside.
#[inline]
fn spherical_excess(a: DVec3, b: DVec3, c: DVec3) -> f64 {
    let num = a.dot(b.cross(c));
    let den = 1.0 + a.dot(b) + b.dot(c) + c.dot(a);
    2.0 * num.atan2(den)
}

/// Group cells into cull patches and fit a conservative bounding cone to each.
///
/// Level <= 7 gives 20 patches (one per icosahedron face); level >= 8 splits each
/// face into its four mid-edge sub-triangles for 80 patches, so culling stays
/// effective when a face holds tens of thousands of cells.
fn build_chunks(
    level: u8,
    centers: &[Vec3],
    corner_offsets: &[u32],
    corners: &[u32],
    vertices: &[Vec3],
) -> (Vec<Chunk>, Vec<u16>) {
    let quarters = level >= 8;
    let fc = icosa::face_centers().map(|v| v.as_vec3());
    let qc = icosa::face_quarter_centers().map(|q| q.map(|v| v.as_vec3()));
    let n_groups = if quarters { 80 } else { 20 };

    let raw: Vec<u16> = centers
        .par_iter()
        .map(|c| {
            let mut f = 0usize;
            let mut best = f32::NEG_INFINITY;
            for (k, v) in fc.iter().enumerate() {
                let s = c.dot(*v);
                if s > best {
                    best = s;
                    f = k;
                }
            }
            if !quarters {
                return f as u16;
            }
            let mut q = 0usize;
            let mut best = f32::NEG_INFINITY;
            for (k, v) in qc[f].iter().enumerate() {
                let s = c.dot(*v);
                if s > best {
                    best = s;
                    q = k;
                }
            }
            (f * 4 + q) as u16
        })
        .collect();

    let mut groups: Vec<Vec<u32>> = vec![Vec::new(); n_groups];
    for (cell, g) in raw.iter().enumerate() {
        groups[*g as usize].push(cell as u32);
    }

    // Coarse levels can leave a group empty (e.g. level 0 has 12 cells / 20
    // faces); drop those and renumber so `cell_chunk` always indexes `chunks`.
    let mut remap = vec![u16::MAX; n_groups];
    let mut chunks: Vec<Chunk> = Vec::new();
    for (g, cells) in groups.into_iter().enumerate() {
        if cells.is_empty() {
            continue;
        }
        remap[g] = chunks.len() as u16;
        chunks.push(fit_chunk(cells, centers, corner_offsets, corners, vertices));
    }
    let cell_chunk: Vec<u16> = raw.iter().map(|g| remap[*g as usize]).collect();
    (chunks, cell_chunk)
}

/// Compute a chunk's cone axis and half-angle, and promote its most central cell
/// to `cells[0]` for use as a walk seed.
fn fit_chunk(
    mut cells: Vec<u32>,
    centers: &[Vec3],
    corner_offsets: &[u32],
    corners: &[u32],
    vertices: &[Vec3],
) -> Chunk {
    let mut sum = DVec3::ZERO;
    for c in cells.iter() {
        sum += centers[*c as usize].as_dvec3();
    }
    let center = if sum.length_squared() > 1e-12 {
        sum.normalize().as_vec3()
    } else {
        centers[cells[0] as usize]
    };

    let mut cos_radius = f32::INFINITY;
    let mut seed = 0usize;
    let mut seed_dot = f32::NEG_INFINITY;
    for (k, c) in cells.iter().enumerate() {
        let d = center.dot(centers[*c as usize]);
        if d > seed_dot {
            seed_dot = d;
            seed = k;
        }
        cos_radius = cos_radius.min(d);
        let (a, b) = (
            corner_offsets[*c as usize] as usize,
            corner_offsets[*c as usize + 1] as usize,
        );
        for v in &corners[a..b] {
            cos_radius = cos_radius.min(center.dot(vertices[*v as usize]));
        }
    }
    cells.swap(0, seed);

    Chunk {
        cells,
        center,
        cos_radius: (cos_radius - CONE_MARGIN).max(-1.0),
    }
}

/// Great-circle distance between two unit vectors, at Earth radius.
#[inline]
pub fn great_circle_km(a: Vec3, b: Vec3) -> f32 {
    a.dot(b).clamp(-1.0, 1.0).acos() * EARTH_RADIUS_KM
}

/// Latitude/longitude (radians) of a unit vector; +z is the north pole,
/// longitude 0 at +x, increasing toward +y.
#[inline]
pub fn latlon_of(v: Vec3) -> [f32; 2] {
    [v.z.clamp(-1.0, 1.0).asin(), v.y.atan2(v.x)]
}
