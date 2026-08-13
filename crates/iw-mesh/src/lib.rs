//! Goldberg-polyhedron planet mesh: the dual of a subdivided icosahedron.
//!
//! Cell count for subdivision `level` is `10 * 4^level + 2`: exactly 12 pentagons
//! (at the icosahedron vertices), hexagons everywhere else, all cells within a
//! factor ~2 of equal area. All per-cell arrays across the project are indexed by
//! cell id (`u32`), in the order of `centers`.
//!
//! API contract is fixed (see IMPLEMENTATION_PLAN.md §2). WP1 fills the `todo!()`
//! bodies; signatures and field meanings must not change.

use glam::Vec3;

pub const EARTH_RADIUS_KM: f32 = 6371.0;
pub const EARTH_RADIUS_M: f64 = 6_371_000.0;

/// A render/cull patch: a contiguous-ish group of cells under one bounding cone.
#[derive(Debug, Clone)]
pub struct Chunk {
    pub cells: Vec<u32>,
    /// Unit vector at the chunk's angular center.
    pub center: Vec3,
    /// Cosine of the cone half-angle that bounds every cell center in the chunk.
    pub cos_radius: f32,
}

/// Immutable spherical cell mesh. Built once per subdivision level.
pub struct Mesh {
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
    pub neighbors: Vec<u32>,
    /// Corner vertex pool (unit vectors) shared between adjacent cells.
    pub vertices: Vec<Vec3>,
    /// CSR corner lists per cell, counter-clockwise seen from outside the sphere.
    pub corner_offsets: Vec<u32>,
    pub corners: Vec<u32>,
    /// Render/cull patches partitioning all cells.
    pub chunks: Vec<Chunk>,
    /// Chunk index of each cell.
    pub cell_chunk: Vec<u16>,
}

impl Mesh {
    pub fn expected_cells(level: u8) -> usize {
        10 * 4usize.pow(level as u32) + 2
    }

    /// Build the Goldberg mesh for `level` (0..=10). Deterministic.
    pub fn build(level: u8) -> Mesh {
        let _ = level;
        todo!("WP1: subdivide icosahedron, dualize, chunk")
    }

    #[inline]
    pub fn n_cells(&self) -> usize {
        self.centers.len()
    }

    #[inline]
    pub fn neighbors_of(&self, cell: u32) -> &[u32] {
        let (a, b) = (
            self.neighbor_offsets[cell as usize] as usize,
            self.neighbor_offsets[cell as usize + 1] as usize,
        );
        &self.neighbors[a..b]
    }

    #[inline]
    pub fn corners_of(&self, cell: u32) -> &[u32] {
        let (a, b) = (
            self.corner_offsets[cell as usize] as usize,
            self.corner_offsets[cell as usize + 1] as usize,
        );
        &self.corners[a..b]
    }

    #[inline]
    pub fn is_pentagon(&self, cell: u32) -> bool {
        self.corners_of(cell).len() == 5
    }

    /// Cell containing the given direction (need not be normalized).
    /// Coarse chunk-cone test, then greedy walk on neighbor angular distance.
    pub fn cell_at(&self, dir: Vec3) -> u32 {
        let _ = dir;
        todo!("WP1: chunk cone prefilter + greedy neighbor descent")
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
